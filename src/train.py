"""10-fold cross-validation training script for emotion recognition.

Supports:
- Heteroscedastic MLP (Two-stage training: MSE warmup on mu followed by Gaussian NLL with early stopping)
- Deterministic MLP (Trained with MSE loss)
- Ridge Regression baseline (Separate RidgeCV for Valence and Arousal)

Strictly adheres to:
- Official fold-file contract (splits/folds_artist.csv) with 80/10/10 split:
    For test fold k:
      - test: fold == k
      - val:  fold == (k + 1) % 10 (early stopping & validation monitoring)
      - train: other 8 folds
- Fitting StandardScaler strictly on the 8 training folds (no leakage).
- Logging per-epoch validation sigma distribution stats to outputs/sigma_logs_<feature_type>.csv.
- Saving out-of-fold test predictions to:
    - outputs/preds_<feature_type>_<model>.csv
    - results/predictions/<model>__artist__seed<seed>.csv (with json sidecar)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from splits.make_folds import N_SPLITS, fold_roles
from src.data import load_dataset
from src.models import (
    DeterministicMLP,
    HeteroscedasticMLP,
    RidgeBaseline,
    gaussian_nll_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fold_roles_for(fold: pd.Series | np.ndarray, k: int, n_splits: int = N_SPLITS) -> np.ndarray:
    """Role of each clip when fold k is the test fold: 'train' (8 folds), 'val' (1 fold), 'test' (1 fold).

    Thin wrapper over the contract helper in splits/make_folds.py, which is the
    single place the (k + 1) % 10 rule is defined.
    """
    return fold_roles(fold, k, n_splits=n_splits)


def train_epoch_heteroscedastic(
    model: HeteroscedasticMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: str = "warmup",  # "warmup" or "nll"
    beta: float = 0.0,
) -> float:
    """Train HeteroscedasticMLP for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        mu, sigma = model(batch_x)

        if stage == "warmup":
            # Stage 1: Warmup on mu only with MSE
            loss = F.mse_loss(mu, batch_y)
        else:
            # Stage 2: Gaussian NLL
            loss = gaussian_nll_loss(mu, sigma, batch_y, reduction="mean", beta=beta)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / max(1, total_samples)


NEAR_BOUND_FRAC = 0.05  # "near a bound" = within 5 % of the sigma range of it
UNSTABLE_SHARE = 0.05   # more than 5 % of sigmas near a bound -> flagged unstable


def near_bound_share(sigma: np.ndarray, smin: np.ndarray, smax: np.ndarray) -> Dict[str, float]:
    """Share of predicted sigmas within NEAR_BOUND_FRAC of the range of either bound (per dim and overall)."""
    rng = smax - smin
    near = (sigma <= smin + NEAR_BOUND_FRAC * rng) | (sigma >= smax - NEAR_BOUND_FRAC * rng)
    return {
        "near_bound_v": float(near[:, 0].mean()),
        "near_bound_a": float(near[:, 1].mean()),
        "near_bound_all": float(near.mean()),
    }


@torch.no_grad()
def evaluate_heteroscedastic(
    model: HeteroscedasticMLP,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate HeteroscedasticMLP on validation set.

    Returns:
        val_nll, val_mse, sigma_stats
    """
    model.eval()
    total_nll = 0.0
    total_mse = 0.0
    total_samples = 0

    all_sigmas = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        mu, sigma = model(batch_x)
        nll = gaussian_nll_loss(mu, sigma, batch_y, reduction="sum")
        mse = F.mse_loss(mu, batch_y, reduction="sum")

        total_nll += nll.item()
        total_mse += mse.item()
        total_samples += batch_x.size(0)

        all_sigmas.append(sigma.cpu().numpy())

    avg_nll = total_nll / (total_samples * 2.0)  # Average per dimension
    avg_mse = total_mse / (total_samples * 2.0)

    sigmas_arr = np.concatenate(all_sigmas, axis=0)  # Shape (N, 2)
    near = near_bound_share(sigmas_arr, model.sigma_min.cpu().numpy(), model.sigma_max.cpu().numpy())
    sig_v = sigmas_arr[:, 0]
    sig_a = sigmas_arr[:, 1]
    sig_all = sigmas_arr.flatten()

    stats = {
        "sigma_v_mean": float(np.mean(sig_v)),
        "sigma_v_std": float(np.std(sig_v)),
        "sigma_v_min": float(np.min(sig_v)),
        "sigma_v_max": float(np.max(sig_v)),
        "sigma_a_mean": float(np.mean(sig_a)),
        "sigma_a_std": float(np.std(sig_a)),
        "sigma_a_min": float(np.min(sig_a)),
        "sigma_a_max": float(np.max(sig_a)),
        "sigma_overall_mean": float(np.mean(sig_all)),
        "sigma_overall_std": float(np.std(sig_all)),
        "sigma_overall_min": float(np.min(sig_all)),
        "sigma_overall_max": float(np.max(sig_all)),
        **near,
    }

    return avg_nll, avg_mse, stats


def train_fold_heteroscedastic(
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    epochs_warmup: int = 15,
    epochs_nll: int = 60,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    beta: float = 0.5,
    sigma_min: float = 0.01,
    sigma_max_factor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]], Dict[str, float]]:
    """Train HeteroscedasticMLP on 8 training folds, validate on 1 fold, evaluate on test fold.

    sigma is bounded to [sigma_min, sigma_max_factor * sd(y_train)] per dimension.
    Stage 2 minimises the beta-NLL; early stopping and model selection use the plain
    validation NLL (beta = 0). Returns (mu, sigma, per-epoch logs, final stability stats).
    """
    if X_test is None:
        X_test = X_val

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    in_features = X_train.shape[1]
    sigma_max = sigma_max_factor * y_train.std(axis=0)
    model = HeteroscedasticMLP(
        in_features=in_features, hidden_dim=hidden_dim, dropout=dropout,
        sigma_min=sigma_min, sigma_max=sigma_max,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    logs: List[Dict[str, float]] = []

    # Stage 1: Warmup (MSE on mu only)
    logger.info("Fold %d: Stage 1 Warmup (%d epochs)", fold, epochs_warmup)
    for epoch in range(1, epochs_warmup + 1):
        train_loss = train_epoch_heteroscedastic(model, train_loader, optimizer, device, stage="warmup")
        val_nll, val_mse, sigma_stats = evaluate_heteroscedastic(model, val_loader, device)

        log_entry = {
            "fold": fold,
            "epoch": epoch,
            "stage": "warmup",
            "train_loss": train_loss,
            "val_loss": val_mse,
            "val_nll": val_nll,
            "val_mse": val_mse,
            **sigma_stats,
        }
        logs.append(log_entry)

    # Stage 2: Gaussian NLL with Early Stopping on validation fold
    logger.info("Fold %d: Stage 2 Gaussian NLL (up to %d epochs, patience=%d)", fold, epochs_nll, patience)
    best_val_nll = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs_warmup + 1, epochs_warmup + epochs_nll + 1):
        train_loss = train_epoch_heteroscedastic(model, train_loader, optimizer, device, stage="nll", beta=beta)
        val_nll, val_mse, sigma_stats = evaluate_heteroscedastic(model, val_loader, device)

        log_entry = {
            "fold": fold,
            "epoch": epoch,
            "stage": "nll",
            "train_loss": train_loss,
            "val_loss": val_nll,
            "val_nll": val_nll,
            "val_mse": val_mse,
            **sigma_stats,
        }
        logs.append(log_entry)

        if val_nll < best_val_nll - 1e-4:
            best_val_nll = val_nll
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Fold %d: Early stopping triggered at epoch %d (Best Val NLL: %.4f)", fold, epoch, best_val_nll)
                break

    # Restore best model
    model.load_state_dict(best_weights)
    model.eval()

    # Out-of-fold inference on test fold
    with torch.no_grad():
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_mu, pred_sigma = model(x_test_tensor)
        oof_mu = pred_mu.cpu().numpy()
        oof_sigma = pred_sigma.cpu().numpy()

    final = {
        "fold": fold,
        "best_val_nll": best_val_nll,
        "stopped_epoch": logs[-1]["epoch"],
        "sigma_min": sigma_min,
        "sigma_max_v": float(sigma_max[0]),
        "sigma_max_a": float(sigma_max[1]),
        **near_bound_share(oof_sigma, np.full(2, sigma_min), sigma_max),
    }
    final["unstable"] = bool(final["near_bound_all"] > UNSTABLE_SHARE)
    if final["unstable"]:
        logger.warning("Fold %d: %.1f %% of test sigmas near a bound (> %.0f %%): flagged unstable",
                       fold, 100 * final["near_bound_all"], 100 * UNSTABLE_SHARE)
    return oof_mu, oof_sigma, logs, final


def train_fold_deterministic_mlp(
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    epochs: int = 60,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Train DeterministicMLP on 8 training folds, early-stop on val fold, evaluate on test fold."""
    if X_test is None:
        X_test = X_val

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    in_features = X_train.shape[1]
    model = DeterministicMLP(in_features=in_features, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mse = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = F.mse_loss(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                val_mse += F.mse_loss(preds, by, reduction="sum").item()
        val_mse /= (len(val_dataset) * 2.0)

        logs.append({"fold": fold, "epoch": epoch, "stage": "mse", "train_loss": train_loss, "val_loss": val_mse})

        if val_mse < best_val_mse - 1e-4:
            best_val_mse = val_mse
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_mu = model(x_test_tensor).cpu().numpy()

    # Estimate training residual std as constant sigma baseline
    with torch.no_grad():
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        train_preds = model(x_train_tensor).cpu().numpy()
        residual_std = np.std(y_train - train_preds, axis=0)
    pred_sigma = np.tile(residual_std, (len(X_test), 1))

    return pred_mu, pred_sigma, logs


def run_cross_validation(
    model_type: str = "heteroscedastic_mlp",
    feature_type: str = "opensmile",
    data_dir: str = "data",
    splits_path: Optional[str] = None,
    epochs_warmup: int = 15,
    epochs_nll: int = 60,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    seed: int = 0,
    device_name: str = "auto",
    mert_dir: str = "cache/mert95m",
    mert_layer: int = 7,
    mert_pool: str = "mean",
    beta: float = 0.5,
) -> pd.DataFrame:
    """Run full 10-fold cross-validation pipeline with 80/10/10 split."""
    set_seed(seed)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info("Using device: %s", device)

    # 1. Load dataset and splits
    data = load_dataset(
        feature_type=feature_type,
        data_dir=data_dir,
        splits_path=splits_path,
        mert_dir=mert_dir,
        mert_layer=mert_layer,
        mert_pool=mert_pool,
    )
    X = data["X"]
    y_mean = data["y_mean"]  # (N, 2)
    song_ids = data["song_ids"]
    splits_df = data["splits"]

    if splits_df is None:
        raise FileNotFoundError(
            "Splits file not found! Expected 'splits/folds_artist.csv'."
        )

    # Merge fold assignments with song_ids
    df_merged = pd.DataFrame({"song_id": song_ids, "idx": np.arange(len(song_ids))})
    df_merged = df_merged.merge(splits_df[["song_id", "fold"]], on="song_id", how="left")

    if df_merged["fold"].isnull().any():
        missing_count = df_merged["fold"].isnull().sum()
        raise ValueError(f"{missing_count} songs have no fold assignment in splits file!")

    folds = sorted(df_merged["fold"].unique().astype(int))
    logger.info("Found %d folds: %s with total %d samples", len(folds), folds, len(song_ids))

    oof_preds_list = []
    all_sigma_logs = []
    stability_rows = []
    # Contract model name, e.g. opensmile_hetero_b0.5 (beta in the name so runs do not overwrite each other)
    if model_type == "heteroscedastic_mlp":
        run_tag = f"hetero_b{beta:g}"
        file_tag = f"heteroscedastic_mlp_b{beta:g}"
    else:
        run_tag = file_tag = model_type
    model_name = f"{feature_type}_{run_tag}"

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)

    for fold in folds:
        logger.info("================== Processing Test Fold %d / %d ==================", fold, len(folds) - 1)
        # Apply official 80/10/10 contract: train (8 folds), val (1 fold: (k+1)%10), test (1 fold: k)
        roles = fold_roles_for(df_merged["fold"].to_numpy(), fold, n_splits=len(folds))
        train_mask = (roles == "train")
        val_mask = (roles == "val")
        test_mask = (roles == "test")

        test_song_ids = df_merged.loc[test_mask, "song_id"].to_numpy()

        X_train_raw = X[train_mask]
        y_train = y_mean[train_mask]
        X_val_raw = X[val_mask]
        y_val = y_mean[val_mask]
        X_test_raw = X[test_mask]
        y_test = y_mean[test_mask]

        # Fit StandardScaler strictly inside the 8 training folds
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        if model_type == "heteroscedastic_mlp":
            oof_mu, oof_sigma, logs, final = train_fold_heteroscedastic(
                fold=fold,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epochs_warmup=epochs_warmup,
                epochs_nll=epochs_nll,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                device=device,
                beta=beta,
            )
            all_sigma_logs.extend(logs)
            stability_rows.append(final)

        elif model_type == "deterministic_mlp":
            oof_mu, oof_sigma, logs = train_fold_deterministic_mlp(
                fold=fold,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epochs=epochs_warmup + epochs_nll,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                device=device,
            )

        elif model_type == "ridge":
            ridge = RidgeBaseline()
            ridge.fit(X_train, y_train)
            oof_mu, oof_sigma = ridge.predict(X_test)


        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'heteroscedastic_mlp', 'deterministic_mlp', or 'ridge'.")

        fold_preds = pd.DataFrame({
            "song_id": test_song_ids,
            "fold": fold,
            "mu_v": oof_mu[:, 0],
            "mu_a": oof_mu[:, 1],
            "sigma_v": oof_sigma[:, 0],
            "sigma_a": oof_sigma[:, 1],
        })
        oof_preds_list.append(fold_preds)

    # Save out-of-fold predictions
    oof_df = pd.concat(oof_preds_list, ignore_index=True)
    oof_df = oof_df.sort_values("song_id").reset_index(drop=True)

    # 1. Save to outputs/ format
    preds_out_path = os.path.join("outputs", f"preds_{feature_type}_{file_tag}.csv")
    oof_df.to_csv(preds_out_path, index=False)
    logger.info("Saved out-of-fold predictions to: %s (Shape: %s)", preds_out_path, oof_df.shape)

    # 2. Save to team contract results/predictions/ format
    contract_preds_path = os.path.join("results", "predictions", f"{model_name}__artist__seed{seed}.csv")
    contract_df = pd.DataFrame({
        "fold_id": oof_df["fold"],
        "split_type": "artist",
        "seed": seed,
        "model": model_name,
        "song_id": oof_df["song_id"],
        "mu_V": oof_df["mu_v"],
        "mu_A": oof_df["mu_a"],
        "sigma_V": oof_df["sigma_v"],
        "sigma_A": oof_df["sigma_a"],
    })
    contract_df.to_csv(contract_preds_path, index=False)
    logger.info("Saved contract-format predictions to: %s", contract_preds_path)

    # Save sidecar JSON
    sidecar_path = os.path.join("results", "predictions", f"{model_name}__artist__seed{seed}.json")
    sidecar_meta = {
        "model": model_name,
        "feature_type": feature_type,
        "split_type": "artist",
        "seed": seed,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
    }
    if feature_type == "mert":
        sidecar_meta.update(
            {"mert_dir": mert_dir, "mert_layer": mert_layer, "mert_pool": mert_pool}
        )
    if model_type in ("heteroscedastic_mlp", "deterministic_mlp"):
        sidecar_meta.update(
            {
                "epochs_warmup": epochs_warmup,
                "epochs_nll": epochs_nll,
                "patience": patience,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
            }
        )
    if stability_rows:
        sidecar_meta.update(
            {
                "beta": beta,
                "sigma_bounds": "sigma_min=0.01, sigma_max=2*sd(y_train) per dimension",
                "early_stopping": "plain validation NLL (beta=0)",
                "near_bound_definition": f"within {NEAR_BOUND_FRAC:.0%} of the sigma range of either bound",
                "unstable_threshold": UNSTABLE_SHARE,
                "unstable_folds": [r["fold"] for r in stability_rows if r["unstable"]],
                "near_bound_share_mean": float(np.mean([r["near_bound_all"] for r in stability_rows])),
            }
        )
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_meta, f, indent=2)

    # Save sigma logs if heteroscedastic
    if all_sigma_logs:
        sigma_logs_df = pd.DataFrame(all_sigma_logs)
        logs_out_path = os.path.join("outputs", f"sigma_logs_{feature_type}_{file_tag}.csv")
        sigma_logs_df.to_csv(logs_out_path, index=False)
        logger.info("Saved sigma logs to: %s (Entries: %d)", logs_out_path, len(sigma_logs_df))
        stab_path = os.path.join("outputs", f"sigma_stability_{feature_type}_{file_tag}.csv")
        pd.DataFrame(stability_rows).to_csv(stab_path, index=False)
        logger.info("Saved per-fold sigma stability to: %s", stab_path)

    return oof_df


def main():
    parser = argparse.ArgumentParser(description="10-Fold Cross-Validation Emotion Training.")
    parser.add_argument("--model", default="heteroscedastic_mlp", choices=["heteroscedastic_mlp", "deterministic_mlp", "ridge"], help="Model type")
    parser.add_argument("--features", default="opensmile", choices=["opensmile", "mert"], help="Feature representation")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--splits", default=None, help="Path to splits file (default: splits/folds_artist.csv)")
    parser.add_argument("--epochs_warmup", type=int, default=15, help="Warmup epochs on MSE (default: 15)")
    parser.add_argument("--epochs_nll", type=int, default=60, help="NLL epochs (default: 60)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension (default: 128)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0, matches results/predictions naming)")
    parser.add_argument("--beta", type=float, default=0.5, help="beta of the beta-NLL (0 = plain NLL; default 0.5)")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--mert-dir", default="cache/mert95m", help="MERT cache dir (--features mert)")
    parser.add_argument("--mert-layer", type=int, default=7, help="MERT hidden state 0-12 to use (--features mert)")
    parser.add_argument("--mert-pool", default="mean", choices=["mean", "std"], help="Window pooling (--features mert)")
    args = parser.parse_args()

    run_cross_validation(
        model_type=args.model,
        feature_type=args.features,
        data_dir=args.data_dir,
        splits_path=args.splits,
        epochs_warmup=args.epochs_warmup,
        epochs_nll=args.epochs_nll,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
        device_name=args.device,
        mert_dir=args.mert_dir,
        mert_layer=args.mert_layer,
        mert_pool=args.mert_pool,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
