"""Data preprocessing, feature aggregation, and dataset loading pipeline.

Handles:
1. Static annotation loading, column cleaning, and linear rescaling to [-1, 1].
2. OpenSMILE frame-level acoustic feature aggregation across time (mean & std).
3. Modular loader for acoustic (openSMILE) and self-supervised (MERT) representations.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_static_annotations_file(data_dir: str = "data") -> str:
    """Find the static annotations CSV file in data_dir."""
    candidates = [
        os.path.join(
            data_dir,
            "DEAM_Annotations",
            "annotations",
            "annotations averaged per song",
            "song_level",
            "static_annotations_averaged_songs_1_2000.csv",
        ),
        os.path.join(data_dir, "static_annotations_averaged_songs_1_2000.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Fallback glob
    matches = glob.glob(os.path.join(data_dir, "**", "*static_annotations_averaged_songs_1_2000.csv"), recursive=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate static_annotations_averaged_songs_1_2000.csv inside '{data_dir}'"
    )


def find_opensmile_feature_dir(data_dir: str = "data") -> str:
    """Find the directory containing per-song openSMILE CSV feature files."""
    candidates = [
        os.path.join(data_dir, "features", "features"),
        os.path.join(data_dir, "features"),
    ]
    for c in candidates:
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.csv"))) > 100:
            return c

    matches = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    # Group by directory and find one with hundreds of numeric CSVs
    dirs = {}
    for m in matches:
        base = os.path.basename(m)
        name, _ = os.path.splitext(base)
        if name.isdigit():
            d = os.path.dirname(m)
            dirs[d] = dirs.get(d, 0) + 1

    if dirs:
        best_dir = max(dirs.items(), key=lambda kv: kv[1])[0]
        return best_dir

    raise FileNotFoundError(f"Could not locate directory containing openSMILE CSV files in '{data_dir}'")


def preprocess_static_targets(
    annotations_path: Optional[str] = None,
    output_path: str = "data/processed_targets.csv",
    data_dir: str = "data",
) -> pd.DataFrame:
    """Load static annotations for the 1,744 45-second clips and linearly rescale.

    Rescaling:
        valence_mean -> (y - 5.0) / 4.0  (maps [1, 9] to [-1, 1])
        arousal_mean -> (y - 5.0) / 4.0  (maps [1, 9] to [-1, 1])
        valence_std  -> y / 4.0
        arousal_std  -> y / 4.0
    """
    if annotations_path is None:
        annotations_path = find_static_annotations_file(data_dir)

    logger.info("Loading static annotations from: %s", annotations_path)
    df = pd.read_csv(annotations_path)

    # Clean whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    expected_cols = ["song_id", "valence_mean", "arousal_mean", "valence_std", "arousal_std"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {annotations_path}. Found: {df.columns.tolist()}")

    df = df[expected_cols].copy()
    df["song_id"] = df["song_id"].astype(int)

    # Apply specified linear transformations
    df["valence_mean"] = (df["valence_mean"] - 5.0) / 4.0
    df["arousal_mean"] = (df["arousal_mean"] - 5.0) / 4.0
    df["valence_std"] = df["valence_std"] / 4.0
    df["arousal_std"] = df["arousal_std"] / 4.0

    df = df.sort_values("song_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Processed targets saved to %s (Shape: %s)", output_path, df.shape)

    return df


def _aggregate_single_opensmile_file(
    file_path: str,
) -> Optional[Tuple[int, np.ndarray, np.ndarray, List[str]]]:
    """Process a single openSMILE CSV file with polars for high throughput."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        song_id = int(base_name)
    except ValueError:
        return None

    try:
        df = pl.read_csv(file_path, separator=";")
    except Exception as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return None

    feature_cols = [c for c in df.columns if c != "frameTime"]

    # Compute mean and standard deviation per column
    means = df.select([pl.col(c).mean() for c in feature_cols]).to_numpy()[0]
    stds = df.select([pl.col(c).std() for c in feature_cols]).to_numpy()[0]

    # Handle zero standard deviation or rare cases
    stds = np.nan_to_num(stds, nan=0.0)
    means = np.nan_to_num(means, nan=0.0)

    return song_id, means, stds, feature_cols


def aggregate_opensmile_features(
    feature_dir: Optional[str] = None,
    output_path: str = "data/opensmile_aggregated.parquet",
    target_song_ids: Optional[List[int]] = None,
    data_dir: str = "data",
    max_workers: int = 8,
) -> pd.DataFrame:
    """Aggregate per-song frame-level openSMILE CSVs across time using mean and std.

    Results in 260 * 2 = 520 features per song.
    """
    if feature_dir is None:
        feature_dir = find_opensmile_feature_dir(data_dir)

    logger.info("Aggregating openSMILE features from: %s", feature_dir)
    all_files = glob.glob(os.path.join(feature_dir, "*.csv"))

    if target_song_ids is not None:
        target_set = set(target_song_ids)
        file_map = {}
        for f in all_files:
            b = os.path.splitext(os.path.basename(f))[0]
            if b.isdigit() and int(b) in target_set:
                file_map[int(b)] = f
        files_to_process = [file_map[sid] for sid in target_song_ids if sid in file_map]
        logger.info(
            "Found %d of %d requested song files in %s",
            len(files_to_process),
            len(target_song_ids),
            feature_dir,
        )
    else:
        files_to_process = all_files
        logger.info("Processing all %d CSV files found in %s", len(files_to_process), feature_dir)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_aggregate_single_opensmile_file, files_to_process):
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: x[0])  # Sort by song_id

    song_ids = [r[0] for r in results]
    feature_cols = results[0][3]

    mean_col_names = [f"{col}_mean" for col in feature_cols]
    std_col_names = [f"{col}_std" for col in feature_cols]
    all_col_names = mean_col_names + std_col_names

    matrix_rows = []
    for r in results:
        # Concatenate mean and std
        vector = np.concatenate([r[1], r[2]])
        matrix_rows.append(vector)

    feat_matrix = np.vstack(matrix_rows)

    df_out = pd.DataFrame(feat_matrix, columns=all_col_names)
    df_out.insert(0, "song_id", song_ids)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    logger.info(
        "openSMILE features aggregated and cached to %s (Shape: %s, Features: %d)",
        output_path,
        df_out.shape,
        len(all_col_names),
    )

    return df_out


def load_mert_features(
    song_ids: List[int],
    mert_dir: str = "data/mert_cache",
) -> Tuple[np.ndarray, List[str]]:
    """Modular loader interface for cached MERT embeddings.

    Supports:
    1. Consolidated file: data/mert_cache/mert_embeddings.parquet / .pt / .npy
    2. Per-song cache: data/mert_cache/{song_id}.pt or {song_id}.npy
    """
    logger.info("Loading MERT features for %d songs from %s", len(song_ids), mert_dir)

    # Check for consolidated parquet
    parquet_path = os.path.join(mert_dir, "mert_embeddings.parquet")
    if os.path.isfile(parquet_path):
        df_mert = pd.read_parquet(parquet_path)
        if "song_id" in df_mert.columns:
            df_mert = df_mert.set_index("song_id")
        df_mert = df_mert.loc[song_ids]
        feature_names = df_mert.columns.tolist()
        return df_mert.to_numpy(dtype=np.float32), feature_names

    # Check for consolidated PyTorch / NumPy file
    pt_path = os.path.join(mert_dir, "mert_embeddings.pt")
    if os.path.isfile(pt_path):
        import torch

        data = torch.load(pt_path, map_location="cpu")
        if isinstance(data, dict):
            # Keyed by song_id
            vectors = []
            for sid in song_ids:
                if sid not in data and str(sid) not in data:
                    raise KeyError(f"Song ID {sid} missing from {pt_path}")
                val = data.get(sid, data.get(str(sid)))
                if hasattr(val, "numpy"):
                    val = val.numpy()
                vectors.append(val.flatten())
            X = np.vstack(vectors).astype(np.float32)
            feature_names = [f"mert_dim_{i}" for i in range(X.shape[1])]
            return X, feature_names

    # Check for per-song files
    if os.path.isdir(mert_dir):
        vectors = []
        dim = None
        for sid in song_ids:
            pt_f = os.path.join(mert_dir, f"{sid}.pt")
            npy_f = os.path.join(mert_dir, f"{sid}.npy")
            if os.path.isfile(pt_f):
                import torch

                t = torch.load(pt_f, map_location="cpu")
                vec = t.numpy().flatten()
            elif os.path.isfile(npy_f):
                vec = np.load(npy_f).flatten()
            else:
                raise FileNotFoundError(
                    f"MERT representation for song_id {sid} not found in '{mert_dir}' (checked {sid}.pt, {sid}.npy)"
                )
            if dim is None:
                dim = len(vec)
            vectors.append(vec)

        X = np.vstack(vectors).astype(np.float32)
        feature_names = [f"mert_dim_{i}" for i in range(X.shape[1])]
        return X, feature_names

    raise FileNotFoundError(
        f"MERT feature directory or cache '{mert_dir}' not found. Please provide cached embeddings."
    )


def load_dataset(
    feature_type: str = "opensmile",
    data_dir: str = "data",
    splits_path: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, List[str], pd.DataFrame]]:
    """Unified loader providing aligned feature matrix X, targets y_mean, y_std, and song_ids.

    Returns:
        dict with keys:
            - 'song_ids': ndarray (N,)
            - 'X': ndarray (N, D)
            - 'y_mean': ndarray (N, 2)  [valence_mean, arousal_mean]
            - 'y_std': ndarray (N, 2)   [valence_std, arousal_std]
            - 'feature_names': list of str
            - 'splits': pd.DataFrame with song_id and fold (if splits_path provided or found)
            - 'df_targets': pd.DataFrame
    """
    targets_path = os.path.join(data_dir, "processed_targets.csv")
    if not os.path.isfile(targets_path):
        logger.info("%s not found; generating targets...", targets_path)
        preprocess_static_targets(output_path=targets_path, data_dir=data_dir)

    df_targets = pd.read_csv(targets_path)
    target_song_ids = df_targets["song_id"].tolist()

    if feature_type.lower() == "opensmile":
        parquet_path = os.path.join(data_dir, "opensmile_aggregated.parquet")
        if not os.path.isfile(parquet_path):
            logger.info("%s not found; aggregating openSMILE features...", parquet_path)
            aggregate_opensmile_features(
                output_path=parquet_path,
                target_song_ids=target_song_ids,
                data_dir=data_dir,
            )

        df_feat = pd.read_parquet(parquet_path)
        # Ensure alignment with targets
        df_feat = df_feat.set_index("song_id").loc[target_song_ids].reset_index()
        feature_cols = [c for c in df_feat.columns if c != "song_id"]
        X = df_feat[feature_cols].to_numpy(dtype=np.float32)
        feature_names = feature_cols

    elif feature_type.lower() == "mert":
        mert_dir = os.path.join(data_dir, "mert_cache")
        X, feature_names = load_mert_features(target_song_ids, mert_dir=mert_dir)

    else:
        raise ValueError(f"Unknown feature_type: '{feature_type}'. Choose 'opensmile' or 'mert'.")

    y_mean = df_targets[["valence_mean", "arousal_mean"]].to_numpy(dtype=np.float32)
    y_std = df_targets[["valence_std", "arousal_std"]].to_numpy(dtype=np.float32)
    song_ids = df_targets["song_id"].to_numpy(dtype=np.int64)

    # Look for splits.csv
    splits_df = None
    splits_candidates = [
        splits_path,
        os.path.join(data_dir, "splits.csv"),
        "splits.csv",
    ]
    for sp in splits_candidates:
        if sp and os.path.isfile(sp):
            splits_df = pd.read_csv(sp)
            splits_df["song_id"] = splits_df["song_id"].astype(int)
            logger.info("Loaded splits from: %s", sp)
            break

    return {
        "song_ids": song_ids,
        "X": X,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_names": feature_names,
        "df_targets": df_targets,
        "splits": splits_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess static targets and aggregate openSMILE features.")
    parser.add_argument("--data-dir", default="data", help="Root data directory (default: 'data')")
    parser.add_argument("--force", action="store_true", help="Force re-generation of targets and features")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for aggregation")
    args = parser.parse_args()

    targets_path = os.path.join(args.data_dir, "processed_targets.csv")
    if args.force or not os.path.isfile(targets_path):
        df_targets = preprocess_static_targets(output_path=targets_path, data_dir=args.data_dir)
    else:
        df_targets = pd.read_csv(targets_path)
        logger.info("Existing targets loaded from %s (Shape: %s)", targets_path, df_targets.shape)

    features_path = os.path.join(args.data_dir, "opensmile_aggregated.parquet")
    if args.force or not os.path.isfile(features_path):
        df_features = aggregate_opensmile_features(
            output_path=features_path,
            target_song_ids=df_targets["song_id"].tolist(),
            data_dir=args.data_dir,
            max_workers=args.workers,
        )
    else:
        df_features = pd.read_parquet(features_path)
        logger.info("Existing features loaded from %s (Shape: %s)", features_path, df_features.shape)

    # Validation
    X = df_features[[c for c in df_features.columns if c != "song_id"]].to_numpy()
    y = df_targets[["valence_mean", "arousal_mean"]].to_numpy()

    nan_count_X = np.isnan(X).sum()
    inf_count_X = np.isinf(X).sum()
    nan_count_y = np.isnan(y).sum()
    inf_count_y = np.isinf(y).sum()

    logger.info("================ DATA PIPELINE VERIFICATION ================")
    logger.info("Targets shape: %s (Expected: (1744, 5))", df_targets.shape)
    logger.info("Features shape: %s (Expected: (1744, 521))", df_features.shape)
    logger.info("Feature Matrix X shape: %s (Expected: (1744, 520))", X.shape)
    logger.info("X NaNs: %d | Infs: %d", nan_count_X, inf_count_X)
    logger.info("y NaNs: %d | Infs: %d", nan_count_y, inf_count_y)
    logger.info("Valence mean range: [%.4f, %.4f]", y[:, 0].min(), y[:, 0].max())
    logger.info("Arousal mean range: [%.4f, %.4f]", y[:, 1].min(), y[:, 1].max())

    assert df_targets.shape == (1744, 5), f"Targets shape mismatch: {df_targets.shape}"
    assert df_features.shape == (1744, 521), f"Features shape mismatch: {df_features.shape}"
    assert nan_count_X == 0 and inf_count_X == 0, "Feature matrix contains NaN or Inf!"
    assert nan_count_y == 0 and inf_count_y == 0, "Targets contain NaN or Inf!"
    logger.info("ALL VERIFICATIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
