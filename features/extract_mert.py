"""Extract frozen MERT-v1-95M embeddings for every DEAM clip.

For each clip, writes cache/<name>/<song_id>.npy with shape
(n_windows, 13, 768), float16:
  - audio is loaded mono and resampled to 24 kHz (MERT's rate);
  - it is cut into non-overlapping 5 s windows (MERT's training context);
    a trailing piece shorter than 5 s is dropped, not padded, so every
    window has the same statistics (a 45 s clip gives 9 windows);
  - all 13 hidden states (CNN output + 12 transformer layers) are taken in
    one forward pass and mean-pooled over time inside each window.
Pooling over windows and choosing a layer happen later, in the model code.

The run is resumable: clips whose .npy already exists are skipped.
cache/<name>/meta.json records the settings.

--random-init builds the same architecture with untrained weights (null
model for H2b, after the progress report). Its weights are seeded.

Usage:
  python features/extract_mert.py --audio data/raw/DEAM_audio \
      --labels data/interim/labels.csv --out cache/mert95m
  python features/extract_mert.py ... --random-init --out cache/mert95m_random
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_ID = "m-a-p/MERT-v1-95M"
SR = 24_000
WIN_SEC = 5.0


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    import librosa  # soundfile backend reads mp3 with libsndfile >= 1.1
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def windows(y: np.ndarray, sr: int = SR, win_sec: float = WIN_SEC) -> np.ndarray:
    """(n_samples,) -> (n_windows, win_len); trailing partial window dropped."""
    win = int(round(sr * win_sec))
    n = len(y) // win
    if n == 0:
        raise ValueError(f"clip shorter than one window ({len(y) / sr:.2f} s)")
    return y[: n * win].reshape(n, win)


def build_model(random_init: bool, seed: int, device: str):
    import torch
    from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if random_init:
        torch.manual_seed(seed)
        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModel.from_config(cfg, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    if processor.sampling_rate != SR:
        raise RuntimeError(f"processor expects {processor.sampling_rate} Hz, script assumes {SR}")
    return processor, model.to(device).eval()


def embed_clip(y: np.ndarray, processor, model, device: str, batch: int) -> np.ndarray:
    import torch

    w = windows(y)
    outs = []
    for i in range(0, len(w), batch):
        inp = processor(list(w[i:i + batch]), sampling_rate=SR, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.inference_mode():
            hs = model(**inp, output_hidden_states=True).hidden_states  # 13 x (B, T, 768)
        h = torch.stack(hs, dim=1).mean(dim=2)                          # (B, 13, 768)
        outs.append(h.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=Path("data/raw/DEAM_audio"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"),
                    help="only song_ids in this file are processed (output of the data audit)")
    ap.add_argument("--out", type=Path, default=Path("cache/mert95m"))
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--seed", type=int, default=415)
    ap.add_argument("--batch", type=int, default=9, help="windows per forward pass")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N clips (smoke test)")
    a = ap.parse_args()

    import pandas as pd
    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = {int(p.stem): p for p in a.audio.rglob("*")
             if p.suffix.lower() in (".mp3", ".wav") and p.stem.isdigit()}
    ids = pd.read_csv(a.labels)["song_id"].astype(int).tolist()
    missing = [i for i in ids if i not in files]
    if missing:
        print(f"WARNING: {len(missing)} labelled clips have no audio, e.g. {missing[:10]}")
    todo = [i for i in ids if i in files and not (a.out / f"{i}.npy").exists()]
    if a.limit:
        todo = todo[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} clips={len(ids)} to_do={len(todo)} out={a.out}")

    processor, model = build_model(a.random_init, a.seed, device)
    meta = {
        "model_id": MODEL_ID, "random_init": a.random_init,
        "seed": a.seed if a.random_init else None,
        "sample_rate": SR, "window_sec": WIN_SEC, "window_overlap": 0,
        "partial_window": "dropped", "layers": "hidden_states[0..12] (13)",
        "within_window_pool": "mean over frames", "dtype": "float16",
        "shape": "(n_windows, 13, 768)", "torch": torch.__version__,
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    t0, failed = time.time(), []
    for k, sid in enumerate(todo, 1):
        try:
            emb = embed_clip(load_audio(files[sid]), processor, model, device, a.batch)
            np.save(a.out / f"{sid}.npy", emb)
        except Exception as e:  # keep going; report at the end
            failed.append((sid, repr(e)))
        if k % 50 == 0 or k == len(todo):
            rate = (time.time() - t0) / k
            print(f"{k}/{len(todo)}  {rate:.2f} s/clip  eta {(len(todo) - k) * rate / 60:.1f} min")
    if failed:
        (a.out / "failed.txt").write_text("\n".join(f"{s}\t{e}" for s, e in failed), encoding="utf-8")
        print(f"{len(failed)} clips failed, see {a.out / 'failed.txt'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
