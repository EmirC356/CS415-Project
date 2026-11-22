"""Aggregate DEAM's frame-level openSMILE features into one vector per clip.

DEAM ships one CSV per song (``;``-separated) with 260 low-level descriptors
computed every 500 ms. We summarise each descriptor over the whole 45 s clip by
its mean and standard deviation, giving a 520-dim vector per clip.

Output (embedding-cache contract): ``cache/opensmile/clip_features.parquet``
with index ``song_id`` and columns ``<descriptor>__mean`` / ``<descriptor>__std``,
plus ``cache/opensmile/meta.json``.

    python features/aggregate_opensmile.py --features data/raw/features/features \
        --labels data/interim/labels.csv --out cache/opensmile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLIP_SECONDS = 45.0


def parse_minsec(v) -> float:
    """DEAM 2013 metadata writes times as min.sec, e.g. '1.29' = 89 s."""
    s = str(v).strip()
    m, _, sec = s.partition(".")
    return 60.0 * int(m or 0) + float(int(sec.ljust(2, "0")) if sec else 0)


def load_segments(metadata_2013: Path | None) -> dict[int, float]:
    """song_id -> start (s) of the annotated 45 s excerpt within the full song."""
    if metadata_2013 is None:
        return {}
    md = pd.read_csv(metadata_2013, dtype=str)
    md.columns = [c.strip() for c in md.columns]
    col = next(c for c in md.columns if c.lower().startswith("start of the segment"))
    return {int(r["song_id"]): parse_minsec(r[col]) for _, r in md.iterrows()}


def aggregate_one(path: Path, start: float | None) -> tuple[pd.Series, int]:
    df = pd.read_csv(path, sep=";")
    tcol = next(c for c in df.columns if c.strip().lower() == "frametime")
    # 2013 feature files cover the full song while the audio and the ratings cover
    # a 45 s excerpt; keep only the frames of that excerpt.
    if start is not None and df[tcol].max() > CLIP_SECONDS + 5:
        t = df[tcol] - start
        df = df[(t >= 0) & (t < CLIP_SECONDS)]
    df = df.drop(columns=[tcol])
    df = df.apply(pd.to_numeric, errors="coerce")
    mean = df.mean(axis=0).add_suffix("__mean")
    std = df.std(axis=0, ddof=0).add_suffix("__std")
    return pd.concat([mean, std]), len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/raw/features/features"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"))
    ap.add_argument("--metadata-2013", type=Path, default=None,
                    help="metadata_2013.csv (segment start times); without it 2013 clips use the full song")
    ap.add_argument("--out", type=Path, default=Path("cache/opensmile"))
    args = ap.parse_args()

    seg = load_segments(args.metadata_2013)
    ids = pd.read_csv(args.labels)["song_id"].astype(int).tolist()
    rows, n_frames = {}, {}
    for sid in ids:
        rows[sid], n_frames[sid] = aggregate_one(args.features / f"{sid}.csv", seg.get(sid))
    X = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    X.index.name = "song_id"

    n_nan = int(X.isna().sum().sum())
    const = X.columns[X.std(axis=0) == 0].tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    X.to_parquet(args.out / "clip_features.parquet")

    nf = np.array(list(n_frames.values()))
    meta = {
        "source": str(args.features),
        "clips": int(len(X)),
        "dims": int(X.shape[1]),
        "aggregation": "mean and population std over the 500 ms frames of the annotated 45 s excerpt",
        "excerpt_cropping": "2013 clips cropped with metadata_2013 segment start" if seg else "none (full song for 2013)",
        "clips_cropped": int(sum(1 for s in ids if s in seg)),
        "frames_per_clip": {"min": int(nf.min()), "median": float(np.median(nf)), "max": int(nf.max())},
        "nan_values": n_nan,
        "constant_columns": const,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
