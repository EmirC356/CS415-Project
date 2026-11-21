"""DEAM data audit (progress-report section 1).

Reads the unzipped DEAM archives under --raw, checks the items in
data/README.md, and writes:

  data/audit/audit_report.md   human-readable report (commit this)
  data/interim/labels.csv      one row per kept clip (local, git-ignored)

labels.csv columns:
  song_id, batch, n_raters, mu_V, s_V, mu_A, s_A   (rescaled to [-1, 1]: (x-5)/4, s/4;
                                         s = per-rater sd, ddof=1; mu from the averaged
                                         file, clip 1968 arousal from per-rater mean)
  mu_V_raw, s_V_raw, mu_A_raw, s_A_raw   (1-9 scale, as distributed)
  artist_raw, artist_norm, title, genre, has_audio, has_features

The DEAM folder layout is located by searching for files, not by fixed
paths, because the layout inside the zips has not been verified yet.
Every file the script uses is listed in the report.

Usage:
  python data/audit/audit_deam.py --raw data/raw
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_CLIPS = 1744          # 2013-2014 set (Aljanaki et al. 2017)
FIRST_2015_ID = 2001           # 2015 full songs use ids 2001-2058
MAX_SONGS_PER_ARTIST = 5       # stated in Aljanaki et al. 2017


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def norm_col(c: str) -> str:
    """'  Valence_Mean ' -> 'valence_mean'."""
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def read_csv_any(path: Path) -> pd.DataFrame:
    """DEAM csv files mix separators and encodings; try the common ones."""
    for enc in ("utf-8", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
            except Exception:
                continue
            # ragged rows make pandas shift fields into the index: reject
            if df.shape[1] > 1 and isinstance(df.index, pd.RangeIndex):
                df.columns = [norm_col(c) for c in df.columns]
                return df
    # Ragged rows (metadata_2014.csv has a variable number of last.fm tag
    # fields): keep the header's columns, truncate longer rows.
    import csv
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as fh:
                rows = list(csv.reader(fh))
        except UnicodeDecodeError:
            continue
        header = [h for h in rows[0] if h.strip()]
        k = len(header)
        body = [(r + [""] * k)[:k] for r in rows[1:] if any(x.strip() for x in r)]
        df = pd.DataFrame(body, columns=[norm_col(c) for c in header])
        return df.apply(lambda col: col.str.strip())
    raise ValueError(f"could not parse {path}")


_FEAT = re.compile(r"\s*(\(|\[)?\s*(feat\.?|ft\.?|featuring)\s+.*$", re.I)


def normalize_artist(name) -> str:
    """Grouping key for artist-disjoint folds.

    casefold, unicode NFKC, drop "feat./ft./featuring ..." tails,
    '&'/'and' unified, punctuation removed, whitespace collapsed.
    """
    if name is None or pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKC", str(name)).casefold().strip()
    s = _FEAT.sub("", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick(df: pd.DataFrame, *cands: str) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    for c in df.columns:                      # fuzzy fallback
        if any(k in c for k in cands):
            return c
    return None


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def find_static_files(raw: Path) -> list[Path]:
    files = [p for p in raw.rglob("*.csv")
             if "static" in p.name.lower() and "averaged" in str(p).lower()]
    if not files:  # fallback: any static csv
        files = [p for p in raw.rglob("*.csv") if "static" in p.name.lower()
                 and "each rater" not in str(p).lower()]
    return sorted(files)


def load_static(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = read_csv_any(f)
        cid = pick(df, "song_id", "songid", "id")
        cols = {
            "mu_V_raw": pick(df, "valence_mean"),
            "s_V_raw": pick(df, "valence_std"),
            "mu_A_raw": pick(df, "arousal_mean"),
            "s_A_raw": pick(df, "arousal_std"),
        }
        if cid is None or any(v is None for v in cols.values()):
            raise ValueError(f"{f}: unexpected columns {list(df.columns)}")
        out = df[[cid, *cols.values()]].copy()
        out.columns = ["song_id", *cols.keys()]
        out["source_file"] = f.name
        frames.append(out)
    lab = pd.concat(frames, ignore_index=True)
    lab["song_id"] = pd.to_numeric(lab["song_id"], errors="coerce").astype("Int64")
    for c in ["mu_V_raw", "s_V_raw", "mu_A_raw", "s_A_raw"]:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    return lab


def load_metadata(raw: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(p for p in raw.rglob("*.csv") if "metadata" in p.name.lower())
    frames = []
    for f in files:
        df = read_csv_any(f)
        cid = pick(df, "song_id", "songid", "id")
        if cid is None:
            continue
        year = re.search(r"(2013|2014|2015)", f.name)
        out = pd.DataFrame({
            "song_id": pd.to_numeric(df[cid], errors="coerce").astype("Int64"),
            "artist_raw": df[pick(df, "artist")] if pick(df, "artist") else pd.NA,
            "title": df[pick(df, "song_title", "track", "title")] if pick(df, "song_title", "track", "title") else pd.NA,
            "genre": df[pick(df, "genre")] if pick(df, "genre") else pd.NA,
            "meta_year": int(year.group(1)) if year else pd.NA,
        })
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["song_id", "artist_raw", "title", "genre", "meta_year"]), files
    meta = pd.concat(frames, ignore_index=True).drop_duplicates("song_id", keep="first")
    for c in ("artist_raw", "title", "genre"):
        meta[c] = meta[c].astype("string").str.strip()
    return meta, files


def load_raters(files: list[Path]) -> pd.DataFrame:
    """Per-rater static ratings: columns song_id, worker, V, A (1-9 scale).

    Only files with plain 'valence'/'arousal' columns are used (the 2013-2014
    file); the 2015 file reports per-rater averages/max/min of a dynamic task.
    """
    frames = []
    for f in files:
        df = read_csv_any(f)
        if not {"valence", "arousal"} <= set(df.columns):
            continue
        sid, wid = pick(df, "songid", "song_id"), pick(df, "workerid", "worker_id")
        frames.append(pd.DataFrame({
            "song_id": pd.to_numeric(df[sid], errors="coerce"),
            "worker": df[wid] if wid else np.arange(len(df)),
            "V": pd.to_numeric(df["valence"], errors="coerce"),
            "A": pd.to_numeric(df["arousal"], errors="coerce"),
        }))
    if not frames:
        return pd.DataFrame(columns=["song_id", "worker", "V", "A"])
    rr = pd.concat(frames, ignore_index=True).dropna(subset=["song_id"])
    rr["song_id"] = rr["song_id"].astype(int)
    return rr


SHRINK = 5.0   # pseudo-ratings pulling rater effects towards 0


def adjust_raters(rr: pd.DataFrame, scale: bool, iters: int = 20,
                  fit_clips: set | None = None) -> pd.DataFrame:
    """Remove each rater's additive tendency (and optionally scale width).

    Fits rating = clip_effect + rater_effect by alternating means, with the
    rater effect shrunk: sum(residuals) / (n_r + SHRINK). With scale=True the
    residual (rating - clip - rater) is divided by the rater's residual sd,
    shrunk towards the pooled sd with the same pseudo-count, then rescaled
    by the pooled sd so units stay comparable.
    """
    out = rr.copy()
    fit = rr if fit_clips is None else rr[rr["song_id"].isin(fit_clips)]
    for d in "VA":
        y = fit[d].to_numpy(float)
        clip = fit["song_id"].to_numpy()
        rater = fit["worker"].astype(str).to_numpy()
        b = pd.Series(0.0, index=pd.unique(rater))
        for _ in range(iters):
            c = pd.Series(y - b.reindex(rater).to_numpy()).groupby(clip).transform("mean").to_numpy()
            res = pd.Series(y - c).groupby(rater)
            b = res.sum() / (res.size() + SHRINK)
        e = y - b.reindex(rater).to_numpy()
        e = e - pd.Series(e).groupby(clip).transform("mean").to_numpy()
        pooled = e.std()
        g = pd.Series(e ** 2).groupby(rater)
        sd_r = np.sqrt((g.sum() + SHRINK * pooled ** 2) / (g.size() + SHRINK))
        # apply to every rating; raters unseen in the fit set get no correction
        r_all = rr["worker"].astype(str)
        adj = rr[d].to_numpy(float) - r_all.map(b).fillna(0.0).to_numpy()
        if scale:
            c = pd.Series(adj).groupby(rr["song_id"].to_numpy()).transform("mean").to_numpy()
            adj = c + (adj - c) / r_all.map(sd_r).fillna(pooled).to_numpy() * pooled
        out[d] = adj
    return out


def split_half(rr: pd.DataFrame, n_splits: int, seed: int) -> dict[str, float]:
    """Spearman-Brown corrected split-half reliability of per-clip mean, sd
    and mean absolute deviation, over clips with exactly 10 raters."""
    rng = np.random.default_rng(seed)
    g = rr.groupby("song_id")
    ids = [s for s, n in g.size().items() if n == 10]
    sub = rr[rr["song_id"].isin(ids)].sort_values("song_id")
    X = sub[["V", "A"]].to_numpy(float).reshape(len(ids), 10, 2)
    acc = {f"{q}_{d}": [] for q in ("mean", "sd", "mad") for d in "VA"}
    for _ in range(n_splits):
        perm = np.argsort(rng.random((len(ids), 10)), axis=1)
        Xp = np.take_along_axis(X, perm[:, :, None], axis=1)
        h = [Xp[:, :5], Xp[:, 5:]]
        stats = []
        for x in h:
            m = x.mean(1)
            stats.append({"mean": m, "sd": x.std(1, ddof=1),
                          "mad": np.abs(x - m[:, None]).mean(1)})
        for q in ("mean", "sd", "mad"):
            for j, d in enumerate("VA"):
                acc[f"{q}_{d}"].append(np.corrcoef(stats[0][q][:, j], stats[1][q][:, j])[0, 1])
    return {k: (lambda r: 2 * r / (1 + r))(float(np.mean(v))) for k, v in acc.items()}


def rater_analysis(rr: pd.DataFrame, kept: pd.DataFrame, problems: list[str],
                   n_splits: int = 200, seed: int = 415) -> list[str]:
    """n per clip, consistency with the averaged file, split-half reliability,
    and the analytic noise ceiling for predicting the mean rating."""
    out = []
    n = rr.groupby("song_id").size()
    out.append(f"- raters per clip: min {n.min()}, median {n.median():.0f}, max {n.max()}; "
               f"clips with n < 10: {(n < 10).sum()}; distinct raters: {rr['worker'].nunique()}")
    out.append(f"- kept clips without per-rater rows: {len(set(kept['song_id'].astype(int)) - set(n.index))}")
    out.append(f"- rating values seen: V {sorted(rr['V'].dropna().unique().tolist())}, "
               f"A {sorted(rr['A'].dropna().unique().tolist())}")

    k = kept.set_index(kept["song_id"].astype(int))
    out.append("")
    out.append("| dim | max abs(mean_raters - mu_file) | max abs(sd_ddof0 - s_file) | max abs(sd_ddof1 - s_file) |")
    out.append("|---|---|---|---|")
    for d in "VA":
        g = rr.groupby("song_id")[d]
        m, s0, s1 = g.mean(), g.std(ddof=0), g.std(ddof=1)
        idx = m.index.intersection(k.index)
        dm = (m[idx] - k.loc[idx, f"mu_{d}_raw"]).abs().max()
        d0 = (s0[idx] - k.loc[idx, f"s_{d}_raw"]).abs().max()
        d1 = (s1[idx] - k.loc[idx, f"s_{d}_raw"]).abs().max()
        out.append(f"| {d} | {dm:.3f} | {d0:.3f} | {d1:.3f} |")
        diff = (m[idx] - k.loc[idx, f"mu_{d}_raw"]).abs()
        bad = diff[diff > 0.06]          # file values are rounded to 0.01
        if len(bad):
            problems.append(f"{d}: {len(bad)} clip(s) whose per-rater mean differs from the averaged "
                            f"file by > 0.06: { {int(i): round(float(v), 2) for i, v in bad.items()} } (replaced by the per-rater mean, decision Q3c)")
    out.append("")
    out.append("(The averaged file is rounded; differences around 0.05 are rounding. The "
               "smaller sd column shows which std definition the file uses.)")

    # split-half reliability, raw and after removing rater tendencies (Q1b)
    variants = {
        "raw": rr,
        "rater bias removed": adjust_raters(rr, scale=False),
        "rater bias + scale removed": adjust_raters(rr, scale=True),
    }
    n10 = rr.groupby("song_id").size()
    out.append("")
    out.append(f"Split-half reliability ({n_splits} random 5/5 splits of each clip's raters; "
               f"clips with exactly 10 raters: {(n10 == 10).sum()} of {len(n10)}; "
               "Spearman-Brown corrected to 10 raters). Rater tendencies are estimated with "
               "an additive clip + rater model, rater effects shrunk towards 0 "
               f"(pseudo-count {SHRINK}); 'scale' also divides by each rater's shrunk residual sd.")
    out.append("")
    out.append("| variant | mean_V | mean_A | sd_V | sd_A | mad_V | mad_A |")
    out.append("|---|---|---|---|---|---|---|")
    for name, df in variants.items():
        rel = split_half(df, n_splits, seed)
        out.append(f"| {name} | " + " | ".join(f"{rel[k]:.3f}" for k in
                   ["mean_V", "mean_A", "sd_V", "sd_A", "mad_V", "mad_A"]) + " |")
    # cross-fitted check: rater effects estimated on one half of the clips,
    # reliability measured on the other half (rules out self-fitting)
    rng = np.random.default_rng(seed)
    ids = rr["song_id"].unique()
    cf = []
    for rep in range(3):
        perm = rng.permutation(ids)
        halves = (set(perm[: len(ids) // 2]), set(perm[len(ids) // 2:]))
        for fit_set, eval_set in (halves, halves[::-1]):
            adj = adjust_raters(rr, scale=True, fit_clips=fit_set)
            cf.append(split_half(adj[adj["song_id"].isin(eval_set)], 100, seed + rep))
    cfm = pd.DataFrame(cf).mean()
    out.append(f"| bias + scale, cross-fitted (6 half-swaps) | " + " | ".join(
        f"{cfm[k]:.3f}" for k in ["mean_V", "mean_A", "sd_V", "sd_A", "mad_V", "mad_A"]) + " |")
    out.append("")
    out.append("Ceiling for any correlation with s = sqrt(reliability): raw s about 0.2, "
               "rater-adjusted s about 0.4 (decision Q6c: report both, raw s primary).")

    # analytic check of the sd reliability: how much of the between-clip
    # variance of s is just sampling noise of an n-rater sd (~ s^2 / (2(n-1)))?
    out.append("")
    for d in "VA":
        g = rr.groupby("song_id")[d]
        s, nn = g.std(ddof=1), g.size()
        total = s.var(ddof=1)
        noise = (s ** 2 / (2 * (nn - 1))).mean()
        out.append(f"- s_{d}: between-clip variance {total:.3f}, expected sampling variance "
                   f"{noise:.3f}, implied reliability {max(0.0, 1 - noise / total):.3f} "
                   f"(normal-theory approximation)")
    out.append("- Note: Spearman(mu, s) in the label section is computed from the same raters; "
               "on a bounded scale a sample mean near the ends forces a small sample sd, so that "
               "correlation is partly mechanical. Split halves remove this coupling.")

    # analytic ceiling for predicting the mean: R2_max ~ 1 - E[s^2/n] / Var(mu)
    out.append("")
    for d in "VA":
        g = rr.groupby("song_id")[d]
        mu, var, nn = g.mean(), g.var(ddof=1), g.size()
        ceil = 1 - (var / nn).mean() / mu.var(ddof=1)
        out.append(f"- noise ceiling for R²({d}) of the mean rating: {ceil:.3f} "
                   f"(1 - mean(s²/n) / Var(mu); rater noise only)")
    return out


def ids_in(dirpath_glob: list[Path], exts: tuple[str, ...]) -> set[int]:
    ids = set()
    for p in dirpath_glob:
        if p.suffix.lower() in exts and p.stem.isdigit():
            ids.add(int(p.stem))
    return ids


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
def run(raw: Path, out_md: Path, out_labels: Path) -> int:
    lines: list[str] = ["# DEAM data audit", ""]
    problems: list[str] = []

    def section(t):
        lines.extend(["", f"## {t}", ""])

    # ---- files ---------------------------------------------------------
    static_files = find_static_files(raw)
    meta, meta_files = load_metadata(raw)
    all_files = list(raw.rglob("*"))
    audio_ids = ids_in(all_files, (".mp3", ".wav"))
    feat_ids = ids_in([p for p in all_files if "feature" in str(p).lower()], (".csv",))
    rater_dir_files = [p for p in all_files if "each rater" in str(p).lower() and p.is_file()]
    rater_static = [p for p in rater_dir_files
                    if "static" in str(p).lower() or "song_level" in str(p).lower()]

    section("Files used")
    lines += [f"- static labels: `{f.relative_to(raw).as_posix()}`" for f in static_files] or ["- static labels: **none found**"]
    lines += [f"- metadata: `{f.relative_to(raw).as_posix()}`" for f in meta_files] or ["- metadata: **none found**"]
    lines.append(f"- audio files with numeric ids: {len(audio_ids)}")
    lines.append(f"- openSMILE feature csv files with numeric ids: {len(feat_ids)}")
    if not static_files:
        problems.append("no static label file found")
        out_md.write_text("\n".join(lines + ["", "**Audit stopped: no static labels.**"]), encoding="utf-8")
        print("\n".join(lines))
        return 1

    # ---- labels --------------------------------------------------------
    lab = load_static(static_files)
    dup = lab["song_id"].duplicated(keep=False)
    lab = lab.drop_duplicates("song_id", keep="first")
    lab = lab.merge(meta, on="song_id", how="left")
    lab["batch"] = np.where(lab["meta_year"].notna(), lab["meta_year"],
                            np.where(lab["song_id"] >= FIRST_2015_ID, 2015, pd.NA))
    keep = lab["song_id"] < FIRST_2015_ID
    kept = lab[keep].copy()
    n_kept = len(kept)

    section("Clip count")
    lines.append(f"- rows in static files: {len(lab) + int(dup.sum() // 2)}; duplicate ids: {int(dup.sum())}")
    lines.append(f"- kept (id < {FIRST_2015_ID}, i.e. 2013-2014): **{n_kept}** (expected {EXPECTED_CLIPS})")
    lines.append(f"- excluded 2015 rows: {int((~keep).sum())}")
    if n_kept != EXPECTED_CLIPS:
        problems.append(f"clip count {n_kept} != {EXPECTED_CLIPS}")
    if kept["batch"].notna().any():
        lines.append(f"- batch from metadata: {kept['batch'].value_counts(dropna=False).to_dict()}")
    else:
        lines.append("- batch: metadata has no year information; 2013/2014 flag missing")
        problems.append("2013/2014 batch flag unavailable")

    section("Label values (1-9 scale)")
    lines.append("| column | missing | min | max | mean |")
    lines.append("|---|---|---|---|---|")
    for c in ["mu_V_raw", "s_V_raw", "mu_A_raw", "s_A_raw"]:
        v = kept[c]
        lines.append(f"| {c} | {int(v.isna().sum())} | {v.min():.3f} | {v.max():.3f} | {v.mean():.3f} |")
        if v.isna().any():
            problems.append(f"{c}: {int(v.isna().sum())} missing")
    for d in "VA":
        mu, s = kept[f"mu_{d}_raw"], kept[f"s_{d}_raw"]
        bad_mu = ((mu < 1) | (mu > 9)).sum()
        bad_s = (s < 0).sum()
        # Bhatia-Davis: Var <= (max-mu)(mu-min) for a variable bounded in [1, 9]
        bd = np.sqrt(np.clip((9 - mu) * (mu - 1), 0, None))
        over_bd = (s > bd + 1e-6).sum()
        lines.append(f"- {d}: mu outside [1,9]: {int(bad_mu)}; s<0: {int(bad_s)}; "
                     f"s above Bhatia-Davis max: {int(over_bd)}")
        if bad_mu or bad_s or over_bd:
            problems.append(f"{d}: {int(bad_mu)} mu out of range, {int(bad_s)} negative s, {int(over_bd)} s over bound")
        lines.append(f"- {d}: Spearman(mu, s) = {mu.corr(s, method='spearman'):.3f}; "
                     f"Spearman(|mu-5|, s) = {(mu - 5).abs().corr(s, method='spearman'):.3f}")

    section("Raters per clip (n)")
    lines.append(f"- files under an 'each rater' folder: {len(rater_dir_files)}")
    kept["n_raters"] = pd.NA
    if rater_static:
        lines.append(f"- **per-rater static files found: {len(rater_static)}** "
                     + ", ".join(f"`{p.relative_to(raw).as_posix()}`" for p in rater_static))
        rr = load_raters(rater_static)
        rr = rr[rr["song_id"].isin(set(kept["song_id"].astype(int)))]
        if rr.empty:
            lines.append("- per-rater file has no usable rows for the kept clips")
        else:
            lines += rater_analysis(rr, kept, problems, n_splits=500)
            n = rr.groupby("song_id").size()
            kept["n_raters"] = kept["song_id"].astype(int).map(n)
            # Label source (decision Q3c, 02 Tur 10): mu from the averaged file,
            # except clips whose per-rater mean disagrees by > 0.06 (clip 1968);
            # s always from the per-rater ratings with ddof=1, consistent with n.
            sid = kept["song_id"].astype(int)
            for d in "VA":
                g = rr.groupby("song_id")[d]
                m_r, s_r = sid.map(g.mean()), sid.map(g.std(ddof=1))
                kept[f"s_{d}_file"] = kept[f"s_{d}_raw"]
                fix = (m_r - kept[f"mu_{d}_raw"]).abs() > 0.06
                if fix.any():
                    lines.append(f"- {d}: mu replaced by the per-rater mean for clip(s) "
                                 f"{sid[fix].tolist()} ({kept.loc[fix, f'mu_{d}_raw'].round(2).tolist()} "
                                 f"-> {m_r[fix].round(2).tolist()})")
                    kept.loc[fix, f"mu_{d}_raw"] = m_r[fix]
                kept[f"s_{d}_raw"] = s_r.where(s_r.notna(), kept[f"s_{d}_raw"])
            lines.append("- s_V, s_A in labels.csv are per-rater sds (ddof=1); the averaged-file "
                         "values are kept as s_V_file, s_A_file")
            adj = adjust_raters(rr, scale=True)
            for d in "VA":
                kept[f"s_{d}_adj_raw"] = sid.map(adj.groupby("song_id")[d].std(ddof=1))
            lines.append("- s_V_adj, s_A_adj: per-clip sd after removing rater tendency and scale "
                         "(fitted on all clips; models that use it as a target must refit the "
                         "rater effects on training folds only)")
    else:
        lines.append("- **per-rater static ratings: not found.** n per clip cannot be read "
                     "from the averaged files. The framework uses n = 10 (minimum stated "
                     "in Aljanaki et al. 2017) as a stated assumption.")

    section("ID alignment")
    ids = set(kept["song_id"].astype(int))
    lines.append(f"- kept clips without audio: {len(ids - audio_ids)}"
                 + (f" (first: {sorted(ids - audio_ids)[:10]})" if ids - audio_ids else ""))
    lines.append(f"- kept clips without openSMILE features: {len(ids - feat_ids)}"
                 + (f" (first: {sorted(ids - feat_ids)[:10]})" if ids - feat_ids else ""))
    if audio_ids and ids - audio_ids:
        problems.append(f"{len(ids - audio_ids)} clips without audio")
    if feat_ids and ids - feat_ids:
        problems.append(f"{len(ids - feat_ids)} clips without features")
    if not audio_ids:
        lines.append("- (audio not unzipped yet: check skipped)")
    if not feat_ids:
        lines.append("- (features not unzipped yet: check skipped)")
    kept["has_audio"] = kept["song_id"].astype(int).isin(audio_ids)
    kept["has_features"] = kept["song_id"].astype(int).isin(feat_ids)

    section("Artists")
    kept["artist_norm"] = kept["artist_raw"].map(normalize_artist)
    missing_artist = (kept["artist_norm"] == "").sum()
    per_artist = kept.loc[kept["artist_norm"] != "", "artist_norm"].value_counts()
    merged = (kept.groupby("artist_norm")["artist_raw"].nunique() > 1).sum()
    lines.append(f"- clips without artist: **{int(missing_artist)}**")
    lines.append(f"- distinct normalised artists: {len(per_artist)}; raw spellings merged "
                 f"by normalisation: {int(merged)} artists")
    if len(per_artist):
        lines.append(f"- songs per artist: max {per_artist.max()}, median {per_artist.median():.0f}; "
                     f"artists with 1 song: {(per_artist == 1).sum()}")
        over = per_artist[per_artist > MAX_SONGS_PER_ARTIST]
        lines.append(f"- artists above {MAX_SONGS_PER_ARTIST} songs: {len(over)}"
                     + (f" ({over.head(10).to_dict()})" if len(over) else ""))
        lines.append("- top 10: " + ", ".join(f"{a} ({n})" for a, n in per_artist.head(10).items()))
    if missing_artist:
        problems.append(f"{int(missing_artist)} clips without artist: they become their own group")
        kept.loc[kept["artist_norm"] == "", "artist_norm"] = (
            "__song_" + kept.loc[kept["artist_norm"] == "", "song_id"].astype(str))

    # ---- write ---------------------------------------------------------
    for d in "VA":
        kept[f"mu_{d}"] = (kept[f"mu_{d}_raw"] - 5) / 4
        kept[f"s_{d}"] = kept[f"s_{d}_raw"] / 4
        if f"s_{d}_adj_raw" in kept.columns:
            kept[f"s_{d}_adj"] = kept[f"s_{d}_adj_raw"] / 4
    cols = ["song_id", "batch", "n_raters", "mu_V", "s_V", "mu_A", "s_A",
            "mu_V_raw", "s_V_raw", "mu_A_raw", "s_A_raw",
            *[c for c in ("s_V_file", "s_A_file", "s_V_adj", "s_A_adj") if c in kept.columns],
            "artist_raw", "artist_norm", "title", "genre", "has_audio", "has_features"]
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    kept[cols].sort_values("song_id").to_csv(out_labels, index=False)

    section("Summary")
    lines += [f"- PROBLEM: {p}" for p in problems] or ["- no problems found"]
    lines.append(f"- labels written to `{out_labels.as_posix()}` ({n_kept} rows)")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--report", type=Path, default=Path("data/audit/audit_report.md"))
    ap.add_argument("--labels", type=Path, default=Path("data/interim/labels.csv"))
    a = ap.parse_args()
    raise SystemExit(run(a.raw, a.report, a.labels))
