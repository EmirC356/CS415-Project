# DEAM data audit


## Files used

- static labels: `DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv`
- static labels: `DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv`
- metadata: `metadata/metadata/metadata_2013.csv`
- metadata: `metadata/metadata/metadata_2014.csv`
- metadata: `metadata/metadata/metadata_2015.csv`
- audio files with numeric ids: 0
- openSMILE feature csv files with numeric ids: 0

## Clip count

- rows in static files: 1802; duplicate ids: 0
- kept (id < 2001, i.e. 2013-2014): **1744** (expected 1744)
- excluded 2015 rows: 58
- batch from metadata: {2014: 1000, 2013: 744}

## Label values (1-9 scale)

| column | missing | min | max | mean |
|---|---|---|---|---|
| mu_V_raw | 0 | 1.600 | 8.400 | 4.903 |
| s_V_raw | 0 | 0.300 | 2.900 | 1.522 |
| mu_A_raw | 0 | 1.600 | 8.100 | 4.813 |
| s_A_raw | 0 | 0.460 | 2.590 | 1.467 |
- V: mu outside [1,9]: 0; s<0: 0; s above Bhatia-Davis max: 0
- V: Spearman(mu, s) = -0.155; Spearman(|mu-5|, s) = -0.204
- A: mu outside [1,9]: 0; s<0: 0; s above Bhatia-Davis max: 0
- A: Spearman(mu, s) = 0.066; Spearman(|mu-5|, s) = -0.303

## Raters per clip (n)

- files under an 'each rater' folder: 3606
- **per-rater static files found: 2** `DEAM_Annotations/annotations/annotations per each rater/song_level/static_annotations_songs_1_2000.csv`, `DEAM_Annotations/annotations/annotations per each rater/song_level/static_annotations_songs_2000_2058.csv`
- raters per clip: min 10, median 10, max 20; clips with n < 10: 0; distinct raters: 187
- kept clips without per-rater rows: 0
- rating values seen: V [1, 2, 3, 4, 5, 6, 7, 8, 9], A [1, 2, 3, 4, 5, 6, 7, 8, 9]

| dim | max abs(mean_raters - mu_file) | max abs(sd_ddof0 - s_file) | max abs(sd_ddof1 - s_file) |
|---|---|---|---|
| V | 0.005 | 0.139 | 0.157 |
| A | 0.620 | 0.092 | 0.139 |

(The averaged file is rounded; differences around 0.05 are rounding. The smaller sd column shows which std definition the file uses.)

Split-half reliability (500 random 5/5 splits of each clip's raters; clips with exactly 10 raters: 1738 of 1744; Spearman-Brown corrected to 10 raters). Rater tendencies are estimated with an additive clip + rater model, rater effects shrunk towards 0 (pseudo-count 5.0); 'scale' also divides by each rater's shrunk residual sd.

| variant | mean_V | mean_A | sd_V | sd_A | mad_V | mad_A |
|---|---|---|---|---|---|---|
| raw | 0.802 | 0.847 | 0.043 | -0.011 | 0.069 | 0.021 |
| rater bias removed | 0.841 | 0.888 | 0.101 | 0.054 | 0.124 | 0.074 |
| rater bias + scale removed | 0.832 | 0.881 | 0.190 | 0.145 | 0.193 | 0.146 |
| bias + scale, cross-fitted (6 half-swaps) | 0.826 | 0.878 | 0.197 | 0.150 | 0.199 | 0.152 |

Ceiling for any correlation with s = sqrt(reliability): raw s about 0.2, rater-adjusted s about 0.4 (decision Q6c: report both, raw s primary).

- s_V: between-clip variance 0.165, expected sampling variance 0.152, implied reliability 0.077 (normal-theory approximation)
- s_A: between-clip variance 0.150, expected sampling variance 0.141, implied reliability 0.060 (normal-theory approximation)
- Note: Spearman(mu, s) in the label section is computed from the same raters; on a bounded scale a sample mean near the ends forces a small sample sd, so that correlation is partly mechanical. Split halves remove this coupling.

- noise ceiling for R²(V) of the mean rating: 0.802 (1 - mean(s²/n) / Var(mu); rater noise only)
- noise ceiling for R²(A) of the mean rating: 0.848 (1 - mean(s²/n) / Var(mu); rater noise only)
- A: mu replaced by the per-rater mean for clip(s) [1968] ([6.18] -> [6.8])
- s_V, s_A in labels.csv are per-rater sds (ddof=1); the averaged-file values are kept as s_V_file, s_A_file
- s_V_adj, s_A_adj: per-clip sd after removing rater tendency and scale (fitted on all clips; models that use it as a target must refit the rater effects on training folds only)

## ID alignment

- kept clips without audio: 1744 (first: [2, 3, 4, 5, 7, 8, 10, 12, 13, 17])
- kept clips without openSMILE features: 1744 (first: [2, 3, 4, 5, 7, 8, 10, 12, 13, 17])
- (audio not unzipped yet: check skipped)
- (features not unzipped yet: check skipped)

## Artists

- clips without artist: **0**
- distinct normalised artists: 898; raw spellings merged by normalisation: 1 artists
- songs per artist: max 23, median 1; artists with 1 song: 596
- artists above 5 songs: 35 ({'jason shaw': 23, 'jahzzar': 20, 'jared c balogh': 19, 'kevin macleod': 17, 'the agrarians': 17, 'dexter britain': 16, 'the new mystikal troubadours': 15, 'mr and mrs smith': 13, 'krackatoa': 12, 'broke for free': 12})
- top 10: jason shaw (23), jahzzar (20), jared c balogh (19), kevin macleod (17), the agrarians (17), dexter britain (16), the new mystikal troubadours (15), mr and mrs smith (13), krackatoa (12), broke for free (12)

## Summary

- PROBLEM: A: 1 clip(s) whose per-rater mean differs from the averaged file by > 0.06: {1968: 0.62} (averaged-file value kept)
- labels written to `data/interim/labels.csv` (1744 rows)
