# Data: DEAM (owner: Can)

Source: https://cvml.unige.ch/databases/DEAM/ (licence CC BY-NC). **Nothing under `data/raw/` is committed.**

## Download

| Archive | Size | Needed for |
|---|---|---|
| `DEAM_Annotations.zip` | 4.7 MB | Labels (everyone) |
| `metadata.zip` | small | Artist, title and genre for the splits (Emir) |
| `features.zip` | 601.5 MB | openSMILE baseline (Efe); no audio needed |
| `DEAM_audio.zip` | 1.3 GB | MERT and PANNs embeddings (Can) |

Unzip each archive into `data/raw/<archive name without .zip>/`.

## Label conventions

- Use only the 2013–2014 clips (1,744). The 2015 set (58 full songs, 5 annotators, different protocol) is excluded.
- Static labels come as mean μ and std s on a 1–9 scale. Rescale them to [−1, 1] with `mu' = (mu − 5) / 4` and `s' = s / 4`.
- The DEAM manual mentions per-rater raw dynamic annotations, but not per-rater static ratings. The data audit found that per-rater static ratings do exist: 10–20 per clip, from 187 raters.
- **The 2013 static valence and arousal are swapped** in DEAM, in both the averaged file and the per-rater file. They are corrected in `labels_fixed.csv`, which is the label file of record. See [`docs/contracts.md`](../docs/contracts.md#label-file-of-record) and `data/audit/va_swap_report.md`.

## Data audit (first section of the progress report)

Run:

```bash
python data/audit/audit_deam.py --raw data/raw
python data/audit/check_va_swap.py --raw data/raw --fix --labels data/interim/labels.csv --out data/interim/labels_fixed.csv
```

The first command writes `data/audit/audit_report.md` (commit this) and `data/interim/labels.csv`. The second turns `labels.csv` into `data/interim/labels_fixed.csv`, which is what every model uses. Both files are local. The audit finds its input files by searching, so the exact folder layout inside the zips does not matter.

The audit also detects the 2013 swap itself. It computes the reliability and the rater-adjusted s in corrected orientation, while `labels.csv` keeps the supplied orientation. The audit reports:

- [ ] Clip count is 1,744; there are no missing μ or s values; μ and s lie in valid ranges.
- [ ] Number of raters n per clip, if it can be recovered.
- [ ] Song IDs match across the audio, `features.zip` and the annotation csv files.
- [ ] Artist names are normalised (case, whitespace, "feat."); songs per artist are counted.
- [ ] Each clip is flagged with its 2013 or 2014 batch.
- [ ] Per-rater static ratings: present or absent.
