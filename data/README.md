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
- The DEAM manual confirms that per-rater raw dynamic annotations exist. It does **not** mention per-rater static ratings. The data audit checks this.

## Data audit (first section of the progress report)

The audit script and its output go in `data/audit/`. It reports:

- [ ] Clip count is 1,744; there are no missing μ or s values; μ and s lie in valid ranges.
- [ ] Number of raters n per clip, if it can be recovered.
- [ ] Song IDs match across the audio, `features.zip` and the annotation csv files.
- [ ] Artist names are normalised (case, whitespace, "feat."); songs per artist are counted.
- [ ] Each clip is flagged with its 2013 or 2014 batch.
- [ ] Per-rater static ratings: present or absent.
