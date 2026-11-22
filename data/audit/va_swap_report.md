# Valence/arousal swap check (static vs dynamic labels)

| batch | n | r(static V, dyn V) | r(static V, dyn A) | r(static A, dyn A) | r(static A, dyn V) | ρ(energy, static V) | ρ(energy, static A) | verdict |
|---|---|---|---|---|---|---|---|---|
| 2013 | 744 | 0.575 | 0.956 | 0.570 | 0.949 | 0.544 | 0.315 | **V/A swapped** |
| 2014 | 1000 | 0.912 | 0.591 | 0.942 | 0.569 | 0.316 | 0.423 | consistent |

Batches with swapped static labels: [2013]
Fixed labels written to `data/interim/labels_fixed.csv` (744 clips swapped back).
