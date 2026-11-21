# features/ (owners: Efe for openSMILE, Can for MERT)

This folder holds extraction code. Outputs go to `cache/` (git-ignored). The format is in [`docs/contracts.md#embedding-cache`](../docs/contracts.md#embedding-cache).

| Encoder | Input | Settings | When |
|---|---|---|---|
| openSMILE | `data/raw/features/` (DEAM's 500 ms frames) | Per-clip mean and std | Now |
| MERT-v1-95M | `data/raw/DEAM_audio/` | 24 kHz, 5 s windows, **all 13 layers in one pass** | Now |
| Random-init MERT | Same as MERT | Same architecture, untrained weights | After the report |
| PANNs CNN14 | Audio | 32 kHz, window mean | After the report |

MERT extraction needs a GPU (Colab is fine). Extracting all 13 layers costs no more than extracting one.

## MERT extraction

```bash
python features/extract_mert.py --audio data/raw/DEAM_audio --labels data/interim/labels.csv --out cache/mert95m
```

- The run is resumable: existing `.npy` files are skipped. Run `--limit 5` first as a smoke test.
- Output: `cache/mert95m/<song_id>.npy` with shape (windows, 13, 768), float16, plus `meta.json`.
- A trailing window shorter than 5 s is dropped, so a 45 s clip gives 9 windows.
- On a CUDA machine, install the CUDA build of torch first (https://pytorch.org/get-started/locally/).
- `--random-init --out cache/mert95m_random` builds the H2b null model, after the report.
