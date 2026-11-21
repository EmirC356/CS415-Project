# features/ (owners: Efe for openSMILE, Can for MERT)

This folder holds extraction code. Outputs go to `cache/` (git-ignored). The format is in [`docs/contracts.md#embedding-cache`](../docs/contracts.md#embedding-cache).

| Encoder | Input | Settings | When |
|---|---|---|---|
| openSMILE | `data/raw/features/` (DEAM's 500 ms frames) | Per-clip mean and std | Now |
| MERT-v1-95M | `data/raw/DEAM_audio/` | 24 kHz, 5 s windows, **all 13 layers in one pass** | Now |
| Random-init MERT | Same as MERT | Same architecture, untrained weights | After the report |
| PANNs CNN14 | Audio | 32 kHz, window mean | After the report |

MERT extraction needs a GPU (Colab is fine). Extracting all 13 layers costs no more than extracting one.
