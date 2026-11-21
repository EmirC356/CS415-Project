# eval/ (owner: Emir)

A single shared metrics script reads `results/predictions/*.csv` and `splits/*.csv` and nothing else. No model code computes its own reported metrics.

## Version 1 (for the progress report)

Computed per dimension (V and A) and per fold, then summarised as mean ± SD over folds:

- R², taken against the test fold's own mean (sklearn definition)
- RMSE, on both the [−1, 1] scale and the 1–9 scale
- CCC and Pearson r
- For σ models: PICP at 68% and 95%, plain Gaussian NLL, and Spearman ρ(σ, |e|)
- Paired per-fold differences between two models (e.g. `mert_ridge` vs `opensmile_ridge`)

Test this version against a fake prediction file before real results arrive.

## Version 2 (after the report)

- Artist-grouped bootstrap CIs, P(A>B), Holm, TOST
- Reliability diagram and ENCE on pooled out-of-fold predictions
- The ρ(σ, |e|) ceiling from simulating e ~ N(0, σ̂²), and the partial correlation on |μ̂|
- ρ(σ, s), raw and partial on μ and μ̂; σ against s/√n
- σ decomposition: regress σ̂² on s²/n, seed-ensemble variance and |μ̂|
- AURC
