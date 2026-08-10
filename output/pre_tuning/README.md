# Pre-tuning results

These files preserve the test-set outputs generated at commit `ae7bea1` before the validation-only recalibration and before correcting the NDI left-right reflection axis.

They are retained for comparison only. Running `main.ipynb` after the improvement commit generates the current metrics and figures in the normal `output/` locations.

| Model | Dice mean | IoU mean | Missed tumors |
|---|---:|---:|---:|
| Baseline GMM | 0.6817 | 0.6110 | 12/60 |
| Spatial GMM | 0.7148 | 0.6394 | 9/60 |
| Spatial GMM + NDI | 0.7423 | 0.6638 | 6/60 |
