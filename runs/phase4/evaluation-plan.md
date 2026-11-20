# Held-out evaluation plan

This plan was written before Phase 4 loaded labels or predicted on the held-out test partition. The five configurations were chosen using Phase 3 validation-only results and the research questions about depth, parameter count, and moderate dropout.

| Configuration ID | Architecture | Adam learning rate | Dropout |
|---|---|---:|---:|
| shallow | 71 → 64 → 10 | 0.001 | 0 |
| intermediate | 71 → 128 → 64 → 10 | 0.001 | 0 |
| deep | 71 → 128 → 64 → 32 → 10 | 0.001 | 0 |
| shallow-matched | 71 → 243 → 10 | 0.001 | 0 |
| deep-dropout-0.2 | 71 → 128 → 64 → 32 → 10 | 0.001 | 0.2 |

Evaluate all 25 existing Phase 3 best checkpoints for these configurations and training seeds 577, 578, 579, 580, and 581. The split seed is 577. Preserve checkpoint selection by the first epoch with maximum validation accuracy. Use the saved training-only mean and scale, fixed genre mapping, all 150 test recordings, and evaluation mode with no gradients. Freeze the configuration set, seeds, and checkpoints after observing results. Use no retraining or test-driven tuning.

Report per-seed accuracy, mean cross-entropy, and macro precision, recall, and F1 across the ten labels, using zero for undefined precision or recall. Report five-seed mean and sample standard deviation of accuracy. Sum raw confusion counts across the five seeds separately for each configuration. Rows are true labels and columns are predictions. Each matrix has 750 predictions from 150 recordings scored five times. Summarize per-genre metrics across seeds with each genre's underlying recording support. Seed variation describes variability on this one fixed split.
