# SYDE 577 final results

## Protocol

We trained fully connected classifiers on 71 fixed-length audio features from 999 usable GTZAN recordings. The duplicate-aware 699/150/150 split used seed 577. Standardization used training statistics. Phase 3 trained eight configurations with five seeds each and saved the first checkpoint to reach maximum validation accuracy. Phase 4 reproduced all 40 saved validation accuracies and cross-entropy losses. It evaluated all five seeds for each of five configurations specified in the [evaluation plan](evaluation-plan.md) before held-out scoring. All metrics below use frozen checkpoints and evaluation mode.

## Architecture comparison

Accuracy is the five-seed mean ± sample standard deviation. Standard deviations are given in percentage points. The gap is mean training accuracy minus validation accuracy, both measured on the selected checkpoint.

| Architecture | Parameters | Validation accuracy | Test accuracy | Gap (pp) | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| Shallow, 71 → 64 → 10 | 5,258 | 68.27% ± 1.53 | 68.27% ± 4.28 | 24.64 | 0.677 |
| Intermediate, 71 → 128 → 64 → 10 | 18,122 | 70.13% ± 1.10 | 68.13% ± 1.73 | 28.87 | 0.683 |
| Deep, 71 → 128 → 64 → 32 → 10 | 19,882 | 69.73% ± 3.42 | 66.13% ± 2.33 | 28.03 | 0.661 |

The intermediate network had the highest mean validation accuracy and the lowest test variability among these three architectures. Its test mean was within 0.14 percentage points of the shallow baseline. The deep baseline had the lowest test mean of the three. The comparison changes depth, layer widths, and parameter count.

## Parameter-matched comparison

| Architecture | Parameters | Validation accuracy | Test accuracy | Gap (pp) | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| Shallow, 71 → 243 → 10 | 19,936 | 68.80% ± 2.23 | 71.07% ± 2.56 | 29.14 | 0.709 |
| Deep, 71 → 128 → 64 → 32 → 10 | 19,882 | 69.73% ± 3.42 | 66.13% ± 2.33 | 28.03 | 0.661 |

The wider shallow model had the highest observed test mean of the five pre-specified configurations. It exceeded the deep baseline by 4.93 percentage points with 54 more parameters. This comparison approximately controls parameter count and still changes how capacity is distributed across layers. The held-out ranking is a descriptive outcome of the locked evaluation.

## Learning rate, dropout, and generalization

- **Learning rate.** For the fixed deep architecture with dropout zero, mean best validation accuracy was 62.13% at 0.0001, 69.73% at 0.001, and 68.27% at 0.01. At 0.0001, evaluation-mode training accuracy averaged 75.45% and training loss was higher than at 0.001. These observations are consistent with slower optimization and incomplete fitting under this early-stopping policy. At 0.01, validation accuracy varied less across seeds (SD 1.01 pp) but selected-checkpoint validation cross-entropy averaged 1.962 versus 1.288 at 0.001.
- **Dropout.** At learning rate 0.001, deep with dropout 0.2 obtained 70.00% ± 2.26 pp on validation and 64.53% ± 1.52 pp on test, with test macro F1 of 0.643. On validation, dropout 0.5 reached 65.87% ± 3.21 pp and had lower evaluation-mode training accuracy (83.55%) than dropout zero (97.77%), consistent with stronger regularization and incomplete fitting under the tested settings. The learning-rate variants and dropout 0.5 were evaluated on validation only.
- **Corrected gaps.** Phase 3 training accuracy came from optimization batches with weights changing and dropout active when configured. Validation used the frozen epoch-end weights with dropout disabled. Phase 4 evaluates both partitions with the same frozen weights and dropout disabled. For dropout 0.2, this changed the mean gap from 18.70 to 25.51 pp. For dropout 0.5, it changed from 1.89 to 17.68 pp. The corrected gaps describe differences between partitions under the same inference conditions.
- **Overfitting.** The deep baseline and intermediate model averaged roughly 98–99% training accuracy at their selected checkpoints and roughly 70% validation accuracy. Across the five seeds, mean deep validation cross-entropy rose from 1.288 at selection to 1.547 at stopping while mean training cross-entropy fell from 0.117 in evaluation mode at selection to 0.032 in the final optimization epoch. With zero dropout, this is consistent with continued training fit alongside worsening held-out loss.
- **Per-genre evaluation.** For the intermediate model, mean test recall was 92.0% for classical and 93.3% for pop, compared with 33.3% for disco. Confusion matrices sum raw counts across seeds. Each matrix contains 750 predictions from the same 150 recordings, each scored five times.

## Limitations

GTZAN is small and contains documented repetitions and questionable labels. The split groups known duplicate recordings and may still share artists or related material across partitions. Validation and test each contain only 15 recordings per genre. Five training seeds characterize run-to-run variation on a single split. The limited hyperparameter search evaluates a few values independently, and the near-matched comparison still varies layer layout. The results support descriptive comparisons and do not establish statistical significance or generalization to other datasets.

## Supporting files

- [Evaluation-mode per-run metrics](eval-mode-per-run.csv) and [aggregate metrics](eval-mode-comparison.csv)
- [Held-out per-seed metrics](test-per-run.csv), [test summary](test-comparison.csv), and [per-genre results](test-per-genre.csv)
- [Validation and test with error bars](validation-vs-test.png), [validation-only tuning](validation-tuning.png), [test confusion matrices](test-confusion.png), and [Phase 3 learning curves](../phase3/learning-curves.png)
