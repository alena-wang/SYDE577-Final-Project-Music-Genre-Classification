# SYDE 577 end-to-end reproducibility summary

The revised project reproduced the original study's results for all six retained configurations. This run started from the Python source and raw GTZAN recordings, then generated new features, checkpoints, and evaluation results. The original study served as a reference for the final comparisons.

## 1. Start from raw audio

I extracted the source, tests, requirements, and documentation from the submission ZIP into a separate project. I skipped the packaged processed data, checkpoints, histories, metrics, and figures. The new project reads the original GTZAN WAVs through `data/raw/genres_original`, a symbolic link. All generated files were written inside the new project.

## 2. Extract features, split recordings, and fit normalization

I ran the existing preprocessor with split seed **577**. It reads each WAV and summarizes the 20 MFCCs, 12 chroma bins, spectral centroid, RMS energy, and zero-crossing rate using means and standard deviations. A global tempo estimate gives **71 features** per recording. The decoder excluded `jazz.00054.wav`, leaving **999 usable recordings**.

The split keeps known duplicate groups together and assigns **699 training, 150 validation, and 150 test recordings**. I checked the feature shape, feature and label order, duplicate groups, and partition counts, and recomputed the means and scales from the 699 training rows. Training and evaluation apply those saved statistics to their respective partitions. The regenerated dataset and split manifest both matched the original study's SHA-256 hashes.

| File | What it contains |
|---|---|
| [dataset.npz](data/processed/dataset.npz) | Feature matrix, labels, split, recording IDs, feature order, and training-only normalization. |
| [splits.csv](data/processed/splits.csv) | Recording-by-recording partition assignments, duplicate groups, and the excluded WAV. |

## 3. Train and validate the six configurations

I ran six configurations across seeds **577–581**, for **30 fresh training runs**. The two baselines use the same fixed data, Adam, cross-entropy, batch size 32, up to 100 epochs, and patience 12. The deep tuning variants change either learning rate or dropout. At each epoch the model was evaluated on validation data, and the first epoch to reach the highest validation accuracy supplied the best checkpoint. The test partition was reserved for the final baseline evaluation.

| File | How to read it |
|---|---|
| [Training plan](runs/training/plan.json) | The six configurations, seeds, training settings, and dataset hash. |
| [Training results by run](runs/training/per-run.csv) | One row per configuration and seed. Includes the selected epoch, validation accuracy and loss, parameter count, and run duration. |
| [Validation comparison](runs/training/comparison.csv) | One row per configuration. Reports mean best validation accuracy and sample standard deviation across five seeds. |
| [Learning curves](runs/training/learning-curves.png) | Training and validation accuracy and loss through each seed's stopping epoch. |

Under `runs/training/<configuration>/seed-<seed>/`, `history.csv` contains the epoch-by-epoch metrics, `best.pt` is the validation-selected checkpoint, and `run.json` records the settings and selected result.

## 4. Audit the baseline checkpoints and score the held-out test set

The [evaluation plan](evaluation-plan.md) specifies shallow and deep, all five seeds each. I audited the ten newly trained baseline checkpoints against the fixed dataset and their validation histories. I scored both training and validation with the same frozen weights and dropout disabled, then calculated the accuracy gap. This makes the two scores directly comparable. The original epoch history measures training accuracy while weights are changing.

After the audit, I scored the same ten frozen checkpoints on all 150 test recordings. The four tuning variants contributed validation results only. Test loss is mean cross-entropy. Macro precision, recall, and F1 give each genre equal weight.

| File | How to read it |
|---|---|
| [Evaluation-mode metrics by run](runs/evaluation/eval-mode-per-run.csv) and [aggregate](runs/evaluation/eval-mode-comparison.csv) | Corrected training and validation scores and their gap for the two baselines. |
| [Test metrics by run](runs/evaluation/test-per-run.csv) | Test accuracy, loss, and macro metrics for each baseline seed. |
| [Test comparison](runs/evaluation/test-comparison.csv) | Baseline means and sample standard deviations of test accuracy, plus mean loss and macro metrics. |
| [Test metrics by genre](runs/evaluation/test-per-genre.csv) | Per-genre precision, recall, and F1 averaged across seeds. |
| [Shallow matrix CSV](runs/evaluation/test-confusion-matrix-shallow.csv), [deep matrix CSV](runs/evaluation/test-confusion-matrix-deep.csv), and [figure](runs/evaluation/test-confusion-matrices.png) | Rows are actual genres, columns are predicted genres, and the diagonal counts correct predictions. Each matrix sums five scores on the same 150 recordings. |
| [Validation vs. test figure](runs/evaluation/validation-vs-test.png) and [validation-only tuning figure](runs/evaluation/validation-tuning.png) | Error bars show sample standard deviation across the five seeds on one fixed split. |

## 5. Compare with the original study

I matched CSV rows by configuration, seed, and genre as applicable. All **six aggregate validation rows** matched exactly. All **30 per-run training rows** matched after excluding elapsed run time. The **ten test rows**, baseline test aggregates, evaluation-mode metrics, per-genre rows, and both confusion matrices also matched exactly. All **14 automated tests** passed.

| Configuration | Best validation accuracy | Held-out test accuracy |
|---|---:|---:|
| Shallow, 71 → 64 → 10 | 68.27% ± 1.53 pp | 68.27% ± 4.28 pp |
| Deep, 71 → 128 → 64 → 32 → 10 | 69.73% ± 3.42 pp | 66.13% ± 2.33 pp |
| Deep, LR 0.0001 | 62.13% ± 3.57 pp | — |
| Deep, LR 0.01 | 68.27% ± 1.01 pp | — |
| Deep, dropout 0.2 | 70.00% ± 2.26 pp | — |
| Deep, dropout 0.5 | 65.87% ± 3.21 pp | — |

The deep baseline scored higher on validation, while the shallow baseline scored higher on the held-out test set. These are descriptive results from one fixed split. The original study also included other architectures, so the comparison here covers the six configurations retained in the revised project.

See the [README](README.md) for commands and the [project guide](PROJECT_GUIDE.md) for the code map.
