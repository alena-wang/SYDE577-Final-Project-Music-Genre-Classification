# SYDE 577 submission guide

The main comparison is shallow `71 → 64 → 10` versus deep `71 → 128 → 64 → 32 → 10`. The models share the same input features, dataset partitions, activation, optimizer, and checkpoint selection rule. They have 5,258 and 19,882 parameters, respectively, so the comparison changes parameter count along with depth.

## Data and code map

| File | Responsibility |
|---|---|
| `genre/dataset.py` | Finds GTZAN files, defines the ten-label order, joins documented and byte-identical duplicates, and assigns whole duplicate groups to splits. |
| `genre/features.py` | Extracts the fixed-length feature vector for one recording. |
| `genre/preprocess.py`, `genre/scaling.py` | Cache features, record exclusions, save the dataset and split manifest, and compute feature means and scales from training rows. |
| `genre/model.py` | One configurable MLP with ReLU hidden layers and ten output logits. |
| `genre/train.py` | Builds shuffled training batches, trains with Adam and cross-entropy, checks validation each epoch, and saves the earliest checkpoint that attains maximum validation accuracy. |
| `genre/experiments.py` | Runs the baseline grid or explicit learning-rate and dropout configurations, saves each seed separately, and summarizes validation results. |
| `genre/evaluate.py` | Checks the fixed archive and the two baselines before scoring their frozen checkpoints. |
| `genre/evaluation_plots.py` | Plots baseline validation versus test accuracy, confusion matrices, and optional validation-only tuning results. |

`data/processed/dataset.npz` contains the features, labels, fixed split, and saved training-only scaler. `splits.csv` records assignment, duplicate groups, and the unreadable clip excluded from the 1,000 WAV files. The final counts are 699/150/150. Each clip contributes the means and standard deviations of 20 MFCCs, 12 chroma bins, spectral centroid, RMS energy, and zero-crossing rate, plus one global tempo estimate.

The tests cover feature extraction, splitting and scaling, model dimensions, training updates, experiment selection and aggregation, and frozen evaluation. The [README](README.md) gives the setup and commands.

## Results reproduced from scratch

The clean run reproduced these numbers from newly extracted features and newly trained checkpoints. The feature archive and split manifest matched the original hashes. Final evaluation used the ten new baseline checkpoints. The tuning comparisons used validation only.

| Model | Five-seed validation accuracy | Five-seed test accuracy | Corrected train–validation gap |
|---|---:|---:|---:|
| Shallow | 68.27% ± 1.53 pp | 68.27% ± 4.28 pp | 24.64 pp |
| Deep | 69.73% ± 3.42 pp | 66.13% ± 2.33 pp | 28.03 pp |

The deep baseline had higher mean validation accuracy and lower mean test accuracy in this comparison. The corrected gap compares training and validation using the same frozen checkpoint in evaluation mode. During optimization, training accuracy is measured as weights change and dropout may be active.

For the deep model, mean best validation accuracy was 62.13%, 69.73%, and 68.27% at learning rates 0.0001, 0.001, and 0.01 with dropout zero. At learning rate 0.001 and dropout 0, 0.2, and 0.5, the validation means were 69.73%, 70.00%, and 65.87%. This submission uses these optional configurations to discuss validation behavior and reserves final test reporting for the shallow and deep baselines.

The small dataset, known repetitions, potential artist overlap, questionable genre labels, and 15 held-out recordings per genre limit interpretation. Sample standard deviations reflect five training seeds on a single split. The comparison describes this experiment and does not establish statistical significance or isolate the effect of depth.

## Where to find results

- `runs/training/shallow/` and `runs/training/deep/` contain the rebuilt baseline checkpoints, histories, curves, and metadata by training seed.
- `runs/training/comparison.csv` has the six validation summaries.
- `runs/evaluation/` contains the newly computed baseline held-out and evaluation-mode metrics and corresponding figures.
- `evaluation-plan.md` describes the baseline-only held-out protocol used for this run.
