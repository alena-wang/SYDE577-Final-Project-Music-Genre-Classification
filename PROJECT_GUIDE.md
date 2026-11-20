# SYDE 577 group handoff

**Music Genre Classification Using Shallow and Deep Neural Networks**

The project classifies GTZAN recordings into ten genres using fixed-length audio features and fully connected PyTorch networks. Start with this guide to understand the project, the [README](README.md) for commands, and the [final results summary](runs/phase4/results-summary.md) for report-ready analysis.

## What the four phases did

| Phase | Work completed | Main outputs |
|---|---|---|
| 1 | Built the 71-feature dataset and fixed duplicate-aware split. Ran a smoke check and trained the shallow baseline. | [Processed data](Data/processed/), [shallow run](runs/shallow-577/) |
| 2 | Trained the deep baseline with the same settings and compared the two seed-577 checkpoints on validation. | [Deep run](runs/deep-577/), [baseline comparison](runs/comparison-577/) |
| 3 | Trained eight configurations with five seeds each, for 40 runs. Selected checkpoints using validation accuracy. | [Experiment plan](runs/phase3/plan.json), [per-run results](runs/phase3/per-run.csv), [aggregate results](runs/phase3/comparison.csv) |
| 4 | Reproduced validation metrics for all 40 checkpoints, recomputed train–validation gaps in evaluation mode, and evaluated 25 pre-specified checkpoints on the held-out test set. | [Locked evaluation plan](runs/phase4/evaluation-plan.md), [final results](runs/phase4/results-summary.md), [presentation summary](runs/phase4/presentation-summary.md) |

## Dataset and folder history

The original `Data/` directory contained `genres_original/` (1,000 WAV recordings), `images_original/`, `features_30_sec.csv`, and `features_3_sec.csv`. The pipeline reads the WAVs and computes its own features. The images and supplied CSVs are unused. The project code did not rewrite these source files.

Phase 1 created `Data/raw/` and `Data/processed/`. On the current Mac, the commands spell the path `data/` and it resolves to the same directory as `Data/`. On a case-sensitive filesystem, pass paths with their actual capitalization.

| Generated path | Purpose |
|---|---|
| `Data/raw/genres_original` | Symbolic link to `Data/genres_original`. The WAVs exist in one place. |
| `Data/processed/cache/` | Cached per-recording feature vectors, keyed by audio file hash and extractor version. |
| `Data/processed/dataset.npz` | Array archive with raw feature vectors, feature names, label IDs, split assignments, recording IDs, split seed, and training-only scaling statistics. This is the input to training and evaluation. |
| `Data/processed/splits.csv` | Readable manifest with the path, genre, label, content hash, duplicate group, split, and any exclusion for each recording. |

One unreadable clip, `jazz.00054.wav`, was excluded, leaving **999 recordings**. The split seed is **577**. The fixed split has **699 training, 150 validation, and 150 test recordings**. Each held-out partition has 15 recordings per genre. The splitter groups documented same-recording repetitions and byte-identical files, including matches that connect through another recording. Whole groups stay in the same partition. Residual artist overlap and questionable labels remain benchmark limitations.

The integer label order is **blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock**, starting at zero. Each audio file is resampled to mono at 22,050 Hz. The extractor computes 20 MFCCs, 12 chroma bins, spectral centroid, RMS energy, and zero-crossing rate. There are 35 measurements over time, with a mean and standard deviation for each. One global tempo estimate supplies feature 71. The model receives one row per recording. Each feature is standardized with the training mean and standard deviation. Validation and test use the same saved statistics.

## How training and evaluation work

1. The MLP maps 71 features through fully connected layers with ReLU to ten output logits. Optional dropout follows each hidden activation. Cross-entropy takes the raw logits. Softmax is available separately for probability output.
2. Adam trains on shuffled mini-batches of 32. The baseline learning rate is 0.001, the epoch limit is 100, and early-stopping patience is 12. Widths, dropout, learning rate, seed, batch size, epoch limit, and patience are configurable.
3. After each epoch, the model scores the validation rows. The checkpoint is saved only when validation accuracy strictly improves, so the first epoch at the best accuracy wins ties. The test partition plays no role in this decision.
4. Training accuracy in the epoch history is measured on batches while weights are changing and dropout is active where configured. Validation runs with frozen weights and dropout disabled. Phase 4 reevaluates training and validation with the same frozen weights in `model.eval()` and `torch.no_grad()`. Subtracting those two accuracies gives a comparable train–validation gap.
5. Final test evaluation uses all five training seeds for each pre-specified configuration. It records accuracy, mean cross-entropy, and macro precision, recall, and F1. Macro metrics weight genres equally. Each confusion matrix sums counts across the five seeds for one configuration. The 750 predictions come from the same 150 recordings scored five times.

## Python file map

| File | Role |
|---|---|
| [genre/__init__.py](genre/__init__.py) | Marks the directory as a Python package. |
| [genre/dataset.py](genre/dataset.py) | Discovers recordings, maps genres to IDs, hashes files, groups duplicates, and assigns splits. |
| [genre/features.py](genre/features.py) | Defines the 71 features and extracts one feature vector from one recording with librosa. |
| [genre/preprocess.py](genre/preprocess.py) | Processes recordings, caches feature vectors, logs exclusions, and saves the dataset and manifest. |
| [genre/scaling.py](genre/scaling.py) | Computes training-only feature means and scales. Constant training features use scale 1. |
| [genre/model.py](genre/model.py) | Defines the configurable MLP and the 64-neuron shallow baseline. |
| [genre/train.py](genre/train.py) | Loads the split and scaler, makes mini-batches, runs training and validation, saves the first best checkpoint, and writes metrics and curves. |
| [genre/compare.py](genre/compare.py) | Reproduces the original shallow/deep validation results from the saved checkpoints and writes validation confusion matrices. |
| [genre/experiments.py](genre/experiments.py) | Defines the Phase 3 matrix, launches independent seeded runs through the shared trainer, and summarizes validation results. It checks the saved plan before resuming. |
| [genre/evaluate.py](genre/evaluate.py) | Audits the 40 Phase 3 checkpoints, writes corrected evaluation-mode metrics, and scores the 25 locked test checkpoints after checking input hashes. |
| [genre/evaluation_plots.py](genre/evaluation_plots.py) | Draws the aggregate test confusion matrices, validation-versus-test error bars, and the validation-only tuning figure. |

Python creates `__pycache__/` folders as needed. Finder creates `.DS_Store` files. Neither contains project source or experiment results.

### Tests

| File | What it verifies |
|---|---|
| [tests/test_phase1.py](tests/test_phase1.py) | Phase 1 and 2 checks for label mapping, duplicate-aware split integrity, actual librosa feature dimensions, scaling, MLP outputs and parameter counts, and training updates versus validation behavior. |
| [tests/test_phase3.py](tests/test_phase3.py) | Argument parsing, parameter counts across configurations, dropout in training and evaluation, and experiment aggregation. |
| [tests/test_phase4.py](tests/test_phase4.py) | Evaluation mode, disabled gradients, unchanged weights, loss across unequal batches, and macro metrics when a class has no predictions. |

The last full suite passed **13 tests**. Running these Python tests is separate from scoring the models on the held-out GTZAN partition.

## Saved runs and how to read them

| Directory | Contents |
|---|---|
| [runs/smoke-577/](runs/smoke-577/) | Two epochs on 40 training clips, four per genre. A pipeline check with training metrics only. |
| [runs/shallow-577/](runs/shallow-577/) | Original shallow run. Best validation accuracy **66.0%**, epoch **9**, with early stopping at epoch 21. |
| [runs/deep-577/](runs/deep-577/) | Original deep run. Best validation accuracy **71.3%**, epoch **17**, with early stopping at epoch 29. |
| [runs/comparison-577/](runs/comparison-577/) | Validation-only summary and confusion matrices for the preceding two original runs. |
| [runs/phase3-integration/](runs/phase3-integration/) | A two-epoch dropout 0.2 trial that checked the runner. This trial is excluded from the 40-run comparison. |
| [runs/phase3/](runs/phase3/) | Eight configuration directories, each with `seed-577` through `seed-581`. The new seed-577 runs reproduced the original baseline histories. This folder also holds the plan, per-run and aggregate CSVs, and training curves. |
| [runs/phase4/](runs/phase4/) | The held-out plan, hashes, corrected train/validation metrics, test metrics, per-genre metrics, figures, and summaries. Models are loaded from Phase 3. |

A full training run has `best.pt` for selected weights and associated metadata, `history.csv` for epoch-by-epoch loss and accuracy, `curves.png` for those trends, and `run.json` for settings and selected metrics. Use the original run folders to discuss the single-seed baselines and the Phase 3/4 summaries to discuss performance across seeds.

## Results at a glance

All accuracies are five-seed means ± sample standard deviation. The standard deviation is in percentage points. These are variations across training seeds on one fixed split. The corrected gap is the mean evaluation-mode training accuracy minus validation accuracy at the selected checkpoint.

| Configuration | Hidden widths | LR | Dropout | Parameters | Validation | Held-out test | Corrected gap (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| Shallow | 64 | 0.001 | 0 | 5,258 | 68.27% ± 1.53 | 68.27% ± 4.28 | 24.64 |
| Intermediate | 128, 64 | 0.001 | 0 | 18,122 | 70.13% ± 1.10 | 68.13% ± 1.73 | 28.87 |
| Deep | 128, 64, 32 | 0.001 | 0 | 19,882 | 69.73% ± 3.42 | 66.13% ± 2.33 | 28.03 |
| Deep, lower LR | 128, 64, 32 | 0.0001 | 0 | 19,882 | 62.13% ± 3.57 | Validation only | 13.32 |
| Deep, higher LR | 128, 64, 32 | 0.01 | 0 | 19,882 | 68.27% ± 1.01 | Validation only | 28.53 |
| Deep, dropout 0.2 | 128, 64, 32 | 0.001 | 0.2 | 19,882 | 70.00% ± 2.26 | 64.53% ± 1.52 | 25.51 |
| Deep, dropout 0.5 | 128, 64, 32 | 0.001 | 0.5 | 19,882 | 65.87% ± 3.21 | Validation only | 17.68 |
| Parameter-matched shallow | 243 | 0.001 | 0 | 19,936 | 68.80% ± 2.23 | 71.07% ± 2.56 | 29.14 |

The original seed-577 deep run led the shallow run on validation. Across five seeds, the deep baseline still had a higher mean validation score, while the shallow baseline had a higher observed mean test score. Comparing those two architectures changes depth, widths, and parameter count together. The 243-unit shallow model and the deep model differ by just 54 parameters and give a closer comparison of model size, with different layer layouts. The learning-rate and dropout comparisons vary one setting at a time within the deep architecture.

The deep and intermediate models reached roughly 98–99% training accuracy at the selected checkpoints while their validation accuracy was near 70%. Deep validation loss rose from checkpoint selection to early stopping as training loss fell, consistent with overfitting. The low-learning-rate run had lower training and validation accuracy, consistent with slower or incomplete fitting under the chosen early-stopping setup. Dropout 0.5 had an apparent train–validation gap of only 1.89 pp in the original history, which rose to 17.68 pp when both splits were scored in evaluation mode.

The dataset is small, with known repetitions and questionable labels. Each validation or test genre has just 15 recordings. The search covers a few settings, and the five seeds share the same split. Treat performance differences as descriptive results for this study.

### Files to use in the report

- [Phase 4 results and interpretation](runs/phase4/results-summary.md) and [short presentation summary](runs/phase4/presentation-summary.md)
- [Corrected training and validation metrics by run](runs/phase4/eval-mode-per-run.csv) and [aggregate](runs/phase4/eval-mode-comparison.csv)
- [Held-out per-seed metrics](runs/phase4/test-per-run.csv), [configuration summary](runs/phase4/test-comparison.csv), and [per-genre metrics](runs/phase4/test-per-genre.csv)
- [Validation and test error bars](runs/phase4/validation-vs-test.png), [validation-only tuning figure](runs/phase4/validation-tuning.png), [learning curves](runs/phase3/learning-curves.png), and [test confusion matrices](runs/phase4/test-confusion.png)

## If the group only wants shallow and deep

**No code changes are required to work with just the two baselines.** Use the `shallow` and `deep` rows of the existing Phase 3/4 CSVs for the multi-seed results. Use `genre.compare` for the original two seed-577 validation runs. Both architectures already use the same model and trainer, so there are no separate intermediate, wide, or dropout training methods to remove. Filtering the existing results also avoids unnecessary training or test scoring.

For a future **validation-only** rerun, the experiment runner already supports a two-configuration, ten-run subset. The first command previews it. The second starts training in a separate directory.

```sh
source .venv/bin/activate
python -m genre.experiments --configs shallow deep --dry-run
python -m genre.experiments --configs shallow deep --output runs/shallow-deep-only
```

If the group later wants a permanently smaller codebase, the places to simplify are the extra entries in `CONFIGURATIONS` and any plots or presentation code that assume those entries exist. Optional dropout support and its dedicated test could be removed from a baseline-only version, although dropout is zero in both baselines and adds no dropout layers or training cost there. Keep the shared model, trainer, preprocessing, normalization, and core tests. Changing the Phase 3 defaults would also require preserving the original eight-entry manifest separately, because the Phase 4 audit checks it. Keep the original Phase 4 plan and results as the record of the completed held-out study. Any reduced view can be made from the saved metrics.

## Setup and verification

Run commands from the project root. Activate the existing virtual environment, or follow the [README](README.md) to create it and install `requirements.txt`.

```sh
source .venv/bin/activate
python -m unittest discover -s tests -v
shasum -a 256 -c runs/phase4/original-sha256.txt
```

The checksum list was captured before Phase 4 and verified against 191 original files. If sharing by Git, check [the ignore rules](.gitignore) first. The run outputs, processed data, and local virtual environment may be absent from a fresh checkout. Share the saved results the team needs alongside this guide.
