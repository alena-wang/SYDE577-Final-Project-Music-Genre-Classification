# SYDE 577 · Music Genre Classification

This clean reproducibility run compares two fully connected classifiers on 71 GTZAN audio features.

| Model | Architecture | Trainable parameters |
|---|---|---:|
| Shallow | 71 → 64 → 10 | 5,258 |
| Deep | 71 → 128 → 64 → 32 → 10 | 19,882 |

The fixed, duplicate-aware split uses seed 577 and contains 699 training, 150 validation, and 150 test recordings. Both baselines use ReLU, Adam at 0.001, cross-entropy, batch size 32, at most 100 epochs, patience 12, and dropout zero. The five training seeds are 577–581.

## Setup

From the project root, use a fresh virtual environment. This clean run extracted only code, tests, and documentation from the submission ZIP. The raw WAVs were read through a local symbolic link. The feature cache, dataset, checkpoints, tables, and figures were generated here. The local virtual environment uses the installed dependency versions from the submission environment.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For a clean run, obtain the [GTZAN WAV mirror](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and place or link `genres_original` under `data/raw/`. Start with an empty processed directory.

```sh
python -m genre.preprocess --raw data/raw --output data/processed --split-seed 577
```

## Training and validation

The default grid selects **shallow and deep across five seeds**. The existing `runs/training/` directory contains the six-configuration reproducibility run. Use a fresh output path for a two-baseline run. A matching plan in the same directory resumes completed runs.

```sh
python -m genre.experiments --dry-run
python -m genre.experiments --output runs/baseline-training
```

The optional experiments use the deep model. Select the deep baseline alongside the variants to compare learning rates (0.0001, 0.001, 0.01) or dropout (0, 0.2, 0.5) on validation:

```sh
python -m genre.experiments --configs deep deep-lr-0.0001 deep-lr-0.01 --output runs/learning-rate
python -m genre.experiments --configs deep deep-dropout-0.2 deep-dropout-0.5 --output runs/dropout
```

For one training run, use `python -m genre.train --model shallow --seed 577 --output runs/shallow-example` or change `--model` to `deep` and specify a new output directory. Each full run saves the first checkpoint with the highest validation accuracy, a CSV history, learning curves, and run metadata.

## Final evaluation

All 30 checkpoints in this directory were trained from the freshly rebuilt data. The baseline-only held-out evaluation follows the [evaluation plan](evaluation-plan.md). The [project guide](PROJECT_GUIDE.md) provides background on the submitted study.

The complete grid and evaluation were run with the following commands in a clean output location. To score the saved checkpoints again here, substitute a fresh evaluation directory for `runs/evaluation` in the setup commands and pass it to both evaluator stages with `--output`:

```sh
python -m genre.experiments --configs shallow deep deep-lr-0.0001 deep-lr-0.01 deep-dropout-0.2 deep-dropout-0.5 --output runs/training
mkdir runs/evaluation
cp evaluation-plan.md runs/evaluation/
python -m genre.evaluate audit --source runs/training
python -m genre.evaluate test --source runs/training
```

See [the validation comparison](runs/training/comparison.csv), [the baseline test comparison](runs/evaluation/test-comparison.csv), and [the per-seed test metrics](runs/evaluation/test-per-run.csv). Regenerated feature and split files matched the original study's hashes. All 14 unit tests passed.

Evaluation checks the fixed dataset hash, verifies checkpoint architecture and normalization against the data, and reproduces validation metrics before test scoring. Test results include cross-entropy, accuracy, macro precision, recall, and F1. In `test-confusion-matrices.png`, rows are actual genres and columns are predicted genres. The diagonal counts correct predictions. Each panel sums five seeds scoring the same 150 recordings. The plots show sample standard deviations across the five seeds on the same split. Test results are for final reporting. Use validation for architecture or hyperparameter selection.
