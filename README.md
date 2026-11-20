# SYDE 577 · Music Genre Classification

Phase 1 builds a recording-level GTZAN feature dataset and trains a shallow fully connected baseline. The held-out test partition is reserved for evaluation after model selection. The deep model, hyperparameter comparisons, and five-seed experimental reporting belong to a later phase.

## Setup in VS Code

Open the project folder and run these commands in the integrated terminal. The workspace has Apple Silicon Python 3.9.6, which has compatible releases for the pinned dependencies. Python 3.11 is also supported. In VS Code, use **Python: Select Interpreter** to select `.venv/bin/python` when the Python extension is available.

```sh
cd /Users/alenawang/SYDE577_final_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
python -m unittest discover -s tests -v
```

## GTZAN audio

This workspace already contains the 1,000-file WAV mirror. The local link `data/raw/genres_original` points to `Data/genres_original`. To restore the link in a copy of this workspace that still has `Data/genres_original`, run the following from the project root.

```sh
mkdir -p data/raw
ln -s ../../Data/genres_original data/raw/genres_original
```

If the audio is absent, download the ZIP from the [GTZAN WAV mirror](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and place the extracted `genres_original` directory under `data/raw/`. The loader also accepts the original `genres` tree with AU files or the ten genre directories directly under `data/raw/`.

## Preprocess

```sh
python -m genre.preprocess --raw data/raw --output data/processed --split-seed 577
```

The first run decodes about eight hours of source audio and computes features on CPU. Budget several minutes to tens of minutes depending on the machine and first-call compilation. Progress appears every 50 clips. Completed feature vectors are cached by file hash and extractor version in `data/processed/cache/` and can be reused after interruptions. Increment `CACHE_VERSION` in `genre/features.py` if you change feature definitions.

Librosa resamples to mono at 22,050 Hz. Each recording contributes 70 statistics from frame-level means and population standard deviations for 20 MFCCs, 12 chroma bins, spectral centroid, RMS energy, and zero-crossing rate. A single global tempo in BPM gives 71 features. The class IDs follow the fixed order blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

The splitter merges byte-identical files and the documented exact and same-recording repetitions, including cross-genre repeats. It keeps each connected group together and targets 70/15/15 within each genre. The known malformed `jazz.00054.wav` is logged as excluded if the decoder rejects it. Artist overlap, versions of songs by different performers, and questionable labels remain limitations of this recording-level benchmark. [Sturm's GTZAN fault catalog, Table 2](https://arxiv.org/pdf/1306.1461#page=9)

- `data/processed/splits.csv` records file paths, labels, hashes, duplicate groups, partitions, and exclusions. Keep the same manifest and split seed for future model comparisons.
- `data/processed/dataset.npz` stores raw features, feature names, labels, partitions, and training-only means and standard deviations. Training applies those statistics to training and validation rows. A constant training feature uses a scale of 1.

## Smoke test and baseline training

Run these after preprocessing. The smoke test draws four training recordings per class, trains for two epochs at most, and writes only training metrics. It should take seconds once dependencies and the feature cache are ready. The full run uses at most 100 epochs on fewer than 1,000 feature vectors. Expect seconds to a few minutes on CPU, depending on the machine.

```sh
python -m genre.train --smoke --seed 577 --output runs/smoke-577
python -m genre.train --seed 577 --epochs 100 --batch-size 32 --patience 12 --output runs/shallow-577
```

Each run requires a fresh output directory. The baseline has 5,258 trainable parameters and uses a 71 → 64 → 10 architecture with ReLU. Adam uses learning rate 0.001. Cross-entropy takes the raw logits, and `predict_proba` applies softmax for probability output.

Both runs save `history.csv`, `curves.png`, and `run.json`. The full run saves `best.pt` when validation accuracy strictly improves and stops after 12 epochs without improvement by default. The checkpoint includes the model state, feature order, label order, and standardization statistics. Inspect `curves.png` for falling training loss alongside rising validation loss or a widening accuracy gap. Keep the test partition held out while selecting settings or comparing configurations.

## Phase 2 deep baseline

Reuse the saved `data/processed/dataset.npz` and the existing shallow run. From the project root with the virtual environment active, run:

```sh
python -m unittest discover -s tests -v
python -m genre.train --model deep --hidden-sizes 128 64 32 --dropout 0 --seed 577 --epochs 100 --batch-size 32 --patience 12 --output runs/deep-577
python -m genre.compare --shallow-run runs/shallow-577 --deep-run runs/deep-577 --output runs/comparison-577
```

Use a fresh training output directory for a rerun, then pass its path to `--deep-run` when comparing. The deep model uses ReLU in each hidden layer and returns 10 logits. It has 19,882 trainable parameters, compared with 5,258 for the shallow model. The comparison changes both depth and parameter count. Dropout is zero for this baseline.

The comparison reloads both best checkpoints and predicts only on the fixed validation split using the saved training normalization. It saves a metrics CSV, a confusion matrix CSV per model, and a side-by-side matrix PNG. The training–validation gap is measured in percentage points at each model's best validation epoch, using the training accuracy recorded during that epoch. Confusion matrix rows are actual genres and columns are predicted genres.

## Phase 3 multi-seed experiments

Phase 3 holds the processed dataset and split seed fixed. Each of the eight configurations trains from scratch on seeds 577 through 581. The default settings are Adam, batch size 32, 100 epochs maximum, and patience 12. Selection uses the first epoch that attains the highest validation accuracy. Baselines from earlier phases stay in their original directories.

```sh
python -m unittest discover -s tests -v
python -m genre.experiments --dry-run
python -m genre.experiments --configs deep-dropout-0.2 --seeds 577 --epochs 2 --output runs/phase3-integration
python -m genre.experiments --output runs/phase3
```

The grid writes `plan.json` with the dataset hash and complete experiment settings, `per-run.csv` with selected-epoch metrics and durations, and `comparison.csv` with mean best validation accuracy, sample standard deviation, mean training–validation gap, and parameter count. Accuracy and its standard deviation are in percent and percentage points, respectively. `validation-accuracy.png` shows mean accuracy with one sample standard deviation on either side. `learning-curves.png` overlays all seeds through their respective stopping epochs. Each configuration and seed directory also contains its checkpoint, history, curves, and metadata. Duration covers loading, training, and writing the training history and curves.

Repeat the grid command with the same plan to skip completed runs and regenerate the summaries. Use a fresh output directory to retrain the entire grid or change the plan. For a single custom MLP, `python -m genre.train --model deep` accepts any list of positive hidden widths via `--hidden-sizes`, including one width, and accepts `--learning-rate` and `--dropout`. Use `--seed`, `--epochs`, `--batch-size`, and `--patience` to control training.

The deep learning-rate and dropout comparisons each vary one setting relative to the deep baseline. The original depth comparison also changes width and parameter count. The matched-width shallow model has 19,936 parameters versus 19,882 for the deep baseline, so that comparison controls model size approximately while changing the layer layout. Gap values use training accuracy recorded during optimization, with dropout active where configured, versus validation accuracy measured in evaluation mode. Negative gaps can therefore occur under dropout.

## Phase 4 final evaluation

The completed results are in `runs/phase4/`. Read `evaluation-plan.md` for the locked set of five configurations and all five training seeds. `results-summary.md` contains the report tables and interpretation, and `presentation-summary.md` gives the short version.

The evaluation CLI has separate audit and test stages. The audit verifies the Phase 3 inventory and validation metrics and evaluates the training and validation splits in evaluation mode. The test stage checks the audited dataset, plan, and checkpoint hashes before scoring the 25 locked checkpoints. Both stages use saved training-only normalization and frozen weights. The test ranking is a final descriptive result.

To reproduce the completed evaluation in a separate directory, from the project root with the virtual environment active:

```sh
mkdir runs/phase4-reproduction
cp runs/phase4/evaluation-plan.md runs/phase4-reproduction/
python -m genre.evaluate audit --output runs/phase4-reproduction
python -m genre.evaluate test --output runs/phase4-reproduction
python -m unittest discover -s tests -v
shasum -a 256 -c runs/phase4/original-sha256.txt
```
