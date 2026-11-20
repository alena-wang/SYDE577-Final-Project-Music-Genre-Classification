"""Compare the saved best checkpoints using only validation recordings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from .dataset import GENRES
from .model import MLP, SHALLOW_WIDTHS


def evaluate_checkpoint(run, features, labels, dataset_metadata):
    metadata = json.loads((run / "run.json").read_text())
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    for key, expected in dataset_metadata.items():
        if not np.array_equal(checkpoint[key], expected):
            raise ValueError(f"{run}: checkpoint has different {key} from the saved dataset.")
    model = MLP(checkpoint.get("hidden_sizes", SHALLOW_WIDTHS), checkpoint.get("dropout", 0.0))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.inference_mode():
        predictions = model(features).argmax(dim=1)
    count = len(GENRES)
    matrix = torch.bincount(labels * count + predictions, minlength=count**2)
    matrix = matrix.reshape(count, count).numpy()

    with (run / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    epoch = checkpoint["epoch"]
    best = next(row for row in history if int(row["epoch"]) == epoch)
    training_accuracy = float(best["train_accuracy"])
    validation_accuracy = float(np.trace(matrix) / matrix.sum())
    if (epoch != metadata["best_epoch"] or
            not np.isclose(validation_accuracy, float(best["validation_accuracy"])) or
            not np.isclose(validation_accuracy, checkpoint["validation_accuracy"])):
        raise ValueError(f"{run}: checkpoint and recorded best metrics differ.")
    return dict(run=run.name, parameter_count=sum(p.numel() for p in model.parameters()),
                best_epoch=epoch, training_accuracy_at_best_epoch=training_accuracy,
                best_validation_accuracy=validation_accuracy,
                accuracy_gap_pp=100 * (training_accuracy - validation_accuracy),
                epochs_run=len(history)), matrix


def compare(dataset, runs, output):
    with np.load(dataset, allow_pickle=False) as data:
        validation = data["split"] == "validation"
        mean, scale = data["mean"], data["scale"]
        features = torch.from_numpy(((data["features"][validation] - mean) / scale).astype(np.float32))
        labels = torch.from_numpy(data["labels"][validation])
        dataset_metadata = dict(mean=mean, scale=scale, split_seed=int(data["split_seed"]),
                                genres=GENRES, feature_names=data["feature_names"])

    results = [evaluate_checkpoint(run, features, labels, dataset_metadata) for run in runs]
    output.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13, 6), layout="constrained")
    for axis, (summary, matrix) in zip(figure.subplots(1, 2), results):
        with (output / f"{summary['run']}-confusion.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["actual / predicted", *GENRES])
            writer.writerows(zip(GENRES, *matrix.T))
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.sum(axis=1).max())
        axis.set(title=summary["run"], xlabel="Predicted genre", ylabel="Actual genre")
        axis.set_xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
        axis.set_yticks(range(len(GENRES)), GENRES)
        for (row, column), count in np.ndenumerate(matrix):
            axis.text(column, row, str(count), ha="center", va="center",
                      color="white" if count > matrix.sum(axis=1).max() / 2 else "black")
    figure.savefig(output / "validation-confusion.png", dpi=150)

    summaries = [summary for summary, _ in results]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.npz"))
    parser.add_argument("--shallow-run", type=Path, default=Path("runs/shallow-577"))
    parser.add_argument("--deep-run", type=Path, default=Path("runs/deep-577"))
    parser.add_argument("--output", type=Path, default=Path("runs/comparison-577"))
    args = parser.parse_args()
    compare(args.dataset, (args.shallow_run, args.deep_run), args.output)


if __name__ == "__main__":
    main()
