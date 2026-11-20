"""Train an MLP baseline or run a training-only smoke test."""

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import GENRES
from .model import DEEP_WIDTHS, SHALLOW_WIDTHS, MLP, ShallowClassifier


def run_epoch(model, batches, loss_function, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    with torch.set_grad_enabled(training):
        for features, labels in batches:
            if training:
                optimizer.zero_grad()
            logits = model(features)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count += labels.numel()
            total_loss += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / count, correct / count


def fit(model, train_batches, validation_batches, output, *, epochs, patience,
        checkpoint_metadata, learning_rate=0.001, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history = []
    best_accuracy, best_epoch = -1, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_batches, loss_function, optimizer)
        row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                   validation_loss="", validation_accuracy="")
        if validation_batches is not None:
            loss, accuracy = run_epoch(model, validation_batches, loss_function)
            row.update(validation_loss=loss, validation_accuracy=accuracy)
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                torch.save(dict(model_state=model.state_dict(), epoch=epoch,
                                validation_accuracy=accuracy, **checkpoint_metadata),
                           output / "best.pt")
        history.append(row)
        if verbose:
            print(row, flush=True)
        if validation_batches is not None and epoch - best_epoch >= patience:
            break
    return history, best_epoch


def write_history(history, output):
    from matplotlib.figure import Figure

    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    figure = Figure(figsize=(10, 4), layout="constrained")
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(figure.subplots(1, 2), ("loss", "accuracy")):
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="Training")
        if history[0][f"validation_{metric}"] != "":
            axis.plot(epochs, [row[f"validation_{metric}"] for row in history],
                      label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize())
        axis.legend()
    figure.savefig(output / "curves.png", dpi=150)


def train(dataset, output, seed, epochs, batch_size, patience, smoke, *,
          architecture="shallow", hidden_sizes=DEEP_WIDTHS, dropout=0.0,
          learning_rate=0.001, verbose=True):
    started = perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.load(dataset, allow_pickle=False) as data:
        train_indices = np.flatnonzero(data["split"] == "train")
        labels = data["labels"]
        features = data["features"]
        mean, scale = data["mean"], data["scale"]
        feature_names = data["feature_names"].tolist()
        split_seed = int(data["split_seed"])
        if smoke:
            train_indices = np.concatenate([
                rng.choice(train_indices[labels[train_indices] == label], size=4, replace=False)
                for label in range(len(GENRES))
            ])
        else:
            validation_indices = np.flatnonzero(data["split"] == "validation")

    def batches(indices, shuffle=False):
        values = ((features[indices] - mean) / scale).astype(np.float32)
        selected = TensorDataset(torch.from_numpy(values), torch.from_numpy(labels[indices]))
        return DataLoader(selected, batch_size=batch_size, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(seed))

    output.mkdir(parents=True, exist_ok=False)
    if architecture == "shallow":
        hidden_sizes, dropout = SHALLOW_WIDTHS, 0.0
        model = ShallowClassifier()
    else:
        model = MLP(hidden_sizes, dropout)
    model_config = dict(architecture=architecture, hidden_sizes=list(hidden_sizes), dropout=dropout)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Trainable parameters: {parameters}", flush=True)
    metadata = dict(genres=list(GENRES), feature_names=feature_names,
                    mean=mean.tolist(), scale=scale.tolist(), seed=seed,
                    split_seed=split_seed, parameter_count=parameters, **model_config)
    history, best_epoch = fit(
        model, batches(train_indices, shuffle=True),
        None if smoke else batches(validation_indices), output,
        epochs=min(epochs, 2) if smoke else epochs, patience=patience,
        checkpoint_metadata=metadata, learning_rate=learning_rate, verbose=verbose,
    )
    write_history(history, output)
    best = history[best_epoch - 1] if best_epoch else {}
    summary = dict(seed=seed, split_seed=split_seed, parameter_count=parameters,
                   batch_size=batch_size, learning_rate=learning_rate, patience=patience,
                   train_examples=len(train_indices), smoke=smoke,
                   epochs_run=len(history), best_epoch=best_epoch or None,
                   best_validation_accuracy=best.get("validation_accuracy"),
                   best_validation_loss=best.get("validation_loss"),
                   train_accuracy_at_best_epoch=best.get("train_accuracy"),
                   accuracy_gap_pp=100 * (best["train_accuracy"] - best["validation_accuracy"])
                   if best else None,
                   duration_seconds=perf_counter() - started, max_epochs=epochs, **model_config)
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.npz"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=577)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", choices=("shallow", "deep"), default="shallow")
    parser.add_argument("--hidden-sizes", type=int, nargs="+")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or min(args.batch_size, args.patience) < 1:
        parser.error("Choose 1–100 epochs and positive batch size and patience.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite.")
    if args.model == "shallow" and (args.hidden_sizes or args.dropout):
        parser.error("--hidden-sizes and --dropout are for the deep model.")
    hidden_sizes = args.hidden_sizes or DEEP_WIDTHS
    if min(hidden_sizes) < 1 or not 0 <= args.dropout < 1:
        parser.error("Use positive hidden sizes and dropout in [0, 1).")
    run_name = args.model
    if args.smoke:
        run_name = "smoke" if args.model == "shallow" else "deep-smoke"
    output = args.output or Path("runs") / f"{run_name}-{args.seed}"
    train(args.dataset, output, args.seed, args.epochs, args.batch_size, args.patience, args.smoke,
          architecture=args.model, hidden_sizes=hidden_sizes, dropout=args.dropout,
          learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
