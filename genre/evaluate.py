"""Audit Phase 3 checkpoints and evaluate the locked Phase 4 test set."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch import nn

from .dataset import GENRES, digest
from .experiments import CONFIGURATIONS, SEEDS, aggregate, write_csv
from .features import FEATURE_NAMES
from .model import MLP

FINAL_CONFIGS = ("shallow", "intermediate", "deep", "shallow-matched", "deep-dropout-0.2")
ARTIFACTS = ("best.pt", "run.json", "history.csv", "curves.png")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate(model, features, labels, batch_size=32):
    """Score frozen weights and return loss and true-by-predicted counts."""
    model.eval()
    count = len(GENRES)
    total_loss = 0.0
    confusion = torch.zeros((count, count), dtype=torch.long)
    with torch.no_grad():
        for x, y in zip(features.split(batch_size), labels.split(batch_size)):
            logits = model(x)
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            confusion += torch.bincount(y * count + logits.argmax(1), minlength=count**2).reshape(
                count, count)
    return total_loss / len(labels), confusion.numpy()


def classification_scores(confusion):
    correct = confusion.diagonal()
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = correct / np.maximum(predicted, 1)
    recall = correct / np.maximum(actual, 1)
    f1 = 2 * correct / np.maximum(actual + predicted, 1)
    return {
        "accuracy": float(correct.sum() / confusion.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }, (precision, recall, f1)


def tensors(dataset, split):
    selected = dataset["split"] == split
    scaled = (dataset["features"][selected] - dataset["mean"]) / dataset["scale"]
    return torch.from_numpy(scaled.astype(np.float32)), torch.from_numpy(dataset["labels"][selected])


def load_checkpoint(run, dataset, config, seed):
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    expected = dict(architecture=config.name, hidden_sizes=list(config.hidden_sizes),
                    dropout=config.dropout, seed=seed, split_seed=int(dataset["split_seed"]),
                    genres=GENRES, feature_names=FEATURE_NAMES,
                    mean=dataset["mean"], scale=dataset["scale"])
    for key, value in expected.items():
        require(np.array_equal(checkpoint[key], value), f"{run}: checkpoint {key} mismatch")
    model = MLP(config.hidden_sizes, config.dropout)
    model.load_state_dict(checkpoint["model_state"])
    parameters = sum(p.numel() for p in model.parameters())
    require(parameters == checkpoint["parameter_count"], f"{run}: parameter count mismatch")
    return model, checkpoint


def audit(dataset, source, output):
    plan = json.loads((source / "plan.json").read_text())
    settings = dict(split_seed=577, batch_size=32, max_epochs=100, patience=12, smoke=False)
    require(all(plan[k] == v for k, v in settings.items()), "Phase 3 training settings mismatch")
    require(plan["seeds"] == list(SEEDS), "Phase 3 seeds mismatch")
    require(plan["configurations"] == {c.name: c.settings() for c in CONFIGURATIONS},
            "Phase 3 configurations mismatch")
    require(int(dataset["split_seed"]) == 577 and dataset["features"].shape == (999, 71)
            and dataset["feature_names"].tolist() == FEATURE_NAMES, "Feature dataset mismatch")
    require(Counter(dataset["split"]) == {"train": 699, "validation": 150, "test": 150},
            "Dataset partition sizes mismatch")

    rows = read_csv(source / "per-run.csv")
    expected_pairs = {(c.name, seed) for c in CONFIGURATIONS for seed in SEEDS}
    require(len(rows) == len(expected_pairs), "Phase 3 per-run row count mismatch")
    reported = {(row["configuration"], int(row["seed"])): row for row in rows}
    require(reported.keys() == expected_pairs, "Phase 3 per-run configuration or seed mismatch")
    expected_paths = {source / name / f"seed-{seed}" / "best.pt" for name, seed in expected_pairs}
    require(set(source.rglob("best.pt")) == expected_paths, "Phase 3 checkpoint inventory mismatch")

    train_data, validation_data = tensors(dataset, "train"), tensors(dataset, "validation")
    records, original = [], []
    for config in CONFIGURATIONS:
        for seed in SEEDS:
            run = source / config.name / f"seed-{seed}"
            require(all((run / name).is_file() for name in ARTIFACTS), f"{run}: incomplete artifacts")
            result = json.loads((run / "run.json").read_text())
            expected = dict(settings, seed=seed, train_examples=699, architecture=config.name,
                            **config.settings())
            require(all(result[k] == v for k, v in expected.items()), f"{run}: run settings mismatch")
            row = reported[config.name, seed]
            require(all(str(value) == row[key] for key, value in result.items()),
                    f"{run}: per-run CSV differs from metadata")
            history = read_csv(run / "history.csv")
            best = max(history, key=lambda epoch: float(epoch["validation_accuracy"]))
            model, checkpoint = load_checkpoint(run, dataset, config, seed)
            require(len(history) == result["epochs_run"]
                    and int(best["epoch"]) == checkpoint["epoch"] == result["best_epoch"]
                    and checkpoint["parameter_count"] == result["parameter_count"],
                    f"{run}: epoch or parameter metadata mismatch")
            validation_loss, validation_matrix = evaluate(model, *validation_data)
            validation_accuracy = classification_scores(validation_matrix)[0]["accuracy"]
            require(validation_accuracy == float(best["validation_accuracy"])
                    == checkpoint["validation_accuracy"] == result["best_validation_accuracy"],
                    f"{run}: could not reproduce validation accuracy")
            require(math.isclose(validation_loss, result["best_validation_loss"], abs_tol=1e-6)
                    and float(best["validation_loss"]) == result["best_validation_loss"],
                    f"{run}: could not reproduce validation loss")
            require(float(best["train_accuracy"]) == result["train_accuracy_at_best_epoch"],
                    f"{run}: original training accuracy mismatch")
            train_loss, train_matrix = evaluate(model, *train_data)
            train_accuracy = classification_scores(train_matrix)[0]["accuracy"]
            original.append(dict(configuration=config.name, **result))
            records.append(dict(
                configuration=config.name, seed=seed, parameter_count=result["parameter_count"],
                learning_rate=config.learning_rate, dropout=config.dropout,
                hidden_sizes=list(config.hidden_sizes), best_epoch=checkpoint["epoch"],
                epochs_run=len(history), original_train_accuracy=float(best["train_accuracy"]),
                train_accuracy=train_accuracy, train_loss=train_loss,
                validation_accuracy=validation_accuracy, validation_loss=validation_loss,
                accuracy_gap_pp=100 * (train_accuracy - validation_accuracy),
                final_train_loss=float(history[-1]["train_loss"]),
                final_validation_loss=float(history[-1]["validation_loss"]),
            ))
    summary_rows = read_csv(source / "comparison.csv")
    summary = aggregate(original)
    require(len(summary_rows) == len(summary) and all(
        all(str(value) == row[key] for key, value in calculated.items())
        for calculated, row in zip(summary, summary_rows)), "Phase 3 aggregate CSV mismatch")

    summaries = []
    for config in CONFIGURATIONS:
        group = [r for r in records if r["configuration"] == config.name]
        summaries.append(dict(
            configuration=config.name, **config.settings(), runs=len(group),
            parameter_count=group[0]["parameter_count"],
            mean_train_accuracy_pct=100 * mean(r["train_accuracy"] for r in group),
            mean_validation_accuracy_pct=100 * mean(r["validation_accuracy"] for r in group),
            sd_validation_accuracy_pp=100 * stdev(r["validation_accuracy"] for r in group),
            mean_train_loss=mean(r["train_loss"] for r in group),
            mean_validation_loss=mean(r["validation_loss"] for r in group),
            mean_accuracy_gap_pp=mean(r["accuracy_gap_pp"] for r in group),
            mean_best_epoch=mean(r["best_epoch"] for r in group),
        ))
    write_csv(output / "eval-mode-per-run.csv", records)
    write_csv(output / "eval-mode-comparison.csv", summaries)
    print("Reproduced all 40 validation accuracies and losses and verified the Phase 3 aggregate.")


def test_checkpoints(dataset, source, output):
    from .evaluation_plots import confusion_plot, comparison_plots

    features, labels = tensors(dataset, "test")
    require(len(labels) == 150, "Expected exactly 150 held-out test recordings")
    per_run, summaries, genres, matrices = [], [], [], {}
    selected = {config.name: config for config in CONFIGURATIONS}
    for name in FINAL_CONFIGS:
        config = selected[name]
        scores, genre_scores, matrix_sum = [], [], np.zeros((len(GENRES), len(GENRES)), dtype=int)
        for seed in SEEDS:
            run = source / name / f"seed-{seed}"
            model, checkpoint = load_checkpoint(run, dataset, config, seed)
            loss, matrix = evaluate(model, features, labels)
            metrics, per_genre = classification_scores(matrix)
            scores.append(dict(test_loss=loss, **metrics))
            genre_scores.append(per_genre)
            matrix_sum += matrix
            per_run.append(dict(configuration=name, seed=seed, best_epoch=checkpoint["epoch"],
                                parameter_count=checkpoint["parameter_count"],
                                **config.settings(), **scores[-1]))
        summaries.append(dict(
            configuration=name, runs=len(scores), parameter_count=checkpoint["parameter_count"],
            mean_test_accuracy_pct=100 * mean(r["accuracy"] for r in scores),
            sd_test_accuracy_pp=100 * stdev(r["accuracy"] for r in scores),
            **{f"mean_{key}": mean(r[key] for r in scores)
               for key in ("test_loss", "macro_precision", "macro_recall", "macro_f1")},
        ))
        precision, recall, f1 = np.mean(genre_scores, axis=0)
        for index, genre in enumerate(GENRES):
            genres.append(dict(configuration=name, genre=genre,
                               support_recordings=int(matrix_sum[index].sum() // len(SEEDS)),
                               mean_precision=precision[index], mean_recall=recall[index],
                               mean_f1=f1[index]))
        with (output / f"test-confusion-{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["actual / predicted (five-seed count sum)", *GENRES])
            writer.writerows(zip(GENRES, *matrix_sum.T))
        matrices[name] = matrix_sum
    write_csv(output / "test-per-run.csv", per_run)
    write_csv(output / "test-comparison.csv", summaries)
    write_csv(output / "test-per-genre.csv", genres)
    confusion_plot(output, matrices)
    comparison_plots(output, read_csv(source / "comparison.csv"), summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "test"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.npz"))
    parser.add_argument("--source", type=Path, default=Path("runs/phase3"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4"))
    args = parser.parse_args()
    require(args.source.resolve() != args.output.resolve(), "Use a separate Phase 4 output")
    plan = args.output / "evaluation-plan.md"
    fingerprint = dict(dataset=digest(args.dataset), plan=digest(plan),
                       phase3_plan=digest(args.source / "plan.json"))
    phase3_plan = json.loads((args.source / "plan.json").read_text())
    require(fingerprint["dataset"] == phase3_plan["dataset_sha256"], "Dataset hash mismatch")
    marker = args.output / "audit.json"
    with np.load(args.dataset, allow_pickle=False) as dataset:
        if args.stage == "audit":
            require(not marker.exists(), "Audit already complete")
            audit(dataset, args.source, args.output)
            fingerprint["checkpoints"] = {
                str(path.relative_to(args.source)): digest(path)
                for path in sorted(args.source.rglob("best.pt"))
            }
            marker.write_text(json.dumps(fingerprint, indent=2) + "\n")
        else:
            audited = json.loads(marker.read_text())
            require(all(audited[key] == value for key, value in fingerprint.items()),
                    "Dataset or evaluation plan changed after audit")
            require(all(digest(args.source / path) == expected
                        for path, expected in audited["checkpoints"].items()),
                    "Checkpoint changed after audit")
            require(not (args.output / "test-per-run.csv").exists(), "Test evaluation already saved")
            test_checkpoints(dataset, args.source, args.output)


if __name__ == "__main__":
    main()
