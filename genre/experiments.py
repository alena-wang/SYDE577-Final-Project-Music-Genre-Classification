"""Run and summarize the fixed Phase 3 validation experiment matrix."""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import numpy as np
import torch
from matplotlib.figure import Figure

from .dataset import digest
from .model import DEEP_WIDTHS, SHALLOW_WIDTHS
from .train import train


@dataclass(frozen=True)
class Configuration:
    name: str
    hidden_sizes: tuple
    learning_rate: float = 0.001
    dropout: float = 0.0

    def settings(self):
        return dict(hidden_sizes=list(self.hidden_sizes), learning_rate=self.learning_rate,
                    dropout=self.dropout)


CONFIGURATIONS = (
    Configuration("shallow", SHALLOW_WIDTHS),
    Configuration("deep", DEEP_WIDTHS),
    Configuration("intermediate", (128, 64)),
    Configuration("deep-lr-0.0001", DEEP_WIDTHS, learning_rate=0.0001),
    Configuration("deep-lr-0.01", DEEP_WIDTHS, learning_rate=0.01),
    Configuration("deep-dropout-0.2", DEEP_WIDTHS, dropout=0.2),
    Configuration("deep-dropout-0.5", DEEP_WIDTHS, dropout=0.5),
    Configuration("shallow-matched", (243,)),
)
SEEDS = (577, 578, 579, 580, 581)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase3"))
    parser.add_argument("--configs", nargs="+", choices=[c.name for c in CONFIGURATIONS])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100 or min(args.batch_size, args.patience) < 1:
        parser.error("Choose 1–100 epochs and positive batch size and patience.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds):
        parser.error("Choose distinct nonnegative seeds below 2**63.")
    if args.configs and len(set(args.configs)) != len(args.configs):
        parser.error("Choose distinct configurations.")
    args.configs = [c for c in CONFIGURATIONS if args.configs is None or c.name in args.configs]
    return args


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["configuration"]].append(record)
    summaries = []
    for name, runs in groups.items():
        first = runs[0]
        accuracies = [100 * run["best_validation_accuracy"] for run in runs]
        summaries.append(dict(
            configuration=name, hidden_sizes=first["hidden_sizes"],
            learning_rate=first["learning_rate"], dropout=first["dropout"],
            parameter_count=first["parameter_count"], runs=len(runs),
            mean_best_validation_accuracy_pct=mean(accuracies),
            sd_best_validation_accuracy_pp=stdev(accuracies) if len(runs) > 1 else None,
            mean_accuracy_gap_pp=mean(run["accuracy_gap_pp"] for run in runs),
        ))
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output, summaries, histories):
    ranked = sorted(summaries, key=lambda row: row["mean_best_validation_accuracy_pct"])
    figure = Figure(figsize=(9, max(3, len(ranked) * 0.55)), layout="constrained")
    axis = figure.subplots()
    for index, row in enumerate(ranked):
        axis.errorbar(row["mean_best_validation_accuracy_pct"], index,
                      xerr=row["sd_best_validation_accuracy_pp"], fmt="o", capsize=4,
                      color="tab:blue")
    axis.set_yticks(range(len(ranked)), [row["configuration"] for row in ranked])
    axis.set_xlabel("Best validation accuracy (%), mean ± sample SD")
    axis.grid(axis="x", alpha=0.3)
    figure.savefig(output / "validation-accuracy.png", dpi=160)

    figure = Figure(figsize=(13, 2.6 * len(histories)), layout="constrained")
    axes = figure.subplots(len(histories), 2, squeeze=False, sharex=True)
    figure.suptitle("Learning curves by configuration (one line per seed)")
    for (name, runs), row in zip(histories.items(), axes):
        for axis, metric in zip(row, ("accuracy", "loss")):
            for index, history in enumerate(runs):
                epochs = [int(r["epoch"]) for r in history]
                for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
                    axis.plot(epochs, [float(r[f"{split}_{metric}"]) for r in history],
                              color=color, alpha=0.7, linewidth=1,
                              label=split.capitalize() if index == 0 else None)
            axis.set(ylabel=f"{name}\n{metric.capitalize()}")
            if metric == "accuracy":
                axis.set_ylim(0, 1)
    axes[0, 0].legend(loc="lower right")
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    figure.savefig(output / "learning-curves.png", dpi=150)


def run_grid(args):
    with np.load(args.dataset, allow_pickle=False) as dataset:
        split_seed = int(dataset["split_seed"])
    settings = dict(split_seed=split_seed, batch_size=args.batch_size,
                    max_epochs=args.epochs, patience=args.patience, smoke=False)
    manifest = dict(settings, dataset_sha256=digest(args.dataset), seeds=list(args.seeds),
                    configurations={c.name: c.settings() for c in args.configs},
                    torch_version=str(torch.__version__), torch_threads=torch.get_num_threads())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "plan.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Output contains a different experiment plan. Choose a fresh --output.")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Choose an empty output directory for a new experiment plan.")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    records, histories = [], defaultdict(list)
    total = len(args.configs) * len(args.seeds)
    for config in args.configs:
        for seed in args.seeds:
            run = args.output / config.name / f"seed-{seed}"
            run.parent.mkdir(exist_ok=True)
            expected = dict(settings, seed=seed, **config.settings())
            if run.exists():
                result = json.loads((run / "run.json").read_text())
                if any(result.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Run configuration mismatch: {run}")
                if not all((run / name).is_file() for name in ("best.pt", "history.csv", "curves.png")):
                    raise ValueError(f"Incomplete run: {run}")
            else:
                with TemporaryDirectory(dir=run.parent) as folder:
                    staged = Path(folder) / "run"
                    result = train(args.dataset, staged, seed, args.epochs, args.batch_size,
                                   args.patience, False, architecture=config.name,
                                   hidden_sizes=config.hidden_sizes, dropout=config.dropout,
                                   learning_rate=config.learning_rate, verbose=False)
                    staged.rename(run)
            records.append(dict(configuration=config.name, **result))
            with (run / "history.csv").open(newline="") as stream:
                histories[config.name].append(list(csv.DictReader(stream)))
            print(f"{len(records)}/{total} {config.name} seed {seed}: "
                  f"{result['best_validation_accuracy']:.1%}, "
                  f"{result['duration_seconds']:.2f}s", flush=True)
    summaries = aggregate(records)
    write_csv(args.output / "per-run.csv", records)
    write_csv(args.output / "comparison.csv", summaries)
    write_plots(args.output, summaries, histories)


def main():
    args = parse_args()
    print(f"{len(args.configs) * len(args.seeds)} runs; seeds: {list(args.seeds)}", flush=True)
    for config in args.configs:
        print(f"{config.name}: {config.settings()}", flush=True)
    if not args.dry_run:
        run_grid(args)


if __name__ == "__main__":
    main()
