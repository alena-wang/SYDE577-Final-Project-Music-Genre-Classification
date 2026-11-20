"""Report figures for the frozen Phase 4 evaluation."""

import numpy as np
from matplotlib.figure import Figure

from .dataset import GENRES


def confusion_plot(output, matrices):
    figure = Figure(figsize=(16, 10), layout="constrained")
    figure.suptitle("Held-out confusion matrices | Counts summed across five seeds")
    axes = figure.subplots(2, 3).flat
    highest = max(matrix.sum(axis=1).max() for matrix in matrices.values())
    for axis, (name, matrix) in zip(axes, matrices.items()):
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=highest)
        axis.set(title=name, xlabel="Predicted genre", ylabel="Actual genre")
        axis.set_xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
        axis.set_yticks(range(len(GENRES)), GENRES)
        for (row, column), count in np.ndenumerate(matrix):
            axis.text(column, row, str(count), ha="center", va="center", fontsize=8,
                      color="white" if count > highest / 2 else "black")
    figure.axes[-1].axis("off")
    figure.axes[-1].text(0.5, 0.5, "Each matrix contains 750 predictions\n"
                         "from 150 recordings scored five times.",
                         ha="center", va="center", transform=figure.axes[-1].transAxes)
    figure.savefig(output / "test-confusion.png", dpi=180)


def comparison_plots(output, validation, test):
    by_config = {row["configuration"]: row for row in validation}
    figure = Figure(figsize=(10, 5), layout="constrained")
    axis = figure.subplots()
    y = np.arange(len(test))
    validation_rows = [by_config[row["configuration"]] for row in test]
    axis.errorbar([float(r["mean_best_validation_accuracy_pct"]) for r in validation_rows],
                  y - 0.12, xerr=[float(r["sd_best_validation_accuracy_pp"]) for r in validation_rows],
                  fmt="o", capsize=4, label="Validation", color="tab:blue")
    axis.errorbar([r["mean_test_accuracy_pct"] for r in test], y + 0.12,
                  xerr=[r["sd_test_accuracy_pp"] for r in test],
                  fmt="s", capsize=4, label="Held-out test", color="tab:orange")
    axis.set_yticks(y, [r["configuration"] for r in test])
    axis.invert_yaxis()
    axis.set_xlabel("Accuracy (%), mean ± sample SD across five seeds")
    axis.grid(axis="x", alpha=0.3)
    axis.legend()
    figure.savefig(output / "validation-vs-test.png", dpi=180)

    figure = Figure(figsize=(10, 4.5), layout="constrained")
    panels = (
        ("Learning rate · dropout 0", ("deep-lr-0.0001", "deep", "deep-lr-0.01"),
         ("0.0001", "0.001", "0.01")),
        ("Dropout · learning rate 0.001", ("deep", "deep-dropout-0.2", "deep-dropout-0.5"),
         ("0", "0.2", "0.5")),
    )
    for axis, (title, names, labels) in zip(figure.subplots(1, 2, sharey=True), panels):
        rows = [by_config[name] for name in names]
        axis.errorbar(range(len(rows)), [float(r["mean_best_validation_accuracy_pct"]) for r in rows],
                      yerr=[float(r["sd_best_validation_accuracy_pp"]) for r in rows],
                      fmt="o", capsize=5, color="tab:blue")
        axis.set_xticks(range(len(rows)), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
    figure.axes[0].set_ylabel("Best validation accuracy (%)")
    figure.suptitle("Deep MLP tuning on validation data | Mean ± sample SD across five seeds")
    figure.savefig(output / "validation-tuning.png", dpi=180)
