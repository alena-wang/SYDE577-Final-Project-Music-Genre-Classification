"""Build a reusable feature dataset and an auditable split manifest."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile

from .dataset import LABELS, assign_splits, digest, discover, duplicate_groups
from .scaling import standardization
from .features import CACHE_VERSION, FEATURE_NAMES, extract


def cached_features(path, destination):
    if destination.exists():
        return np.load(destination, allow_pickle=False)
    features = extract(path)
    with NamedTemporaryFile(dir=destination.parent, suffix=".npy", delete=False) as file:
        temp = Path(file.name)
        try:
            np.save(file, features)
            file.flush()
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
    return features


def preprocess(raw, output, seed):
    recordings = discover(raw)
    cache = output / "cache" / CACHE_VERSION
    cache.mkdir(parents=True, exist_ok=True)
    hashes = {key: digest(path) for key, path in recordings.items()}
    groups = duplicate_groups(hashes)
    rows, vectors = [], []
    for index, (key, path) in enumerate(recordings.items(), 1):
        genre = key.split(".")[0]
        row = dict(recording=key, path=str(path.relative_to(raw)), genre=genre,
                   label=LABELS[genre], sha256=hashes[key], group=groups[key],
                   split="", exclusion="")
        try:
            # Detect the known malformed mirror file without invoking a fallback decoder.
            if key == "jazz.00054":
                soundfile.info(path)
            vectors.append(cached_features(path, cache / f"{hashes[key]}.npy"))
        except soundfile.LibsndfileError as error:
            if key != "jazz.00054":
                raise
            row["exclusion"] = str(error)
            print(f"Excluded unreadable recording: {path}", flush=True)
        rows.append(row)
        if index % 50 == 0:
            print(f"Processed {index}/{len(recordings)} recordings", flush=True)

    included = [row for row in rows if not row["exclusion"]]
    splits = assign_splits({row["recording"]: row["group"] for row in included}, seed)
    for row in included:
        row["split"] = splits[row["recording"]]
    partitions = np.array([row["split"] for row in included])
    labels = np.array([row["label"] for row in included])
    if any(set(labels[partitions == split]) != set(LABELS.values())
           for split in ("train", "validation", "test")):
        raise ValueError("Every partition must contain all ten genres.")
    features = np.stack(vectors)
    mean, scale = standardization(features[partitions == "train"])
    np.savez_compressed(output / "dataset.npz", features=features, labels=labels,
                        split=partitions, mean=mean, scale=scale,
                        feature_names=FEATURE_NAMES, split_seed=seed,
                        recordings=[row["recording"] for row in included])
    with (output / "splits.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(dict(Counter(partitions)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--split-seed", type=int, default=577)
    args = parser.parse_args()
    preprocess(args.raw, args.output, args.split_seed)


if __name__ == "__main__":
    main()
