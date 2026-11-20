"""Recording discovery, duplicate grouping, and stratified partitioning."""

import hashlib
import random
from collections import Counter, defaultdict

GENRES = ("blues", "classical", "country", "disco", "hiphop", "jazz",
          "metal", "pop", "reggae", "rock")
LABELS = {name: index for index, name in enumerate(GENRES)}
SPLITS = ("train", "validation", "test")
RATIOS = (0.70, 0.15, 0.15)

# Exact and same-recording repetitions from Sturm (2014), Table 2.
# The metal/rock pair below is byte-identical in the local WAV mirror.
# Pop 37 is described in the Distortions column.
REPETITIONS = {
    "classical": [(42, 53), (51, 80)],
    "country": [(8, 51), (52, 60)],
    "disco": [(50, 51, 70), (55, 60, 89), (71, 74), (98, 99), (38, 78)],
    "hiphop": [(39, 45), (76, 78), (1, 42), (46, 65), (47, 67),
               (48, 68), (49, 69), (50, 72)],
    "jazz": [(33, 51), (34, 53), (35, 55), (36, 58), (37, 60), (38, 62),
             (39, 65), (40, 67), (42, 68), (43, 69), (44, 70), (45, 71), (46, 72)],
    "metal": [(4, 13), (34, 94), (40, 61), (41, 62), (42, 63), (43, 64),
              (44, 65), (45, 66)],
    "pop": [(15, 22), (30, 31), (45, 46), (47, 80), (52, 57), (54, 60),
            (56, 59), (67, 71), (87, 90), (68, 73), (15, 21, 22, 37),
            (47, 48, 51), (52, 54), (57, 60)],
    "reggae": [(3, 54), (5, 56), (8, 57), (10, 60), (13, 58), (41, 69),
               (73, 74), (80, 81, 82), (75, 91, 92), (7, 59), (33, 44), (85, 96)],
}
KNOWN_GROUPS = [
    tuple(f"{genre}.{number:05d}" for number in group)
    for genre, groups in REPETITIONS.items()
    for group in groups
] + [("metal.00058", "rock.00016")]


def discover(raw):
    """Read the canonical genre tree or either common archive layout."""
    for root in (raw, raw / "genres", raw / "genres_original"):
        if all((root / genre).is_dir() for genre in GENRES):
            break
    else:
        raise ValueError(f"No GTZAN genre tree under {raw}. See README.md.")
    recordings = {}
    for genre in GENRES:
        paths = sorted(p for p in (root / genre).iterdir()
                       if p.suffix.lower() in {".wav", ".au"})
        if not paths:
            raise ValueError(f"No recordings for {genre} under {root}")
        for path in paths:
            if not path.stem.startswith(f"{genre}.") or path.stem in recordings:
                raise ValueError(f"Invalid or repeated GTZAN recording ID: {path}")
            recordings[path.stem] = path
    return recordings


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def duplicate_groups(hashes):
    """Combine documented matches and byte-identical files transitively."""
    parents = {recording: recording for recording in hashes}

    def find(recording):
        while parents[recording] != recording:
            recording = parents[recording]
        return recording

    identical = defaultdict(list)
    for recording, sha in hashes.items():
        identical[sha].append(recording)
    for group in [*KNOWN_GROUPS, *identical.values()]:
        roots = sorted({find(recording) for recording in group if recording in parents})
        for root in roots[1:]:
            parents[root] = roots[0]
    return {recording: find(recording) for recording in parents}


def assign_splits(groups, seed=577):
    """Place larger groups first, minimizing squared per-genre count error."""
    members = defaultdict(list)
    for recording, group in sorted(groups.items()):
        members[group].append(recording)
    ordered = list(members.values())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    totals = Counter(recording.split(".")[0] for recording in groups)
    counts = [Counter() for _ in SPLITS]
    assignments = {}
    for group in ordered:
        labels = Counter(recording.split(".")[0] for recording in group)

        def cost(index):
            # Difference in squared error if this group joins the split.
            return sum(n * (2 * (counts[index][name] - RATIOS[index] * totals[name]) + n)
                       for name, n in labels.items())

        index = min(range(len(SPLITS)), key=cost)
        counts[index].update(labels)
        assignments.update(dict.fromkeys(group, SPLITS[index]))
    return assignments
