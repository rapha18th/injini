"""Turn the Kaggle 'zeyadzsm/engine-sounds' dump into a source-split manifest.

Two problems with that dataset, both handled here:

1. It ships augmented variants next to their source clips, with names like
   ``1_augmented_10_Alternator Bearing Noise.wav``, ``1_segment_0_augmented.wav``,
   ``002_Engine-A_augmented_1.wav``. Splitting those at file level leaks
   near-identical audio across train and test. Every file is reduced to a
   ``source key`` (strip augmentation / segment / trailing-index suffixes) and
   the split is done on the (class, source key) pair.
2. Two near-duplicate top folders (``Data/Data_Fixed`` and
   ``Data_AA/Data_Fixed``). Both are read; the source key is taken relative to
   the class folder so a clip appearing in both maps to one key.

Output: data/engine_sounds_manifest.csv with columns
    path, label, class_name, source_key, split
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re

AUG_PATTERNS = [
    r"_augmented(_\d+)?(_[A-Za-z].*)?$",
    r"_segment_\d+(_augmented.*)?$",
    r"_aug(_\d+)?$",
]
TRAIL_IDX = re.compile(r"(_\d+)+$")


def source_key(stem: str) -> str:
    s = stem
    for pat in AUG_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = TRAIL_IDX.sub("", s)
    return s.strip() or stem


def collect(root: str) -> list[dict]:
    """Only files under a literal Data_Fixed/<class>/... path count.

    root is not trusted to be the engine-sounds dataset alone: on Kaggle it can
    be a shared mount point (/kaggle/input/datasets) holding other attached
    datasets too, glob(**/*.wav) picks all of them up, and a fallback class
    name (parts[-2]) silently turns unrelated audio (birds, rain, traffic...)
    into bogus fault classes. Requiring the Data_Fixed segment, and skipping a
    file where the very next segment is itself "Data_Fixed" (a doubly-nested
    path with no class folder), rejects everything that isn't really this
    dataset instead of guessing a class for it.
    """
    rows = []
    classes = set()
    skipped_no_marker = skipped_bad_nesting = 0
    for wav in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        rel = os.path.relpath(wav, root).replace("\\", "/")
        parts = rel.split("/")
        try:
            di = parts.index("Data_Fixed")
            cls = parts[di + 1]
        except (ValueError, IndexError):
            skipped_no_marker += 1
            continue
        if cls == "Data_Fixed" or di + 1 >= len(parts) - 1:
            skipped_bad_nesting += 1
            continue
        classes.add(cls)
        stem = os.path.splitext(parts[-1])[0]
        rows.append({"path": wav, "class_name": cls, "source_key": f"{cls}::{source_key(stem)}"})
    if skipped_no_marker or skipped_bad_nesting:
        print(f"  skipped {skipped_no_marker} files with no Data_Fixed path segment "
              f"(likely another dataset sharing the mount), "
              f"{skipped_bad_nesting} with a malformed nested path")
    labels = {c: i for i, c in enumerate(sorted(classes))}
    for r in rows:
        r["label"] = labels[r["class_name"]]
    return rows


def split(rows: list[dict], val_frac=0.2, seed=42) -> list[dict]:
    rng = random.Random(seed)
    keys_by_class: dict[str, list[str]] = {}
    for r in rows:
        keys_by_class.setdefault(r["class_name"], set()).add(r["source_key"])
    val_keys: set[str] = set()
    for cls, keys in keys_by_class.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_frac))
        val_keys.update(keys[:n_val])
    for r in rows:
        r["split"] = "val" if r["source_key"] in val_keys else "train"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing Data/ and Data_AA/")
    ap.add_argument("--out", default="data/engine_sounds_manifest.csv")
    args = ap.parse_args()

    rows = split(collect(args.root))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "class_name", "source_key", "split"])
        w.writeheader()
        w.writerows(rows)

    n_tr = sum(r["split"] == "train" for r in rows)
    n_va = sum(r["split"] == "val" for r in rows)
    n_keys = len({r["source_key"] for r in rows})
    print(f"{len(rows)} files  {n_keys} source keys  ->  train {n_tr} / val {n_va}")
    by_cls: dict[str, int] = {}
    for r in rows:
        by_cls[r["class_name"]] = by_cls.get(r["class_name"], 0) + 1
    for c, n in sorted(by_cls.items()):
        print(f"  {n:5d}  {c}")


if __name__ == "__main__":
    main()
