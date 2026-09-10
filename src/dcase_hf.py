"""DCASE 2025 Task 2 dev set via the HuggingFace mirror HTill/dcase2025_task2_dev.

Zenodo (the official host) has been returning 504s, so the eval reads the
mirror instead. Same content: a train split (normal only) and a test split
(normal + anomaly), tagged with machine_type, domain and section.

    from dcase_hf import load, by_machine
    clips = load()                      # list[Clip], .wave is float32 @ features.SR
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import features as F  # noqa: E402

REPO = "HTill/dcase2025_task2_dev"
DEV_MACHINES = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]


@dataclass
class Clip:
    machine: str
    section: str
    domain: str          # "source" | "target"
    split: str           # "train" | "test"
    label: int           # 0 normal, 1 anomaly
    wave: np.ndarray = field(repr=False)


def _decode(row, names) -> Clip | None:
    lab = names["label"][row["label"]]
    if lab == "unknown":
        return None
    dom = names["domain"][row["domain"]]
    if dom == "unknown":
        dom = "source"
    audio = row["audio"]
    y = np.asarray(audio["array"], dtype=np.float32)
    y = F.resample_linear(y, audio["sampling_rate"], F.SR)
    return Clip(
        machine=names["machine_type"][row["machine_type"]],
        section=str(row.get("section", "00")),
        domain=dom,
        split=row["split"],
        label=1 if lab == "anomaly" else 0,
        wave=y,
    )


def load(splits=("train", "test"), machines: list[str] | None = None, limit=None) -> list[Clip]:
    from datasets import load_dataset

    ds = load_dataset(REPO)
    names = {
        "label": ds["test"].features["label"].names,
        "domain": ds["test"].features["domain"].names,
        "machine_type": ds["test"].features["machine_type"].names,
    }
    want = set(machines or DEV_MACHINES)
    out: list[Clip] = []
    for split in splits:
        d = ds[split]
        for i, row in enumerate(d):
            if limit and i >= limit:
                break
            c = _decode(row, names)
            if c is not None and c.machine in want:
                c.split = split
                out.append(c)
    return out


def by_machine(clips: list[Clip]) -> dict[str, list[Clip]]:
    out: dict[str, list[Clip]] = {}
    for c in clips:
        out.setdefault(c.machine, []).append(c)
    return out
