"""DCASE 2025 Task 2 (development set) directory reader.

Expected layout under ``--root`` (as distributed):

    <root>/<machine>/train/section_00_source_train_normal_0001_<attrs>.wav
    <root>/<machine>/test/section_00_source_test_normal_0001_<attrs>.wav
    <root>/<machine>/test/section_00_target_test_anomaly_0002_<attrs>.wav

Train is normal only. Test mixes normal and anomaly across source and target
domains. Machine types in the 2025 dev set: ToyCar, ToyTrain, bearing, fan,
gearbox, slider, valve.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

DEV_MACHINES = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]


@dataclass
class Clip:
    path: str
    machine: str
    section: str
    domain: str          # "source" | "target"
    split: str           # "train" | "test"
    label: int           # 0 normal, 1 anomaly


def _parse(path: str, machine: str) -> Clip | None:
    base = os.path.basename(path).lower()
    if not base.endswith(".wav"):
        return None
    parts = base.split("_")
    try:
        section = "section_" + parts[1]
    except IndexError:
        section = "section_00"
    domain = "target" if "target" in base else "source"
    split = "train" if "train" in base else "test"
    label = 1 if "anomaly" in base else 0
    return Clip(path, machine, section, domain, split, label)


def load(root: str, machines: list[str] | None = None) -> list[Clip]:
    machines = machines or DEV_MACHINES
    clips: list[Clip] = []
    for m in machines:
        for sub in ("train", "test"):
            for p in sorted(glob.glob(os.path.join(root, m, sub, "*.wav"))):
                c = _parse(p, m)
                if c is not None:
                    clips.append(c)
    return clips


def by_machine(clips: list[Clip]) -> dict[str, list[Clip]]:
    out: dict[str, list[Clip]] = {}
    for c in clips:
        out.setdefault(c.machine, []).append(c)
    return out
