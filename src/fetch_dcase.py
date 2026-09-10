"""Download and unpack the DCASE 2025 Task 2 development set (Zenodo 15097779).

CC BY-NC-SA 4.0. Seven machine-type zips, ~2.3 GB total. Unpacks to
<out>/<machine>/{train,test}/*.wav, the layout dcase_data.py expects.

    python src/fetch_dcase.py --out data/dcase2025_dev
    python src/fetch_dcase.py --out data/dcase2025_dev --machines bearing fan
"""
from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile

RECORD = "https://zenodo.org/records/15097779/files"
MACHINES = ["bearing", "fan", "gearbox", "slider", "valve", "ToyCar", "ToyTrain"]


def fetch(out: str, machines: list[str]) -> None:
    os.makedirs(out, exist_ok=True)
    for m in machines:
        zpath = os.path.join(out, f"dev_{m}.zip")
        if not os.path.exists(zpath):
            url = f"{RECORD}/dev_{m}.zip?download=1"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, zpath)
        print(f"unpacking {zpath}")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    # Zenodo zips unpack as <out>/<machine>/{train,test}; some editions nest an
    # extra top folder. Flatten if needed.
    for m in machines:
        nested = os.path.join(out, m, m)
        if os.path.isdir(nested):
            for sub in os.listdir(nested):
                os.replace(os.path.join(nested, sub), os.path.join(out, m, sub))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dcase2025_dev")
    ap.add_argument("--machines", nargs="*", default=MACHINES)
    args = ap.parse_args()
    fetch(args.out, args.machines)
    print("done")
