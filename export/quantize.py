"""Static QDQ INT8 quantisation of the embedder ONNX graph.

Static, not dynamic: the embedder is almost entirely Conv2d, and ONNX Runtime's
dynamic path only touches MatMul/Gemm, so it would leave the whole network in
FP32. Static QDQ with a real calibration pass over log-mels quantises the
convolutions, which is what the Arm 8-bit dot product accelerates.

Calibration mels come from --calib-dir (any folder of wavs; DCASE train clips or
engine-sounds). Reports FP32 vs INT8 size and a rough MAC count.

    python export/quantize.py --fp32 models/injini_mn10_as_fp32.onnx \
        --calib-dir data/dcase2025_dev/fan/train --n-calib 128
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))
import features as F  # noqa: E402


class MelCalib(CalibrationDataReader):
    def __init__(self, wavs, input_name):
        self.wavs = wavs
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.wavs):
            return None
        mel = F.logmel(F.load_audio(self.wavs[self.i]), fixed=True)
        self.i += 1
        return {self.input_name: mel[None, None, :, :].astype(np.float32)}


def onnx_macs(path: str) -> int:
    """Rough MAC estimate: Conv and Gemm only, from static shapes where present."""
    m = onnx.load(path)
    macs = 0
    init = {i.name for i in m.graph.initializer}
    shapes = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        d = vi.type.tensor_type.shape.dim
        shapes[vi.name] = [x.dim_value if x.dim_value > 0 else 1 for x in d]
    w = {i.name: list(i.dims) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Conv":
            wt = next((w[i] for i in n.input if i in w), None)
            out = shapes.get(n.output[0])
            if wt and out and len(wt) == 4 and len(out) == 4:
                cout, cin, kh, kw = wt
                _, _, oh, ow = out
                macs += cout * cin * kh * kw * oh * ow
        elif n.op_type in ("Gemm", "MatMul"):
            wt = next((w[i] for i in n.input if i in w), None)
            if wt and len(wt) == 2:
                macs += wt[0] * wt[1]
    return macs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or args.fp32.replace("_fp32.onnx", "_int8.onnx")
    pre = args.fp32.replace("_fp32.onnx", "_fp32_preproc.onnx")
    quant_pre_process(args.fp32, pre, skip_symbolic_shape=True)

    wavs = sorted(glob.glob(os.path.join(args.calib_dir, "**", "*.wav"), recursive=True))
    if not wavs:
        raise SystemExit(f"no calibration wavs under {args.calib_dir}")
    rng = np.random.RandomState(42)
    wavs = list(rng.choice(wavs, size=min(args.n_calib, len(wavs)), replace=False))

    iname = ort.InferenceSession(pre, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quantize_static(
        pre, out, MelCalib(wavs, iname),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )

    fp32_mb = os.path.getsize(args.fp32) / 1e6
    int8_mb = os.path.getsize(out) / 1e6
    report = {
        "fp32_onnx": args.fp32, "int8_onnx": out,
        "fp32_mb": round(fp32_mb, 3), "int8_mb": round(int8_mb, 3),
        "size_ratio": round(fp32_mb / int8_mb, 2),
        "approx_macs": onnx_macs(args.fp32),
        "n_calibration": len(wavs),
        "quant": "static QDQ INT8, per-channel weights, MinMax",
    }
    rp = out.replace(".onnx", "_quant_report.json")
    json.dump(report, open(rp, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("wrote", rp)


if __name__ == "__main__":
    main()
