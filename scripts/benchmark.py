"""Benchmark all exported ONNX models for CPU latency and storage size."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import ONNX_DIR, RESULTS_DIR

REPS = 30
WARMUP = 5

rows = []
for onnx_path in sorted(ONNX_DIR.glob("*.onnx")):
    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.random.randn(*meta["input_shape"]).astype(np.float32)
    for _ in range(WARMUP):
        session.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(REPS):
        session.run(None, {name: x})
    latency = 1000.0 * (time.perf_counter() - t0) / REPS
    rows.append({
        "model": onnx_path.stem,
        "latency_ms": round(latency, 3),
        "throughput_fps": round(1000.0 / latency, 1),
        "size_kb": round(onnx_path.stat().st_size / 1024.0, 1),
        "opset": meta.get("opset"),
        "parity_max_abs_diff": meta.get("parity_max_abs_diff"),
    })

out = RESULTS_DIR / "benchmark.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out} ({len(rows)} models)")
for row in rows:
    print(f"  {row['model']:24s} {row['latency_ms']:8.3f} ms  {row['size_kb']:10.1f} KB")
