"""Export every trained checkpoint to ONNX and record parity/metadata."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy.export import export_onnx
from src.paths import CKPT_DIR
from src.deploy.runtime import Predictor

RUNS = ["teacher_resnet50", "baseline_mbv2", "baseline_squeezenet", "baseline_efflite0",
        "kd_mbv2", "kd_squeezenet", "kd_efflite0", "nas_best"]

for run in RUNS:
    path = CKPT_DIR / run / "best.pt"
    if not path.exists():
        print(f"skip {run}: no checkpoint")
        continue
    state = torch.load(path, map_location="cpu", weights_only=False)
    meta = state.get("meta", {})
    model = Predictor._build(meta)
    model.load_state_dict(state["model"])
    export_target = model.backbone if hasattr(model, "backbone") else model
    onnx_path, info = export_onnx(export_target, run, input_shape=(1, 3, 32, 32), meta=meta)
    print(f"exported {onnx_path.name} | parity diff {info['parity_max_abs_diff']:.2e}")
