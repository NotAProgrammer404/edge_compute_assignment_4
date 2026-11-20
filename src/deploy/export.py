"""ONNX export with a numerical parity check against PyTorch."""
import json

import numpy as np
import onnxruntime as ort
import torch

from ..paths import ONNX_DIR


def export_onnx(model, name, input_shape=(1, 3, 32, 32), opset=13, meta=None):
    model = model.eval().cpu()
    dummy = torch.randn(*input_shape)
    path = ONNX_DIR / f"{name}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, (dummy,), str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    parity = float(np.abs(torch_out - onnx_out).max())

    payload = dict(meta or {})
    payload.update({
        "name": name,
        "onnx_file": path.name,
        "input_shape": list(input_shape),
        "opset": opset,
        "parity_max_abs_diff": parity,
    })
    (ONNX_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    return path, payload
