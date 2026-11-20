"""Unified predictor supporting both PyTorch checkpoints and ONNX Runtime models."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

from ..data import CIFAR_MEAN, CIFAR_STD


class Predictor:
    """Loads a `.onnx` (+ `.json`) or a `.pt` checkpoint and predicts on images."""

    def __init__(self, model_path, meta_path=None, device="cpu"):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        if self.model_path.suffix == ".onnx":
            self.kind = "onnx"
            self.session = ort.InferenceSession(str(self.model_path),
                                                providers=["CPUExecutionProvider"])
            meta_file = meta_path or self.model_path.with_suffix(".json")
            self.meta = json.loads(Path(meta_file).read_text())
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.kind = "torch"
            self.meta = json.loads(Path(meta_path).read_text()) if meta_path else {}
            state = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.meta = state.get("meta", self.meta)
            self.model = self._build(self.meta).to(self.device)
            self.model.load_state_dict(state["model"])
            self.model.eval()
        res = self.meta.get("input_shape", [1, 3, 32, 32])[-1]
        self.transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    @staticmethod
    def _build(meta):
        if meta.get("kind") == "nas":
            from ..nas.builder import EdgeCNN
            return EdgeCNN(meta["arch"], meta["num_classes"])
        from ..models.factory import build_model
        return build_model(meta["arch"], meta["num_classes"])[0]

    def preprocess(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        return self.transform(image).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image, topk=5):
        x = self.preprocess(image)
        if self.kind == "onnx":
            logits = self.session.run(None, {self.input_name: x.numpy()})[0]
            probs = torch.softmax(torch.tensor(logits), dim=1)
        else:
            logits = self.model(x.to(self.device))
            probs = logits.softmax(dim=1).cpu()
        values, indices = probs[0].topk(min(topk, probs.shape[1]))
        return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist())]
