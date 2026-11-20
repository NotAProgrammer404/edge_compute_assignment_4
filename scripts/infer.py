"""Unified inference on any exported model (PyTorch or ONNX Runtime)."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deploy.runtime import Predictor

parser = argparse.ArgumentParser(description="Run unified inference")
parser.add_argument("--model", required=True, help="Path to .onnx or .pt")
parser.add_argument("--meta", default=None, help="JSON metadata (needed with .pt if absent)")
parser.add_argument("--image", default=None, help="Image path; omit for a random demo tensor")
parser.add_argument("--topk", type=int, default=5)
args = parser.parse_args()

predictor = Predictor(args.model, args.meta)
image = args.image if args.image else np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
print(f"backend={predictor.kind} model={args.model}")
for class_id, confidence in predictor.predict(image, topk=args.topk):
    print(f"  class {class_id:3d}  p={confidence:.4f}")
