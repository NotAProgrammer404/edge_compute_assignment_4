"""Central, environment-overridable paths for the whole project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


DATA_DIR = _env("EC_DATA_DIR", ROOT / "data")
CKPT_DIR = _env("EC_CKPT_DIR", ROOT / "checkpoints")
RESULTS_DIR = _env("EC_RESULTS_DIR", ROOT / "results")
ONNX_DIR = _env("EC_ONNX_DIR", ROOT / "onnx")


def ensure_dirs() -> None:
    for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, ONNX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run_dir(run: str) -> Path:
    return CKPT_DIR / run
