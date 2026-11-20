"""Hugging Face dataset, accelerate and Hub helpers used by the training scripts."""
import os
from pathlib import Path

import torch
from accelerate import Accelerator


def load_hf_cifar(name="cifar100"):
    """Return the raw HF dataset object (used by data.py on cache miss)."""
    from datasets import load_dataset

    return load_dataset(f"uoft-cs/{name}")


class AccelerateRunner:
    """Small wrapper around `Accelerator` for device/AMP orchestration."""

    def __init__(self, mixed_precision="fp16"):
        mode = mixed_precision if torch.cuda.is_available() else "no"
        self.accelerator = Accelerator(mixed_precision=mode)

    def prepare(self, *objects):
        return self.accelerator.prepare(*objects)

    def backward(self, loss):
        self.accelerator.backward(loss)

    @property
    def device(self):
        return self.accelerator.device


def push_checkpoint_to_hub(local_path, repo_id=None, token=None):
    """Optional checkpoint upload. No-op unless credentials are configured."""
    repo_id = repo_id or os.environ.get("EC_HF_REPO")
    token = token or os.environ.get("HF_TOKEN")
    if not repo_id or not token:
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=True)
    path = Path(local_path)
    return api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
    )
