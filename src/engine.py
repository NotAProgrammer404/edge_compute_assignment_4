"""Generic training/evaluation utilities shared by Q1 and Q2."""
import contextlib
import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def autocast(enabled=True):
    if enabled and torch.cuda.is_available():
        with torch.autocast("cuda", dtype=torch.float16):
            yield
    else:
        yield


def make_scaler(enabled=True):
    return torch.amp.GradScaler("cuda", enabled=enabled and torch.cuda.is_available())


def accuracy(logits, y):
    return int((logits.argmax(dim=1) == y).sum().item())


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)


def _logits(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None,
                    scheduler=None, amp=True, max_batches=None):
    model.train()
    losses, correct, seen = AverageMeter(), 0, 0
    for k, (x, y) in enumerate(loader):
        if max_batches is not None and k >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(amp):
            logits = _logits(model(x))
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = y.size(0)
        losses.update(loss.item(), bs)
        correct += accuracy(logits.detach(), y)
        seen += bs
    return {"loss": losses.avg, "acc": 100.0 * correct / max(seen, 1)}


@torch.no_grad()
def evaluate(model, loader, device, amp=True, max_batches=None):
    model.eval()
    losses, correct, seen = AverageMeter(), 0, 0
    for k, (x, y) in enumerate(loader):
        if max_batches is not None and k >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(amp):
            logits = _logits(model(x))
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
        bs = y.size(0)
        losses.update(loss.item(), bs)
        correct += accuracy(logits, y)
        seen += bs
    return {"loss": losses.avg, "acc": 100.0 * correct / max(seen, 1)}


def save_checkpoint(path, model, optimizer=None, scheduler=None, extra=None, dtype=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    if dtype is not None:
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    state = {"model": state_dict}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state["meta"] = extra
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("meta", {})


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header:
            writer.writeheader()
        writer.writerow(row)
