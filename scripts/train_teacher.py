"""Q1 stage 1: train the ResNet-50 teacher on CIFAR-100."""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_config, parse_args
from src.data import NUM_CLASSES, make_loaders
from src.engine import (append_csv, evaluate, make_scaler, save_checkpoint,
                        set_seed, train_one_epoch)
from src.models.factory import build_model
from src.paths import CKPT_DIR, RESULTS_DIR


def main():
    args = parse_args("Train the ResNet-50 teacher").parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(
        cfg["dataset"], cfg["batch_size"], cfg["num_workers"],
        cfg["max_train"], cfg["max_val"], seed=cfg["seed"],
    )
    model, feature_layer = build_model(cfg["model"], NUM_CLASSES[cfg["dataset"]])
    model.to(device)
    print(f"Teacher: {cfg['model']} | feature tap: {feature_layer} | device: {device}")

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg["epochs"] * len(train_loader), 1))
    scaler = make_scaler(cfg["amp"])
    log_path = RESULTS_DIR / "logs" / "teacher_resnet50.csv"
    best_acc = -1.0

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train = train_one_epoch(model, train_loader, optimizer, F.cross_entropy,
                                device, scaler, scheduler, cfg["amp"])
        val = evaluate(model, val_loader, device, cfg["amp"])
        append_csv(log_path, {"epoch": epoch + 1, **train,
                              "val_loss": val["loss"], "val_acc": val["acc"]})
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            save_checkpoint(CKPT_DIR / "teacher_resnet50" / "best.pt", model,
                            dtype=torch.float16,  # keeps the file below GitHub's 100 MB limit
                            extra={"arch": cfg["model"], "feature_layer": feature_layer,
                                   "num_classes": NUM_CLASSES[cfg["dataset"]],
                                   "kind": "teacher", "epoch": epoch + 1,
                                   "dtype": "float16"})
        print(f"epoch {epoch + 1}/{cfg['epochs']} train {train['acc']:.2f}% "
              f"val {val['acc']:.2f}% ({time.time() - t0:.1f}s)", flush=True)

    print(f"Best teacher val accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
