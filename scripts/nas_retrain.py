"""Q2 stage 2: retrain the discovered architecture from scratch."""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_config, parse_args
from src.data import NUM_CLASSES, make_loaders
from src.engine import append_csv, evaluate, make_scaler, save_checkpoint, set_seed, train_one_epoch
from src.nas.builder import EdgeCNN
from src.nas.space import describe
from src.paths import CKPT_DIR, RESULTS_DIR


def main():
    args = parse_args("Retrain the best NAS architecture").parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = json.loads((RESULTS_DIR / "nas" / "best_arch.json").read_text())
    n_classes = NUM_CLASSES[cfg["dataset"]]
    print("Retraining:", describe(arch), flush=True)

    train_loader, val_loader = make_loaders(
        cfg["dataset"], cfg["batch_size"], cfg["num_workers"],
        cfg["max_train"], cfg["max_val"], seed=cfg["seed"],
    )
    model = EdgeCNN(arch, n_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg["retrain_epochs"] * len(train_loader), 1))
    scaler = make_scaler(cfg["amp"])
    best = -1.0
    for epoch in range(cfg["retrain_epochs"]):
        t0 = time.time()
        train = train_one_epoch(model, train_loader, optimizer, F.cross_entropy,
                                device, scaler, scheduler, cfg["amp"])
        val = evaluate(model, val_loader, device, cfg["amp"])
        append_csv(RESULTS_DIR / "logs" / "nas_best.csv",
                   {"epoch": epoch + 1, **train, "val_loss": val["loss"], "val_acc": val["acc"]})
        if val["acc"] > best:
            best = val["acc"]
            save_checkpoint(CKPT_DIR / "nas_best" / "best.pt", model,
                            extra={"arch": arch, "num_classes": n_classes,
                                   "kind": "nas", "epoch": epoch + 1})
        print(f"epoch {epoch + 1}/{cfg['retrain_epochs']} train {train['acc']:.2f}% "
              f"val {val['acc']:.2f}% ({time.time() - t0:.1f}s)", flush=True)
    print(f"Best NAS val accuracy: {best:.2f}%")


if __name__ == "__main__":
    main()
