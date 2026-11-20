"""Q1 stages 2-3: baseline or knowledge-distilled student training."""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_config, parse_args
from src.data import NUM_CLASSES, make_loaders
from src.engine import (append_csv, autocast, evaluate, make_scaler, save_checkpoint,
                        set_seed)
from src.kd.distill import DistillCriterion, FeatureAdapter
from src.models.factory import build_model
from src.paths import CKPT_DIR, RESULTS_DIR


def train_epoch(model, loader, optimizer, criterion, device, scaler, cfg,
                teacher=None, adapter=None, use_feat=False):
    model.train()
    if adapter is not None:
        adapter.train()
    correct, seen, total_loss = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        t_logits = t_feat = None
        with torch.no_grad():
            if teacher is not None:
                with autocast(cfg["amp"]):
                    t_logits, t_feat = teacher(x, return_feats=True)
                t_logits, t_feat = t_logits.float(), t_feat.float()
        with autocast(cfg["amp"]):
            if use_feat:
                s_logits, s_feat = model(x, return_feats=True)
            else:
                s_logits = model(x)
                s_feat = None
            loss, _ = criterion(s_logits.float(), y, t_logits,
                                s_feat.float() if use_feat else None, t_feat, adapter)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += int((s_logits.argmax(1) == y).sum().item())
        seen += bs
    return {"loss": total_loss / max(seen, 1), "acc": 100.0 * correct / max(seen, 1)}


def main():
    parser = parse_args("Train student models with/without distillation")
    parser.add_argument("--mode", choices=["baseline", "kd"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    mode = args.mode
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = NUM_CLASSES[cfg["dataset"]]

    train_loader, val_loader = make_loaders(
        cfg["dataset"], cfg["batch_size"], cfg["num_workers"],
        cfg["max_train"], cfg["max_val"], seed=cfg["seed"],
    )

    teacher = None
    if mode == "kd":
        state = torch.load(CKPT_DIR / "teacher_resnet50" / "best.pt",
                           map_location="cpu", weights_only=False)
        teacher, _ = build_model(state["meta"]["arch"], n_classes)
        teacher.load_state_dict(state["model"])
        teacher.to(device).eval()
        kd_cfg = cfg["kd"]
        criterion = DistillCriterion(kd_cfg)
        use_feat = bool(kd_cfg.get("use_feat", False))
    else:
        criterion = lambda logits, y, *a, **k: (F.cross_entropy(logits, y), {})
        use_feat = False

    for spec in cfg["students"]:
        name, arch = spec["name"], spec["arch"]
        run = f"{mode}_{name}"
        model, feature_layer = build_model(arch, n_classes)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                    momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
        scaler = make_scaler(cfg["amp"])
        adapter = None

        if use_feat:
            model.eval()
            with torch.no_grad():
                # batch > 1: some backbones have BN layers that need >1 spatial value
                dummy = torch.zeros(4, 3, 32, 32, device=device)
                _, s_feat = model(dummy, return_feats=True)
                _, t_feat = teacher(dummy, return_feats=True)
            model.train()
            adapter = FeatureAdapter(s_feat.shape[1], t_feat.shape[1]).to(device)
            optimizer.add_param_group({"params": adapter.parameters(), "lr": cfg["lr"]})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg["epochs"] * len(train_loader), 1))

        print(f"{run}: {arch} | feature tap: {feature_layer}", flush=True)
        best_acc = -1.0
        for epoch in range(cfg["epochs"]):
            t0 = time.time()
            train = train_epoch(model, train_loader, optimizer, criterion, device,
                                scaler, cfg, teacher, adapter, use_feat)
            val = evaluate(model, val_loader, device, cfg["amp"])
            append_csv(RESULTS_DIR / "logs" / f"{run}.csv",
                       {"epoch": epoch + 1, **train,
                        "val_loss": val["loss"], "val_acc": val["acc"]})
            if val["acc"] > best_acc:
                best_acc = val["acc"]
                save_checkpoint(CKPT_DIR / run / "best.pt", model,
                                extra={"arch": arch, "feature_layer": feature_layer,
                                       "num_classes": n_classes, "kind": "student",
                                       "mode": mode, "epoch": epoch + 1})
            print(f"  {run} epoch {epoch + 1}/{cfg['epochs']} train {train['acc']:.2f}% "
                  f"val {val['acc']:.2f}% ({time.time() - t0:.1f}s)", flush=True)
        print(f"  {run} best val accuracy: {best_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
