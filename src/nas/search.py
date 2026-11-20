"""Optuna TPE hardware-aware architecture search on CIFAR-10."""
import json

import optuna
import torch

from ..data import NUM_CLASSES, make_loaders
from ..engine import append_csv, evaluate, set_seed, train_one_epoch
from ..paths import RESULTS_DIR
from .builder import EdgeCNN
from .hw_profile import measure_latency, measure_peak_kb, measure_size_kb
from .space import describe, optuna_arch


def _profile_on_cpu(arch, state_dict, num_classes):
    """Profile a CPU copy so GPU training and CPU benchmarking never interfere."""
    model = EdgeCNN(arch, num_classes)
    model.load_state_dict(state_dict)
    return {
        "latency_ms": measure_latency(model, device="cpu"),
        "size_kb": measure_size_kb(model),
        "peak_kb": measure_peak_kb(model),
    }


def run_search(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = NUM_CLASSES[cfg["dataset"]]
    train_loader, val_loader = make_loaders(
        cfg["dataset"], cfg["batch_size"], cfg["num_workers"],
        cfg["max_train"], cfg["max_val"], seed=cfg["seed"],
    )
    constraints = cfg["constraints"]
    history_path = RESULTS_DIR / "nas" / "history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        arch = optuna_arch(trial)
        model = EdgeCNN(arch, n_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                    momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg["epochs_per_trial"], 1))

        for _ in range(cfg["epochs_per_trial"]):
            train_one_epoch(model, train_loader, optimizer, torch.nn.functional.cross_entropy,
                            device, scheduler=scheduler, amp=cfg["amp"])
        val_acc = evaluate(model, val_loader, device, amp=cfg["amp"])["acc"]

        hw = _profile_on_cpu(arch, {k: v.cpu() for k, v in model.state_dict().items()}, n_classes)
        feasible = (hw["latency_ms"] <= constraints["max_latency_ms"]
                    and hw["size_kb"] <= constraints["max_size_kb"]
                    and hw["peak_kb"] <= constraints["max_peak_kb"])

        row = {"trial": trial.number, "val_acc": val_acc, "feasible": feasible,
               "arch": describe(arch), **hw}
        trial.set_user_attr("arch", arch)
        trial.set_user_attr("metrics", row)
        append_csv(history_path, row)

        penalty = 0.0
        for key, limit in (("latency_ms", "max_latency_ms"),
                           ("size_kb", "max_size_kb"),
                           ("peak_kb", "max_peak_kb")):
            penalty += 0.05 * max(0.0, hw[key] - constraints[limit]) / constraints[limit]
        return val_acc - 100.0 * penalty

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
    )
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=True)

    trials = [t for t in study.trials if t.user_attrs.get("arch")]
    feasible = [t for t in trials if t.user_attrs["metrics"]["feasible"]]
    best = max(feasible or trials, key=lambda t: t.user_attrs["metrics"]["val_acc"])

    best_arch = best.user_attrs["arch"]
    (RESULTS_DIR / "nas" / "best_arch.json").write_text(json.dumps(best_arch, indent=2))
    history = [t.user_attrs["metrics"] for t in trials]
    (RESULTS_DIR / "nas" / "history.json").write_text(json.dumps(history, indent=2))
    return best_arch, best.user_attrs["metrics"]
