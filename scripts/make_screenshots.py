"""Generate result screenshots (PNG plots) from the produced CSV files."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import RESULTS_DIR

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_teacher():
    rows = read_csv(RESULTS_DIR / "logs" / "teacher_resnet50.csv")
    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(r["acc"]) for r in rows], "o-", label="Train accuracy")
    ax.plot(epochs, [float(r["val_acc"]) for r in rows], "s-", label="Validation accuracy")
    ax.set(title="ResNet-50 teacher on CIFAR-100", xlabel="Epoch", ylabel="Accuracy (%)")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q1_teacher_curve.png", dpi=150)
    plt.close(fig)


def plot_students():
    pairs = [("mbv2", "MobileNetV2"), ("squeezenet", "SqueezeNet"),
             ("efflite0", "EfficientNet-Lite0")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (key, title) in zip(axes, pairs):
        base = read_csv(RESULTS_DIR / "logs" / f"baseline_{key}.csv")
        kd = read_csv(RESULTS_DIR / "logs" / f"kd_{key}.csv")
        epochs = [int(r["epoch"]) for r in kd]
        ax.plot(epochs, [float(r["val_acc"]) for r in base], "o-", label="Baseline")
        ax.plot(epochs, [float(r["val_acc"]) for r in kd], "s-", label="Distilled")
        ax.set(title=title, xlabel="Epoch", ylabel="Val accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Knowledge distillation vs baseline (CIFAR-100)")
    fig.tight_layout()
    fig.savefig(OUT / "q1_baseline_vs_kd.png", dpi=150)
    plt.close(fig)


def plot_nas():
    rows = read_csv(RESULTS_DIR / "nas" / "history.csv")
    xs = [float(r["latency_ms"]) for r in rows]
    ys = [float(r["val_acc"]) for r in rows]
    sizes = [max(float(r["size_kb"]), 20.0) for r in rows]
    colors = ["tab:green" if r["feasible"].lower() == "true" else "tab:red" for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.7, edgecolors="black")
    for i, r in enumerate(rows):
        ax.annotate(f"#{r['trial']}", (xs[i], ys[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set(title="NAS search: accuracy vs measured CPU latency",
           xlabel="CPU latency (ms)", ylabel="Validation accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green",
                   label="Constraint-feasible"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",
                   label="Rejected"),
    ])
    fig.tight_layout()
    fig.savefig(OUT / "q2_nas_search.png", dpi=150)
    plt.close(fig)


def plot_benchmark():
    rows = read_csv(RESULTS_DIR / "benchmark.csv")
    rows = sorted(rows, key=lambda r: float(r["latency_ms"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([r["model"] for r in rows], [float(r["latency_ms"]) for r in rows],
            color="tab:blue", alpha=0.85)
    ax.set(title="ONNX Runtime CPU inference latency", xlabel="Latency (ms, lower is better)")
    ax.grid(axis="x", alpha=0.3)
    for i, r in enumerate(rows):
        ax.text(float(r["latency_ms"]) + 0.03, i, f"{float(r['latency_ms']):.3f}",
                va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "onnx_benchmark.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_teacher()
    plot_students()
    plot_nas()
    plot_benchmark()
    print("Screenshots written to", OUT)
