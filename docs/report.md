# Technical Report — Edge KD + NAS (Assignment 4)

## 1. Overview

This submission implements two independent pipelines for edge deployment of vision models:

1. **Knowledge distillation (Q1)** — a ResNet-50 teacher on CIFAR-100 transfers both logit and
   intermediate-feature information to three lightweight students: MobileNetV2, SqueezeNet and
   EfficientNet-Lite0. Baseline (hard-label only) students are trained as a control.
2. **Hardware-aware NAS (Q2)** — Optuna TPE searches a CNN space on CIFAR-10. Every candidate is
   measured for real CPU latency, serialized size and peak allocation, and architectures violating
   deployment constraints are rejected. The winner is retrained from scratch.

Both pipelines export ONNX models and provide a unified inference/benchmark interface.

## 2. Q1 — Knowledge distillation

### 2.1 Teacher and students

The teacher is a `resnet50` backbone (timm) with a 100-way classifier. Feature hints are tapped
from `layer2`. Students are `mobilenetv2_100` (tap `blocks.3`), `squeezenet1_1` (tap `features`,
torchvision build) and `efficientnet_lite0` (tap `blocks.2`). Taps are implemented as forward
hooks in `src/models/factory.py`, so the model definitions themselves are untouched.

### 2.2 Distillation objective

For student logits `z_s`, teacher logits `z_t`, labels `y`, temperature `T`:

```
L_hard = CE(z_s, y)
L_soft = T^2 * KL(softmax(z_s / T) || softmax(z_t / T))
L_feat = || norm(proj(f_s)) - norm(f_t) ||^2
L      = (1 - alpha) * L_hard + alpha * L_soft + beta * L_feat
```

`proj` is a 1x1 conv + batch-norm adapter with adaptive pooling to align spatial sizes. Feature
maps are L2-normalised per sample, which bounds the hint term in `[0, 4]` and makes it independent
of activation scale. All loss terms are evaluated in FP32 even when the forward pass uses AMP.

Baselines use exactly the same loop with `teacher=None`, reducing the objective to `L_hard`.

### 2.3 Inference and deployment

`src/deploy/runtime.py::Predictor` accepts either a `.pt` checkpoint plus its JSON metadata or an
`.onnx` file plus its sidecar JSON. ONNX export uses opset 13, dynamic batch axis, and performs a
max-absolute-difference parity check against PyTorch before writing metadata.

## 3. Q2 — Hardware-aware NAS

### 3.1 Search space

Each architecture is a JSON-serialisable dict varying: number of stages (2-4), channels per stage,
convolution kernel (3 or 5), block type (regular vs depthwise-separable), activation (ReLU, ReLU6,
SiLU, GELU), blocks per stage, stem/head widths, and input resolution (32 or 40). `src/nas/builder.py`
constructs a PyTorch module from any sampled dict.

### 3.2 Search strategy and constraints

Optuna's TPE sampler proposes architectures. Each candidate is trained for a small number of
epochs on a CIFAR-10 subset and evaluated for validation accuracy. A **CPU copy** of the trained
weights is then profiled for mean latency (30 timed forwards after 5 warm-ups), serialized state
dict size, and peak allocator usage. Candidates exceeding any configured limit are penalised and
excluded from final selection. Search history is written to `results/nas/history.{csv,json}`.

### 3.3 Retraining

`results/nas/best_arch.json` is read by `scripts/nas_retrain.py`, which rebuilds the architecture
from scratch and trains it with the same schedule before exporting to ONNX.

## 4. Hugging Face integration

- `datasets` sources CIFAR-10/100 (`uoft-cs/cifar10`, `uoft-cs/cifar100`) and caches them locally.
- `accelerate` handles device placement and AMP configuration in `src/hf/pipelines.py`.
- `timm` supplies the ResNet-50, MobileNetV2 and EfficientNet-Lite0 backbones.
- `huggingface_hub` provides optional checkpoint upload when `HF_TOKEN` and `EC_HF_REPO` are set.

## 5. Reproducing

```bash
pip install -r requirements.txt
python scripts/train_teacher.py --config configs/teacher.yaml
python scripts/train_student.py --config configs/student.yaml --mode baseline
python scripts/train_student.py --config configs/student.yaml --mode kd
python scripts/nas_search.py --config configs/nas.yaml
python scripts/nas_retrain.py --config configs/nas.yaml
python scripts/export_onnx.py
python scripts/benchmark.py
python scripts/summarize.py
```

The shipped configs use reduced subsets so the full pipeline completes quickly on a consumer GPU.
Increase `max_train`, `max_val` and epoch counts for a full-scale run.

## 6. Results

All runs used the full CIFAR-100/CIFAR-10 training splits (10 epochs for Q1, 6 NAS trials with
1 epoch per candidate, 3 epochs for the NAS retrain) on an RTX 2060 6 GB.

### Q1 — teacher and students (best validation accuracy)

| Run | Best val acc (%) |
|---|---:|
| teacher_resnet50 | 49.74 |
| baseline_mbv2 | 18.97 |
| kd_mbv2 | **32.69** |
| baseline_squeezenet | 32.66 |
| kd_squeezenet | **37.50** |
| baseline_efflite0 | 20.70 |
| kd_efflite0 | **30.94** |

Distillation improves all three students, by +13.72 / +4.84 / +10.24 percentage points
respectively. Full curves are in `results/logs/`.

### Q2 — NAS

Best feasible architecture: `res32 stem16 [1xdwsep5-16-relu | 1xconv3-48-relu6] head256`.
Search-history metrics: validation accuracy 21.3%, CPU latency 0.557 ms, serialized size
100.8 KB, peak allocator usage 2.3 KB. After retraining from scratch, validation accuracy
reached **27.0%**. The full history is in `results/nas/history.{csv,json}`.

### ONNX CPU benchmark

| Model | Latency (ms) | Size (KB) |
|---|---:|---:|
| nas_best | 0.149 | 93.7 |
| kd_squeezenet | 0.992 | 3,038.2 |
| baseline_squeezenet | 1.320 | 3,038.2 |
| kd_mbv2 | 1.853 | 9,158.2 |
| baseline_mbv2 | 2.160 | 9,158.2 |
| kd_efflite0 | 2.230 | 13,621.6 |
| baseline_efflite0 | 2.246 | 13,621.6 |
| teacher_resnet50 | 4.856 | 92,548.7 |

All exports passed the PyTorch/ONNX Runtime parity check (max absolute difference below
`4e-6`). The NAS model is ~33x faster and ~980x smaller than the teacher on CPU.

## 7. Attribution

All source code in this repository was written for this submission. The following third-party
work is used as libraries/pretrained architecture definitions and is acknowledged:

- He et al., *Deep Residual Learning for Image Recognition* (ResNet-50).
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.
- Iandola et al., *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters*.
- Tan et al., *EfficientNet: Rethinking Model Scaling for CNNs* (EfficientNet-Lite variant).
- Hinton et al., *Distilling the Knowledge in a Neural Network*.
- Romero et al., *FitNets: Hints for Thin Deep Nets*.
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*.
- Hugging Face `datasets`, `accelerate`, `huggingface_hub`; `timm`; ONNX Runtime.
