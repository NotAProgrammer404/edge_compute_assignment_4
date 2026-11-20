"""JSON-serialisable, searchable CNN architecture space.

An architecture is a plain dict with variable depth, width, kernels, block type,
activation and input resolution — all six requirements from the assignment.
"""
import random

SPACE = {
    "depths": (2, 4),  # inclusive min/max for suggest_int
    "widths": (16, 24, 32, 48),
    "kernels": (3, 5),
    "blocks_per_stage": (1, 2),
    "blocks": ("conv", "dwsep"),
    "activations": ("relu", "relu6", "silu", "gelu"),
    "resolutions": (32, 40),
    "head_widths": (64, 128, 256),
}


def _stage_strides(n):
    return [1] + [2] * (n - 1)


def random_arch(rng=None, space=SPACE):
    """Uniform sample — handy for smoke tests and profiling sweeps."""
    rng = rng or random
    depth = rng.randint(*space["depths"])
    return {
        "resolution": rng.choice(space["resolutions"]),
        "stem": rng.choice(space["widths"]),
        "head": rng.choice(space["head_widths"]),
        "stages": [
            {
                "blocks": rng.randint(*space["blocks_per_stage"]),
                "channels": rng.choice(space["widths"]),
                "kernel": rng.choice(space["kernels"]),
                "block": rng.choice(space["blocks"]),
                "act": rng.choice(space["activations"]),
                "stride": stride,
            }
            for stride in _stage_strides(depth)
        ],
    }


def optuna_arch(trial, space=SPACE):
    """Sample one architecture from an Optuna TPE trial."""
    depth = trial.suggest_int("depth", *space["depths"])
    arch = {
        "resolution": trial.suggest_categorical("resolution", space["resolutions"]),
        "stem": trial.suggest_categorical("stem", space["widths"]),
        "head": trial.suggest_categorical("head", space["head_widths"]),
        "stages": [],
    }
    for i, stride in enumerate(_stage_strides(depth)):
        arch["stages"].append({
            "blocks": trial.suggest_int(f"s{i}_blocks", *space["blocks_per_stage"]),
            "channels": trial.suggest_categorical(f"s{i}_channels", space["widths"]),
            "kernel": trial.suggest_categorical(f"s{i}_kernel", space["kernels"]),
            "block": trial.suggest_categorical(f"s{i}_block", space["blocks"]),
            "act": trial.suggest_categorical(f"s{i}_act", space["activations"]),
            "stride": stride,
        })
    return arch


def describe(arch):
    stages = " | ".join(
        f"{s['blocks']}x{s['block']}{s['kernel']}-{s['channels']}-{s['act']}"
        for s in arch["stages"]
    )
    return f"res{arch['resolution']} stem{arch['stem']} [{stages}] head{arch['head']}"
