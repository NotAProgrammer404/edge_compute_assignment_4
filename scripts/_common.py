"""Shared CLI/config helpers for the pipeline scripts."""
import argparse
from pathlib import Path

import yaml


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Dot-separated config overrides")
    return parser


def _coerce(value):
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return cfg
