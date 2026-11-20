"""Q2 stage 1: run the Optuna hardware-aware architecture search."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_config, parse_args
from src.nas.search import run_search
from src.nas.space import describe

if __name__ == "__main__":
    args = parse_args("Run hardware-aware NAS").parse_args()
    cfg = load_config(args.config, args.set)
    arch, metrics = run_search(cfg)
    print("Best feasible architecture:", describe(arch))
    print("Metrics:", metrics)
