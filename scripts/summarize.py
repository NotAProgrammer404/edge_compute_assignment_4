"""Aggregate per-epoch logs into a Q1 summary table."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import RESULTS_DIR

rows = []
for log in sorted((RESULTS_DIR / "logs").glob("*.csv")):
    with log.open() as f:
        entries = list(csv.DictReader(f))
    if not entries:
        continue
    best = max(entries, key=lambda r: float(r.get("val_acc", 0)))
    rows.append({
        "run": log.stem,
        "epochs": len(entries),
        "best_val_acc": best.get("val_acc"),
        "final_train_acc": entries[-1].get("acc"),
    })

out_csv = RESULTS_DIR / "q1_summary.csv"
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run", "epochs", "best_val_acc", "final_train_acc"])
    writer.writeheader()
    writer.writerows(rows)

lines = ["# Q1 summary", "", "| Run | Epochs | Best val acc (%) | Final train acc (%) |",
         "|---|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['run']} | {r['epochs']} | {r['best_val_acc']} | {r['final_train_acc']} |")
(RESULTS_DIR / "q1_summary.md").write_text("\n".join(lines) + "\n")
print(f"Wrote {out_csv} and q1_summary.md")
