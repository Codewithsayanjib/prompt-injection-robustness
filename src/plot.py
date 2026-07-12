"""ASR-vs-FPR scatter, one point per (model, defense).

    python -m src.plot --csv results/aggregated.csv --out figures/asr_vs_fpr.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/aggregated.csv")
    ap.add_argument("--out", default="figures/asr_vs_fpr.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty CSV; run metrics.py first")
        return

    defenses = sorted({r["defense"] for r in rows})
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    dmark = {d: markers[i % len(markers)] for i, d in enumerate(defenses)}

    fig, ax = plt.subplots(figsize=(6, 5))
    for r in rows:
        x = float(r["FPR"])
        y = float(r["ASR"])
        ax.scatter(x, y, marker=dmark[r["defense"]], s=70, alpha=0.8)
        ax.annotate(f"{r['model'].split('/')[-1]}\n{r['defense']}",
                    (x, y), fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("FPR / over-defense (benign wrongly blocked)  -> lower better")
    ax.set_ylabel("ASR (canary leaked)  -> lower better")
    ax.set_title("Robustness vs usability: ASR vs FPR per (model, defense)")
    ax.grid(True, alpha=0.3)
    ax.annotate("best\n(deploy here)", (0.02, 0.02), fontsize=8, color="green")

    handles = [plt.Line2D([], [], marker=dmark[d], linestyle="", label=d)
               for d in defenses]
    ax.legend(handles=handles, title="defense", fontsize=7)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
