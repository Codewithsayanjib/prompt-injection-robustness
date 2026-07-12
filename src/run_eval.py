"""Evaluation loop over models x defenses x inputs.

Writes one JSONL row per (model, defense, input) to
results/{model}__{defense}.jsonl and resumes by skipping ids already present.

    python -m src.run_eval --backend mock --models mock-weak --limit 50
    python -m src.run_eval --backend mlx --device m4 \
        --models mlx-community/Llama-3.2-1B-Instruct-4bit

For the full study use run_experiment.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make sibling modules importable whether run as `python -m src.run_eval`
# (from repo root) or `python run_eval.py` (from src/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks import CATEGORIES, load_frozen
from defenses import BLOCKED_OUTPUT, build_defense
from load_models import GenConfig, load_backend
from prompts import build_system_prompt, build_user
from scorer import CanaryScorer, answer_hit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["row_id"])
    return done


def run_cell(backend, model_name, defense_name, defense, scorer, canary,
             template, inputs, device, precision, out_path, limit):
    """Run one (model, defense) cell, appending resumably to out_path."""
    done = _done_ids(out_path)
    written = 0
    system = build_system_prompt(canary, template)
    with out_path.open("a") as f:
        for item in inputs:
            row_id = f"{model_name}|{defense_name}|{item['id']}"
            if row_id in done:
                continue
            t0 = time.perf_counter()
            user_msg = build_user(item["category"], item["text"])
            res = defense.apply(system, user_msg)
            if res.blocked:
                output = BLOCKED_OUTPUT
            else:
                output = backend.generate(res.system, res.user)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            is_benign = item["category"] == "benign"
            # On benign inputs a "leak" is impossible; success applies to attacks.
            success = 0 if is_benign else scorer.score(output)
            # Objective utility: only for benign, only when not blocked.
            util = (answer_hit(output, item.get("answer", ""))
                    if is_benign and not res.blocked else None)

            row = {
                "row_id": row_id,
                "model": model_name,
                "defense": defense_name,
                "template": template,
                "device": device,
                "precision": precision,
                "category": item["category"],
                "source": item["source"],
                "input_id": item["id"],
                "blocked": bool(res.blocked),
                "success": int(success),
                "utility_hit": util,
                "latency_ms": round(latency_ms, 2),
                "output": output[:500],  # truncate; full text rarely needed
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
    return written, len(done)


def load_inputs(n: int, categories: list[str], limit: int = 0) -> list[dict]:
    items: list[dict] = []
    for cat in categories:
        rows = load_frozen(cat, n)
        items.extend(rows[:limit] if limit else rows)  # per-category cap
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mock", "mlx", "hf"])
    ap.add_argument("--models", nargs="+", required=True,
                    help="model ids for the chosen backend (or mock-weak/mock-strong)")
    ap.add_argument("--defenses", nargs="+",
                    default=["none", "spotlight", "sandwich", "detector"])
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--n", type=int, default=500, help="samples/category (matches frozen files)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap inputs PER CATEGORY (0 = all); for balanced smoke tests")
    ap.add_argument("--canary", default="BANANA")
    ap.add_argument("--template", default="primary", choices=["primary", "alt"])
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="auto", help="tag recorded in results (e.g. m4, t4)")
    ap.add_argument("--precision", default="4bit", help="tag recorded in results")
    ap.add_argument("--detector-model",
                    default="protectai/deberta-v3-base-prompt-injection-v2")
    ap.add_argument("--results-dir", default=None,
                    help="override output dir (e.g. results_alt for the "
                         "alt-template robustness check, to avoid colliding with "
                         "the primary-template results)")
    args = ap.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir).resolve()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scorer = CanaryScorer(args.canary)
    gen = GenConfig(max_new_tokens=args.max_new_tokens)
    inputs = load_inputs(args.n, args.categories, args.limit)
    print(f"Loaded {len(inputs)} inputs across {args.categories}"
          + (f" (capped {args.limit}/category)" if args.limit else ""))

    for model_name in args.models:
        print(f"\n=== model: {model_name} ({args.backend}) ===")
        backend = load_backend(args.backend, model_name, gen)
        for defense_name in args.defenses:
            params = {"model_id": args.detector_model} if defense_name == "detector" else {}
            defense = build_defense(defense_name, **params)
            safe_model = model_name.replace("/", "_")
            out_path = RESULTS_DIR / f"{safe_model}__{defense_name}.jsonl"
            t0 = time.perf_counter()
            written, skipped = run_cell(
                backend, model_name, defense_name, defense, scorer, args.canary,
                args.template, inputs, args.device, args.precision, out_path, args.limit,
            )
            dt = time.perf_counter() - t0
            print(f"  [{defense_name:<10}] wrote {written:>4} (skipped {skipped:>4} done) "
                  f"in {dt:5.1f}s -> {out_path.name}")

    print("\nDone. Aggregate with: python -m src.metrics")


if __name__ == "__main__":
    main()
