"""Load public attack/benign datasets and freeze fixed samples to JSONL.

500 samples per category, sampled once with a fixed seed so every run scores the
same inputs. Freeze with:

    python -m src.attacks --freeze --n 500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def _hf(path: str, split: str, **kw):
    from datasets import load_dataset

    return load_dataset(path, split=split, **kw)


def _col(colnames, candidates: list[str]) -> str:
    # Case-insensitive column pick; datasets vary ('Text'/'Label' vs lowercase).
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return colnames[0]


def load_deepset_direct() -> list[dict]:
    ds = _hf("deepset/prompt-injections", "train")
    tcol = _col(ds.column_names, ["text"])
    lcol = _col(ds.column_names, ["label"])
    return [{"text": r[tcol]} for r in ds if int(r.get(lcol, 0)) == 1 and r[tcol]]


def load_xtram_direct() -> list[dict]:
    ds = _hf("xTRam1/safe-guard-prompt-injection", "train")
    tcol = _col(ds.column_names, ["text"])
    lcol = _col(ds.column_names, ["label"])
    return [{"text": r[tcol]} for r in ds if int(r.get(lcol, 0)) == 1 and r[tcol]]


def load_advbench() -> list[dict]:
    # Public CSV from the llm-attacks repo (the HF mirror is gated).
    import pandas as pd

    url = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
           "main/data/advbench/harmful_behaviors.csv")
    try:
        df = pd.read_csv(url)
        col = "goal" if "goal" in df.columns else df.columns[0]
        return [{"text": str(x)} for x in df[col].tolist() if str(x).strip()]
    except Exception:
        ds = _hf("walledai/AdvBench", "train")
        col = _col(ds.column_names, ["prompt", "goal", "text"])
        return [{"text": r[col]} for r in ds if r[col]]


def load_llmail_indirect(cap: int = 5000, seed: int = 0) -> list[dict]:
    # 371k emails in the Phase1 split; stream + cap to bound memory.
    from datasets import load_dataset

    ds = load_dataset("microsoft/llmail-inject-challenge", split="Phase1",
                      streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out: list[dict] = []
    for r in ds:
        body = r.get("body") or r.get("subject")
        if body:
            out.append({"text": body})
        if len(out) >= cap:
            break
    return out


def load_benign() -> list[dict]:
    # SQuAD questions + gold answers -> benign control + objective utility.
    try:
        ds = _hf("rajpurkar/squad", "validation")
        qcol = _col(ds.column_names, ["question"])
        out = []
        for r in ds:
            if not r[qcol]:
                continue
            ans = r.get("answers", {}).get("text", []) if isinstance(r.get("answers"), dict) else []
            out.append({"text": r[qcol], "answer": ans[0] if ans else ""})
        return out
    except Exception:
        ds = _hf("deepset/prompt-injections", "train")
        tcol = _col(ds.column_names, ["text"])
        return [{"text": r[tcol], "answer": ""} for r in ds
                if int(r.get("label", 0)) == 0 and r[tcol]]


CATEGORIES = {
    "direct": [("xtram", load_xtram_direct), ("deepset", load_deepset_direct)],
    "jailbreak": [("advbench", load_advbench)],
    "indirect": [("llmail", load_llmail_indirect)],
    "benign": [("squad", load_benign)],
}


def freeze(n: int, seed: int) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for category, sources in CATEGORIES.items():
        pool: list[tuple[str, dict]] = []
        for src_name, loader in sources:
            try:
                items = loader()
                print(f"  loaded {len(items):>6} from {category}/{src_name}")
                pool.extend((src_name, it) for it in items)
            except Exception as e:
                print(f"  !! failed {category}/{src_name}: {e}")
        if not pool:
            print(f"  !! no data for category {category}, skipping")
            continue
        rng.shuffle(pool)
        take = pool[:n]
        out = SAMPLES_DIR / f"{category}_{n}.jsonl"
        with out.open("w") as f:
            for i, (src, item) in enumerate(take):
                row = {
                    "id": f"{category}-{i:04d}",
                    "category": category,
                    "source": src,
                    "text": item["text"],
                    "answer": item.get("answer", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  -> wrote {len(take)} -> {out.name}"
              + ("" if len(take) == n else f"  (WARNING: fewer than n={n})"))


def load_frozen(category: str, n: int) -> list[dict]:
    path = SAMPLES_DIR / f"{category}_{n}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.attacks --freeze --n {n}"
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze fixed dataset samples to JSONL")
    ap.add_argument("--freeze", action="store_true", help="download + sample + write JSONL")
    ap.add_argument("--n", type=int, default=500, help="samples per category")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.freeze:
        print(f"Freezing {args.n} samples/category (seed={args.seed}) -> {SAMPLES_DIR}")
        freeze(args.n, args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
