"""Run the whole study end to end: generate (primary + alt templates),
aggregate, and plot.

Each model runs in its own run_eval subprocess (frees Metal/MPS memory between
models) and every cell is resumable, so an interrupted run continues where it
stopped. Launch detached with `python launch.py`, or run directly here.
"""

import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent
RESULTS = REPO / "results"
RESULTS_ALT = REPO / "results_alt"
LOG = RESULTS / "run.log"
DONE = RESULTS / "DONE.txt"
PIDFILE = RESULTS / "run.pid"

MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/gemma-2-2b-it-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
]
DEFENSES = ["none", "spotlight", "sandwich", "detector"]
PY = sys.executable


def log(msg: str):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> int:
    log("RUN: " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    log(f"  -> exit {rc}")
    return rc


def gen(results_dir, template):
    # One model per subprocess: each run_eval process exits before the next
    # model loads, so all mlx/Metal memory is released between models.
    for model in MODELS:
        cmd = [PY, "-m", "src.run_eval",
               "--backend", "mlx", "--device", "m4", "--precision", "4bit",
               "--template", template,
               "--models", model,
               "--defenses", *DEFENSES]
        if results_dir is not None:
            cmd += ["--results-dir", str(results_dir)]
        run(cmd)


def main():
    PIDFILE.write_text(str(os.getpid()))
    log("===== experiment started =====")

    # 1-2. PRIMARY template
    gen(None, "primary")
    run([PY, "-m", "src.metrics", "--results", "results",
         "--out", "results/aggregated.csv"])
    run([PY, "-m", "src.plot"])
    log("primary done (generation + metrics + plot).")

    # 3-4. ALTERNATIVE template (template-dependence check)
    RESULTS_ALT.mkdir(parents=True, exist_ok=True)
    gen(RESULTS_ALT, "alt")
    run([PY, "-m", "src.metrics", "--results", "results_alt",
         "--out", "results_alt/aggregated.csv"])
    log("alt done (generation + metrics).")

    # 5. DONE
    DONE.write_text(
        "Experiment complete.\n"
        f"finished: {time.ctime()}\n"
        "primary: results/aggregated.csv + figures/asr_vs_fpr.png\n"
        "alt:     results_alt/aggregated.csv\n"
    )
    log("===== ALL DONE -> results/DONE.txt =====")


if __name__ == "__main__":
    main()
