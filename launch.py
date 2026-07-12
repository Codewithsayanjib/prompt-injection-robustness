"""Run the full pipeline detached and caffeinated (survives the terminal
closing; keeps the Mac awake). Resumable.

    python launch.py                 # start, print PID, exit
    tail -f results/run.log          # watch
    cat results/DONE.txt             # written when complete
    kill $(cat results/run.pid)      # stop

caffeinate can't beat a lid close, so leave the lid open.
"""

import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent
LOG = REPO / "results" / "run.log"
PIDFILE = REPO / "results" / "run.pid"
DONE = REPO / "results" / "DONE.txt"

CMD = ["caffeinate", "-ims", sys.executable, "run_experiment.py"]


def already_running():
    if PIDFILE.exists():
        try:
            pid = int(PIDFILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            return None
    return None


def main():
    existing = already_running()
    if existing:
        print(f"already running (pid {existing}); not starting a second copy.")
        print(f"  watch: tail -f {LOG}")
        return
    if DONE.exists():
        DONE.unlink()  # clear any stale completion marker from a prior run
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as logf:
        proc = subprocess.Popen(
            CMD, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PIDFILE.write_text(str(proc.pid))
    print(f"pipeline started: pid {proc.pid} (detached + caffeinated)")
    print(f"  watch:    tail -f {LOG}")
    print(f"  done when: results/DONE.txt exists")
    print(f"  stop:     kill {proc.pid}")
    print("  REMINDER: keep the laptop lid OPEN.")


if __name__ == "__main__":
    main()
