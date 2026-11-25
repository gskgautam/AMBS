"""A100 <-> H200 handoff for one (experiment, backbone) GPU stage.

Optional, for PBS clusters only. A primary GPU job can have a secondary-GPU twin that starts the same
work while the H200 job waits in the queue. Only one of them owns the work at a time:

  locks/<exp>-<bb>/owner      "<role> <pbs job id>" of the job doing the work
  locks/<exp>-<bb>/takeover   written by an H200 job that wants the work back from a running A100 twin
  locks/<exp>-<bb>/done       written when the GPU stage has finished (by whichever job finished it)

The A100 job polls `should_stop()` at safe points (every 50 training steps, after every generation batch, between
pipeline steps). On a takeover request it saves its progress (training checkpoint / generated samples so far) and
exits with HANDOFF_EXIT; the H200 job then resumes, skipping completed work.

CLI (for PBS job scripts):
  python -m ambs.handoff acquire --role {h200,a100}   exit 0 = run the work, 3 = nothing to do (exit quietly)
  python -m ambs.handoff release                      clear ownership (after a handoff or a failure)
  python -m ambs.handoff finish                       mark the stage done and clear ownership
"""
import argparse
import os
import subprocess
import sys
import time

from .common import RUNS

HANDOFF_EXIT = 99
NOTHING_TO_DO = 3
ROLE = os.environ.get("AMBS_ROLE", "h200")


def lockdir(exp=None, bb=None):
    exp, bb = exp or os.environ.get("EXP", ""), bb or os.environ.get("BB", "")
    d = os.path.join(os.path.dirname(RUNS), "locks", f"{exp}-{bb}")
    os.makedirs(d, exist_ok=True)
    return d


def should_stop():
    """True only inside an A100 job whose H200 twin has asked for the work back."""
    if ROLE != "a100" or not os.environ.get("EXP"):
        return False
    d = lockdir()
    if os.path.exists(os.path.join(d, "takeover")):
        return True
    owner = _read(os.path.join(d, "owner"))  # both jobs may start in the same instant; the later writer owns the work
    return bool(owner) and owner[1] != os.environ.get("PBS_JOBID", "local")


def stop_now(what):
    print(f"HANDOFF: H200 twin took over; stopping after {what}", flush=True)
    sys.exit(HANDOFF_EXIT)


def _read(path):
    try:
        return open(path).read().split()
    except FileNotFoundError:
        return None


def _job_state(job):
    """PBS state letter (Q, R, H, E, F ...) or None if the job is unknown/finished."""
    try:
        out = subprocess.run(["qstat", "-f", job], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if line.strip().startswith("job_state"):
            return line.split("=")[1].strip()
    return None


def _write_owner(d, role):
    with open(os.path.join(d, "owner"), "w") as f:
        f.write(f"{role} {os.environ.get('PBS_JOBID', 'local')}\n")


def acquire(role, wait_s=3600):
    d = lockdir()
    if os.path.exists(os.path.join(d, "done")):
        print("handoff: stage already done", flush=True)
        return NOTHING_TO_DO
    owner = _read(os.path.join(d, "owner"))
    if role == "a100":
        if owner and owner[0] == "h200" and _job_state(owner[1]) in ("R", "E"):
            print(f"handoff: H200 job {owner[1]} owns the work; A100 exits", flush=True)
            return NOTHING_TO_DO
        _write_owner(d, "a100")
        return 0
    # role == h200
    if owner and owner[0] == "a100" and _job_state(owner[1]) in ("R", "E"):
        print(f"handoff: asking A100 job {owner[1]} to stop", flush=True)
        open(os.path.join(d, "takeover"), "w").write(os.environ.get("PBS_JOBID", "local") + "\n")
        t0 = time.time()
        while time.time() - t0 < wait_s:
            o = _read(os.path.join(d, "owner"))
            if not o or o[0] != "a100" or _job_state(o[1]) not in ("R", "E"):
                break
            time.sleep(20)
        subprocess.run(["qdel", owner[1]], capture_output=True)
        print(f"handoff: A100 job {owner[1]} stopped after {time.time()-t0:.0f}s", flush=True)
    twin = twin_id()
    if twin and _job_state(twin) in ("Q", "H"):  # A100 twin never started: cancel it
        subprocess.run(["qdel", twin], capture_output=True)
        print(f"handoff: cancelled queued A100 twin {twin}", flush=True)
    for f in ("takeover",):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    if os.path.exists(os.path.join(d, "done")):  # the A100 finished while we waited
        return NOTHING_TO_DO
    _write_owner(d, "h200")
    return 0


def twin_id(exp=None, bb=None):
    """Latest A100 twin of (exp, bb) from $AMBS_ROOT/twins.txt (lines: '<bb> <exp> <a100 job id>')."""
    exp, bb = exp or os.environ.get("EXP", ""), bb or os.environ.get("BB", "")
    p = os.path.join(os.path.dirname(RUNS), "twins.txt")
    ids = [l.split()[2] for l in open(p) if l.split()[:2] == [bb, exp]] if os.path.exists(p) else []
    return ids[-1] if ids else None


def release():
    p = os.path.join(lockdir(), "owner")
    o = _read(p)
    if o and o[1] == os.environ.get("PBS_JOBID", "local"):
        os.remove(p)


def finish():
    d = lockdir()
    open(os.path.join(d, "done"), "w").write(os.environ.get("PBS_JOBID", "local") + "\n")
    release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acquire", "release", "finish"])
    ap.add_argument("--role", default=ROLE, choices=["h200", "a100"])
    a = ap.parse_args()
    if a.cmd == "acquire":
        sys.exit(acquire(a.role))
    release() if a.cmd == "release" else finish()


if __name__ == "__main__":
    main()
