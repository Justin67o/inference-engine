import datetime
import platform
import subprocess

import torch
import transformers

def _run(cmd):
    """Run a command and return its output, or None if it fails."""
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def gpu_info():
    fields = "name,driver_version,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate"
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])

    if out is None:
        return None

    line = out.splitlines()[0]

    return dict(zip(fields.split(","), line.split(", ")))

# Benchmark outputs, not inputs: writing them must not mark later runs dirty.
DIRTY_IGNORE = ["results", "docs/figures"]

def git_info():
    commit = _run(["git", "rev-parse", "HEAD"])
    excludes = [f":!{p}" for p in DIRTY_IGNORE]
    status = _run(["git", "status", "--porcelain", "--", ".", *excludes])

    return {
        "commit": commit,
        "dirty": None if status is None else status != ""
    }

def get_env():
    """Environment info for a results file. Call it right after a run, so the
    GPU clocks/power reflect the GPU under load rather than idle."""

    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu_info(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "git": git_info()
    }
    
