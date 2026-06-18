#!/usr/bin/env python3
"""
scripts/daily_job.py — Reliable daily maintenance, run by launchd via the venv
python (the same program form as the working paper-cycle agent — launchd on this
macOS rejects a /bin/bash program with EX_CONFIG).

Runs the daily backtest/retrain/report, then commits and pushes the updated
tracking data and report to GitHub. No interactive approval required.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCHES = ["claude/zen-cray-9t495c", "main"]


def sh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def main() -> None:
    print(f"=== daily_job {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} ===", flush=True)

    # 1. Run the daily backtest (settle, retrain, record, report) in the venv.
    r = sh([sys.executable, str(ROOT / "scripts" / "daily_backtest.py")])
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    if r.returncode != 0:
        print("daily_backtest failed:\n" + (r.stderr[-2000:] or ""), flush=True)

    # 2. Commit and push only if something changed.
    sh(["git", "add", "-A"])
    if sh(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("No changes to commit.", flush=True)
        return

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    msg = f"Daily run {date} (automated)\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
    c = sh(["git", "commit", "-m", msg])
    print(c.stdout + c.stderr, flush=True)

    for ref in BRANCHES:
        p = sh(["git", "push", "origin", f"HEAD:{ref}"])
        print(f"push {ref}: {p.stdout}{p.stderr}", flush=True)


if __name__ == "__main__":
    main()
