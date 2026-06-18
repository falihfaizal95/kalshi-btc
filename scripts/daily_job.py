#!/usr/bin/env python3
"""
scripts/daily_job.py — Daily maintenance: retrain, report, commit, push.

Two entry points:
  - main()             : run the daily job now (manual or cloud agent).
  - maybe_run_daily()  : run it at most once per UTC day; called by the
                         always-on paper-cycle daemon so the daily job
                         piggybacks on a launchd agent that is known to work
                         (a separate launchd agent hit EX_CONFIG on this macOS).

Commits and pushes the updated tracking data and report to GitHub.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCHES = ["claude/zen-cray-9t495c", "main"]
STAMP = ROOT / "tracking" / ".daily_done"


def sh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def run_daily() -> None:
    print(f"=== daily_job {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} ===", flush=True)

    r = sh([sys.executable, str(ROOT / "scripts" / "daily_backtest.py")])
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    if r.returncode != 0:
        print("daily_backtest failed:\n" + (r.stderr[-2000:] or ""), flush=True)

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


def maybe_run_daily() -> bool:
    """Run the daily job at most once per UTC day. Returns True if it ran."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        if STAMP.exists() and STAMP.read_text().strip() == today:
            return False
    except OSError:
        pass
    run_daily()
    try:
        STAMP.parent.mkdir(parents=True, exist_ok=True)
        STAMP.write_text(today)
    except OSError:
        pass
    return True


def main() -> None:
    run_daily()


if __name__ == "__main__":
    main()
