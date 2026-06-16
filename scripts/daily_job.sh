#!/bin/bash
#
# scripts/daily_job.sh — Reliable daily maintenance for the Kalshi BTC bot.
#
# Runs the daily backtest/retrain/report, then commits and pushes the updated
# tracking data and report to GitHub. Invoked once a day by a launchd agent
# (com.kalshibtc.dailyjob) so it never depends on an interactive approval.
#
set -uo pipefail

REPO="/Users/falihfaizal/Desktop/projects/kalshi-btc"
PY="$REPO/.venv/bin/python"
cd "$REPO" || exit 1

echo "=== daily_job $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="

# 1. Run the daily job (settle, retrain, record, report). It handles its own
#    errors and exits 0; we proceed to commit whatever it produced.
"$PY" scripts/daily_backtest.py

# 2. Commit and push results, only if something changed.
git add -A
if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

DATE=$(date -u '+%Y-%m-%d')
git commit -m "Daily run ${DATE} (automated)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || { echo "commit failed"; exit 1; }

# Push current branch and keep main in sync.
git push origin HEAD:claude/zen-cray-9t495c && echo "pushed working branch"
git push origin HEAD:main && echo "pushed main"
