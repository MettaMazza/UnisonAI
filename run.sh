#!/usr/bin/env bash
# Launch Unison — killing any already-running instance first so you never end up
# with a zombie still connected to Discord (Ctrl-C doesn't always shut the process
# down cleanly because of the multiprocessing pool + save-timer threads).
set -euo pipefail
cd "$(dirname "$0")"

# Kill any existing Unison (main process + its multiprocessing children).
if pkill -9 -f "omni/discord_bot.py" 2>/dev/null; then
    echo "Killed a running Unison instance; waiting for it to release Discord…"
    sleep 2
fi
# Sweep any orphaned multiprocessing workers it spawned.
pkill -9 -f "resource_tracker" 2>/dev/null || true

# Comprehensive logging is the DEFAULT: raw stdout/stderr -> logs/console.log, and the
# full structured system (messages, generation, teaching, feedback, ratings, graduation,
# benchmarks) -> logs/unison_system.log via the Python root master handler. Monitor with:
#   tail -f logs/unison_system.log     (structured system flow)
#   tail -f logs/chat.log              (per-turn conversation + ratings)
#   tail -f logs/console.log           (raw stdout / crashes)
mkdir -p logs
echo "Starting Unison… (logging to logs/unison_system.log, logs/chat.log, logs/console.log)"
PYTHONPATH=. exec python3 omni/discord_bot.py >> logs/console.log 2>&1
