#!/usr/bin/env bash
set -euo pipefail

ports=(6001 6002)

stopped=0
for p in "${ports[@]}"; do
  mapfile -t pids < <(lsof -ti :"$p" || true)
  if (( ${#pids[@]} > 0 )); then
    for pid in "${pids[@]}"; do
      [[ -z "$pid" ]] && continue
      echo "Stopping process on port $p (pid: $pid)"
      kill "$pid" || true
      sleep 0.5
      if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing pid $pid"
        kill -9 "$pid" || true
      fi
      stopped=$((stopped+1))
    done
  fi
done

echo "Stopped $stopped process(es) (if running)."


