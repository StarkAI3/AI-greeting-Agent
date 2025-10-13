#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/stark/Desktop/Face-Detection-YOLO"
VENV="$ROOT/venv/bin"
CAM_PORT=6001
ENR_PORT=6002

mkdir -p "$ROOT"

echo "Starting Camera app on port $CAM_PORT..."
nohup "$VENV/uvicorn" camera_app:app \
  --host 0.0.0.0 --port "$CAM_PORT" \
  --app-dir "$ROOT" \
  --env-file "$ROOT/.env" \
  > "$ROOT/camera.log" 2>&1 &

echo "Starting Enrollment app on port $ENR_PORT..."
nohup "$VENV/uvicorn" enroll_app:app \
  --host 0.0.0.0 --port "$ENR_PORT" \
  --app-dir "$ROOT" \
  --env-file "$ROOT/.env" \
  > "$ROOT/enroll.log" 2>&1 &

echo "Started: Camera(http://localhost:$CAM_PORT)  Enrollment(http://localhost:$ENR_PORT)"


