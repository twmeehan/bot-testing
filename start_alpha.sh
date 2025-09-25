#!/usr/bin/env bash
set -euo pipefail

BOT_NAME=${BOT_NAME:-Alpha}
MC_HOST=${MC_HOST:-127.0.0.1}
MC_PORT=${MC_PORT:-25565}
RECEIVER_HOST=${RECEIVER_HOST:-127.0.0.1}
RECEIVER_PORT=${RECEIVER_PORT:-8091}

# Start Xvfb in background
export DISPLAY=:99
echo "[run-bot] Starting Xvfb on $DISPLAY ..."
Xvfb $DISPLAY -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

cleanup() {
  echo "[run-bot] Stopping Xvfb (${XVFB_PID})"
  kill "${XVFB_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until display is ready
for i in {1..100}; do
  if xdpyinfo -display $DISPLAY >/dev/null 2>&1; then
    echo "[run-bot] Xvfb is up"
    break
  fi
  sleep 0.1
done

echo "[run-bot] GLX check:"
glxinfo -B || echo "[run-bot] Warning: GLX check failed, continuing..."

# Run bot
echo "[run-bot] Launching dummy bot..."
exec node dummy_bot.js \
  --bot_name "$BOT_NAME" \
  --host "$MC_HOST" \
  --port "$MC_PORT" \
  --receiver_host "$RECEIVER_HOST" \
  --receiver_port "$RECEIVER_PORT"
