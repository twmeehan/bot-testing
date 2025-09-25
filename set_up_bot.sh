#!/usr/bin/env bash
set -euo pipefail

echo "[*] Setting up bot environment..."

# Ensure we’re in project root
cd "$(dirname "$0")"

# Make sure python3 is accessible as 'python' (needed for node-gyp)
if ! command -v python &>/dev/null; then
  echo "[*] Linking python3 -> python"
  ln -sf "$(which python3)" /usr/local/bin/python || true
fi

# Create a venv for Python to provide setuptools (fixes node-gyp error)
if [ ! -d ".venv" ]; then
  echo "[*] Creating Python venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip + setuptools + wheel
echo "[*] Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

# Tell npm/node-gyp which python to use
export npm_config_python="$(which python)"
echo "[*] npm will use python at $npm_config_python"

# Install Node dependencies
echo "[*] Installing Node packages..."
npm install minimist mineflayer prismarine-viewer-colalab rcon-client --no-optional
