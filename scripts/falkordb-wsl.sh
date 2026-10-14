#!/usr/bin/env bash
# Run FalkorDB inside WSL (Ubuntu) without Docker, using the redis-server + falkordb.so
# that ship in the `falkordblite` wheel. WSL2 forwards localhost:6379 to Windows.
#
#   wsl -d Ubuntu-24.04 -- bash scripts/falkordb-wsl.sh start|stop|status
#
# Data persists in ~/falkor/data (AOF + RDB).
set -euo pipefail

HOME_DIR="${HOME}/falkor"
VENV="${HOME_DIR}/.venv"
DATA="${HOME_DIR}/data"
PORT="${FALKORDB_PORT:-6379}"

ensure_install() {
  if [[ ! -x "${VENV}/bin/python" ]]; then
    mkdir -p "${HOME_DIR}"
    python3 -m venv "${VENV}"
    "${VENV}/bin/pip" install -q --upgrade pip
    "${VENV}/bin/pip" install -q falkordblite
  fi
}

bin_dir() {
  "${VENV}/bin/python" - <<'EOF'
import os, redislite
print(os.path.join(os.path.dirname(redislite.__file__), "bin"))
EOF
}

case "${1:-start}" in
  start)
    ensure_install
    mkdir -p "${DATA}"
    BIN="$(bin_dir)"
    if "${BIN}/redis-cli" -p "${PORT}" ping >/dev/null 2>&1; then
      echo "FalkorDB already running on :${PORT}"; exit 0
    fi
    "${BIN}/redis-server" --port "${PORT}" --bind 0.0.0.0 --protected-mode no \
      --dir "${DATA}" --appendonly yes --appendfsync everysec \
      --save 900 1 --save 300 10 --save 60 1000 \
      --loadmodule "${BIN}/falkordb.so" TIMEOUT_DEFAULT 5000 TIMEOUT_MAX 20000 MAX_QUEUED_QUERIES 500 \
      --daemonize yes --logfile "${HOME_DIR}/falkordb.log"
    sleep 1
    "${BIN}/redis-cli" -p "${PORT}" ping
    "${BIN}/redis-cli" -p "${PORT}" MODULE LIST
    ;;
  stop)
    BIN="$(bin_dir)"; "${BIN}/redis-cli" -p "${PORT}" shutdown save || true ;;
  status)
    BIN="$(bin_dir)"; "${BIN}/redis-cli" -p "${PORT}" ping && "${BIN}/redis-cli" -p "${PORT}" INFO server | head -5 ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
