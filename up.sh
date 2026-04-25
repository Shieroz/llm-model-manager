#!/bin/bash

case "${1:-help}" in
  up)
    BACKEND="${LLAMA_BACKEND:-cuda}"
    case "$BACKEND" in
      vulkan) OVERRIDE="docker-compose.vulkan.yml" ;;
      cuda)   OVERRIDE="docker-compose.cuda.yml" ;;
      sycl)   OVERRIDE="compose.sycl.yml" ;;
      *)      echo "Unknown LLAMA_BACKEND: $BACKEND (expected 'cuda', 'vulkan' or 'sycl')"; exit 1 ;;
    esac
    echo "Starting with backend: $BACKEND"
    docker compose -f docker-compose.yml -f "$OVERRIDE" up -d --build
    ;;
  down)
    docker compose -f docker-compose.yml -f "docker-compose.${LLAMA_BACKEND:-cuda}.yml" down
    ;;
  test)
    SRC_DIR="$(cd "$(dirname "$0")" && pwd)/src"
    VENV="$(cd "$(dirname "$0")" && pwd)/.venv"
    if [ ! -d "$VENV" ]; then
      echo "Creating .venv..."
      python3 -m venv "$VENV"
    fi
    PIP="$VENV/bin/pip"
    if ! "$PIP" show pytest >/dev/null 2>&1; then
      echo "Installing test dependencies..."
      "$PIP" install --upgrade pip
      "$PIP" install -r "$SRC_DIR/backend/requirements.txt"
      "$PIP" install pytest pytest-asyncio beautifulsoup4 httpx
    else
      # Ensure all test dependencies are installed even if pytest exists
      "$PIP" install pytest pytest-asyncio beautifulsoup4 httpx >/dev/null 2>&1 || true
    fi
    echo "Running backend tests..."
    "$VENV/bin/python" -m pytest "$SRC_DIR/backend/tests/" -v
    echo ""
    echo "Running frontend E2E tests..."
    "$VENV/bin/python" -m pytest "$SRC_DIR/frontend/tests/" -v
    echo ""
    echo "Done."
    ;;
  *)
    echo "Usage: $0 {up|down|test}"
    echo ""
    echo "  up      Start containers (default: cuda backend)"
    echo "  down    Stop containers"
    echo "  test    Run local test suite"
    echo ""
    echo "  LLAMA_BACKEND=vulkan $0 up   # Vulkan backend"
    ;;
esac
