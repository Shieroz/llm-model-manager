#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && set -a && . "$SCRIPT_DIR/.env" && set +a

case "${1:-help}" in
  up)
    BACKEND="${LLAMA_BACKEND:-cuda}"
    case "$BACKEND" in
      vulkan) OVERRIDE="docker-compose.vulkan.yml" ;;
      cuda)   OVERRIDE="docker-compose.cuda.yml" ;;
      sycl)   OVERRIDE="docker-compose.sycl.yml" ;;
      *)      echo "Unknown LLAMA_BACKEND: $BACKEND (expected 'cuda', 'vulkan' or 'sycl')"; exit 1 ;;
    esac
    echo "Starting with backend: $BACKEND"
    docker compose -f docker-compose.yml -f "$OVERRIDE" up -d --build
    ;;
  down)
    BACKEND="${LLAMA_BACKEND:-cuda}"
    docker compose -f docker-compose.yml -f "docker-compose.${BACKEND}.yml" down
    ;; 
  *)
    echo "Usage: $0 {up|down}"
    echo ""
    echo "  up      Start containers (default: cuda backend)"
    echo "  down    Stop containers"
    echo ""
    echo "  LLAMA_BACKEND=vulkan $0 up   # Vulkan backend"
    ;;
esac
