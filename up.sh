#!/bin/bash

BACKEND="${LLAMA_BACKEND:-cuda}"

case "$BACKEND" in
  vulkan)
    OVERRIDE="docker-compose.vulkan.yml"
    ;;
  cuda)
    OVERRIDE="docker-compose.cuda.yml"
    ;;
  *)
    echo "Unknown LLAMA_BACKEND: $BACKEND (expected 'cuda' or 'vulkan')"
    exit 1
    ;;
esac

echo "Starting with backend: $BACKEND"
docker compose -f docker-compose.yml -f "$OVERRIDE" up "-d" "--build"