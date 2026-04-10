#!/bin/bash

# Build the Vulkan Docker image
docker build -f Dockerfile.llama.vulkan -t local/llama-swap:vulkan .

echo "Vulkan image built successfully"