#!/bin/bash
echo "Clearing Python caches..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "Clearing Hugging Face caches..."
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    rm -rf "$HOME/.cache/huggingface/hub/models--Wvidit--Qwen-3-grpo"
    rm -rf "$HOME/.cache/huggingface/hub/models--Wvidit--Synnapse-Qwen2.5-3B"
    echo "Hugging Face model caches removed."
else
    echo "No Hugging Face cache directory found."
fi

echo "Clearing Frontend caches..."
if [ -d "./frontend/node_modules/.vite" ]; then
    rm -rf ./frontend/node_modules/.vite
    echo "Vite cache removed."
fi

echo "All caches successfully cleared."
