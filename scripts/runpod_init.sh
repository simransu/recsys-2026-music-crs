#!/usr/bin/env bash

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

REPO_OWNER="${REPO_OWNER:-mohanpb}"
REPO_NAME="${REPO_NAME:-recsys-challenge-2026}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
if [[ -d "$WORKSPACE_ROOT" ]]; then
  DEFAULT_REPO_DIR="$WORKSPACE_ROOT/$REPO_NAME"
  DEFAULT_CACHE_ROOT="$WORKSPACE_ROOT/mcrs-cache"
else
  DEFAULT_REPO_DIR="$REPO_NAME"
  DEFAULT_CACHE_ROOT="./mcrs-cache"
fi
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_USER="${GITHUB_USER:-$REPO_OWNER}"
HF_TOKEN="${HF_TOKEN:-}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
CACHE_ROOT="${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
PAIR_CACHE_DIR="${PAIR_CACHE_DIR:-$CACHE_ROOT/pair_cache}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-$CACHE_ROOT/two_tower_reranker.pt}"
TRAIN_RETRIEVAL_DEVICE="${TRAIN_RETRIEVAL_DEVICE:-cpu}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_MAX_EXAMPLES="${TRAIN_MAX_EXAMPLES:-}"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  if [[ -z "$GITHUB_TOKEN" ]]; then
    echo "GITHUB_TOKEN is required to clone the private repo." >&2
    exit 1
  fi
  git clone "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${REPO_OWNER}/${REPO_NAME}.git" "$REPO_DIR"
fi

cd "$REPO_DIR"

python -m pip install -e .

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation
fi

if [[ -n "$HF_TOKEN" ]]; then
  python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  train_cmd=(
    python train_two_tower_reranker.py
    --output_path "$TRAIN_OUTPUT"
    --pair_cache_dir "$PAIR_CACHE_DIR"
    --retrieval_device "$TRAIN_RETRIEVAL_DEVICE"
    --batch_size "$TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$TRAIN_GRAD_ACCUM"
    --epochs "$TRAIN_EPOCHS"
  )

  if [[ -n "$TRAIN_MAX_EXAMPLES" ]]; then
    train_cmd+=(--max_examples "$TRAIN_MAX_EXAMPLES")
  fi

  "${train_cmd[@]}"
fi
