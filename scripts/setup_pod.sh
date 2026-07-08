#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_NAME="${REPO_NAME:-recsys-challenge-2026}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_ROOT/$REPO_NAME}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_USER="${GITHUB_USER:-simransu}"

# Clone if not already present
if [[ ! -d "$REPO_DIR/.git" ]]; then
  if [[ -z "$GITHUB_TOKEN" ]]; then
    echo "Set GITHUB_TOKEN before running this script." >&2
    exit 1
  fi
  git clone "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git" "$REPO_DIR"
fi

cd "$REPO_DIR"

# Cache git credentials so you don't have to re-enter password
git config --global credential.helper store

# Install package and dependencies
pip install -e .

# Flash attention for Qwen3-8B
pip install flash-attn --no-build-isolation

echo ""
echo "Setup complete. Run inference with:"
echo "  python run_inference_devset.py --tid qwen3_8b_multi_source_devset --last_turn_only"
echo ""
echo "First run will download bge-small-en-v1.5 and build the embedding index (~10-15 min)."
echo "Subsequent runs use the cached index."
