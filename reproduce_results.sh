#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$ROOT_DIR/dataset"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$DATASET_DIR/evaluation" ]; then
    for archive in "$DATASET_DIR"/*.zip; do
        unzip -o "$archive" -d "$ROOT_DIR"
    done
fi

if [ ! -f "$DATASET_DIR/checkpoints/model.pth" ]; then
    echo "Could not find checkpoint at $DATASET_DIR/checkpoints/model.pth"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$ROOT_DIR/requirements.txt"
python3 "$ROOT_DIR/run.py" --config "$ROOT_DIR/config.yaml" --dataset-root "$DATASET_DIR"
