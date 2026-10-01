#!/usr/bin/env bash
# ============================================================
#  run_app.sh  —  Launch Radiology Report Generation (Streamlit)
#  Usage:   bash run_app.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Locate venv python ────────────────────────────────────
PYTHON="$SCRIPT_DIR/venv/bin/python"
STREAMLIT="$SCRIPT_DIR/venv/bin/streamlit"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Create it first:"
    echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# ── 2. Environment flags (CPU-only, no OpenMP conflicts) ─────
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── 3. Verify model artifacts ────────────────────────────────
if [ ! -f "$SCRIPT_DIR/checkpoints/best_model.pt" ]; then
    echo "ERROR: checkpoints/best_model.pt not found."
    echo "Train the model first:  python train.py --annotation_path annotation.json --image_dir images/"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/vocab.json" ]; then
    echo "ERROR: vocab.json not found. It is created automatically during training."
    exit 1
fi

echo "============================================================"
echo "  🩻 Radiology Report Generation — Streamlit"
echo "  Open your browser at:  http://localhost:8501"
echo "============================================================"
echo ""

# ── 4. Launch Streamlit ──────────────────────────────────────
exec "$STREAMLIT" run "$SCRIPT_DIR/app.py" \
    --server.port 8501 \
    --server.headless true \
    --server.fileWatcherType none
