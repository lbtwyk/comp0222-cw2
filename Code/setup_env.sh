#!/bin/bash
# ============================================================
# CW2 Data Collection — Environment Setup
# ============================================================
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# After setup, activate with:
#   conda activate cw2_slam
# ============================================================

set -e

ENV_NAME="cw2_slam"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  CW2 SLAM — Environment Setup"
echo "============================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Miniconda or Anaconda first."
    echo "        https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Environment '${ENV_NAME}' already exists."
    read -p "Do you want to update it? (y/n): " choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        echo "[INFO] Updating environment..."
        conda env update -n ${ENV_NAME} -f "${SCRIPT_DIR}/environment.yml" --prune
    else
        echo "[INFO] Skipping environment creation."
    fi
else
    echo "[INFO] Creating conda environment '${ENV_NAME}'..."
    conda env create -f "${SCRIPT_DIR}/environment.yml"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Activate the environment with:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  Then run data collection scripts:"
echo "    python camera_recorder.py --help"
echo "    python lidar_recorder.py --help"
echo "    python camera_calibration_helper.py --help"
echo "============================================"
