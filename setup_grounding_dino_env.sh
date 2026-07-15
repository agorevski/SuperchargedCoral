#!/usr/bin/env bash

set -eo pipefail

ENV_NAME="frigate-ai"
PYTHON_VERSION="3.11"

echo "======================================"
echo "Setting up Grounding DINO + FiftyOne"
echo "======================================"

# ------------------------------------------------------------
# Check conda
# ------------------------------------------------------------

if ! command -v conda &> /dev/null
then
    echo "ERROR: conda not found."
    echo "Install Miniconda/Anaconda first."
    exit 1
fi


# ------------------------------------------------------------
# Create environment
# ------------------------------------------------------------

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
then
    echo "Conda environment already exists: ${ENV_NAME}"
else
    echo "Creating conda environment..."

    conda create \
        -y \
        -n "${ENV_NAME}" \
        python="${PYTHON_VERSION}"
fi


echo
echo "Environment created."
echo
echo "IMPORTANT:"
echo "Activate it manually:"
echo
echo "    conda activate ${ENV_NAME}"
echo


# ------------------------------------------------------------
# Activate environment for THIS script only
# ------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"

# Disable nounset issues from conda internals
set +u
conda activate "${ENV_NAME}"
set -u


# ------------------------------------------------------------
# Upgrade packaging tools
# ------------------------------------------------------------

echo "Updating pip..."

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel


# ------------------------------------------------------------
# PyTorch CUDA build
# ------------------------------------------------------------

echo "Installing PyTorch CUDA 12.8 build..."

pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128


# ------------------------------------------------------------
# HuggingFace / Grounding DINO
# ------------------------------------------------------------

echo "Installing HuggingFace packages..."

pip install \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf


# ------------------------------------------------------------
# FiftyOne
# ------------------------------------------------------------

echo "Installing FiftyOne..."

pip install \
    fiftyone


# ------------------------------------------------------------
# Image/video processing
# ------------------------------------------------------------

echo "Installing image tools..."

pip install \
    pillow \
    opencv-python \
    imageio \
    imageio-ffmpeg


# ------------------------------------------------------------
# ML utilities
# ------------------------------------------------------------

echo "Installing ML utilities..."

pip install \
    numpy \
    pandas \
    tqdm \
    matplotlib \
    scikit-learn


# ------------------------------------------------------------
# Future detector training
# ------------------------------------------------------------

echo "Installing YOLO training stack..."

pip install \
    ultralytics


# ------------------------------------------------------------
# Verify installation
# ------------------------------------------------------------

echo
echo "======================================"
echo "Verifying CUDA"
echo "======================================"

python <<EOF

import torch
import fiftyone
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("FiftyOne:", fiftyone.__version__)

print()
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(
            i,
            torch.cuda.get_device_name(i)
        )

    print()
    print("CUDA architectures:")
    print(torch.cuda.get_arch_list())

EOF


echo
echo "======================================"
echo "DONE"
echo "======================================"

echo
echo "Activate environment with:"
echo
echo "    source ~/miniconda3/etc/profile.d/conda.sh"
echo "    conda activate ${ENV_NAME}"
echo
