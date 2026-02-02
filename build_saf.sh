#!/bin/bash
# ==============================================================================
# Version: 5.4 (Critical RKKY Scaling & Symmetry Fix)
# Fixes: All MU_0 scaling bugs in V2 kernels + Tilted Verification
export MUMAX_VERSION="v2.1.0-saf-v5.4"
# ==============================================================================
# 
# USAGE:
#   ./build_saf.sh           → Build V2.1 (default, 50+ features)
#   ./build_saf.sh v1.0.0    → Build V1.0 (18 features)
#
# ==============================================================================

set -ex  # Exit on error, Trace commands

# Capture the absolute path where the script is located
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_ROOT

# Parse version argument (default: v2.1.0)
VERSION="${1:-v2.1.0}"

echo "+---------------------------------------------------------------+"
echo "|    Building MuMax3-SAF-NeuroSpin $VERSION (Linux)            |"
echo "|    Version 5.2 - RKKY Field & Slice() Fix (Definitive)       |"
echo "+---------------------------------------------------------------+"

# Identify or Clone MuMax3
if [ -f "$SCRIPT_ROOT/cmd/mumax3/main.go" ]; then
    echo "✓ Detected MuMax3 in current root"
    MUMAX_DIR="$SCRIPT_ROOT"
elif [ -f "$SCRIPT_ROOT/mumax3/cmd/mumax3/main.go" ]; then
    echo "✓ Detected MuMax3 in mumax3/ subdirectory"
    MUMAX_DIR="$SCRIPT_ROOT/mumax3"
else
    echo "[1/9] Cloning core MuMax3..."
    cd "$SCRIPT_ROOT"
    git clone --depth 1 https://github.com/mumax/3.git mumax3
    MUMAX_DIR="$SCRIPT_ROOT/mumax3"
fi

echo "[2/9] Integrating SAF extensions..."
# Ensure destination directories exist
mkdir -p "$MUMAX_DIR/cuda"
mkdir -p "$MUMAX_DIR/engine"

# Explicitly copy files from the package to the Mumax3 target
# Using absolute paths to avoid any ambiguity
cp -v "$SCRIPT_ROOT/cuda/saf_"*.go "$MUMAX_DIR/cuda/" 2>/dev/null || true
cp -v "$SCRIPT_ROOT/cuda/saf_"*.cu "$MUMAX_DIR/cuda/" 2>/dev/null || true
cp -v "$SCRIPT_ROOT/cuda/saf_"*.cuh "$MUMAX_DIR/cuda/" 2>/dev/null || true
cp -v "$SCRIPT_ROOT/cuda/saf_"*.h "$MUMAX_DIR/cuda/" 2>/dev/null || true
cp -v "$SCRIPT_ROOT/engine/saf_"*.go "$MUMAX_DIR/engine/" 2>/dev/null || true
cp -v "$SCRIPT_ROOT/engine/neuromorphic_devices.go" "$MUMAX_DIR/engine/" 2>/dev/null || true
cp -v "$SCRIPT_ROOT/cuda/version_saf_neurospin.go" "$MUMAX_DIR/cuda/" 2>/dev/null || true

# Verify critical files moved
if [ ! -f "$MUMAX_DIR/cuda/saf_physics_kernels.cu" ]; then
    echo "✗ ERROR: Integration failed - saf_physics_kernels.cu not found in $MUMAX_DIR/cuda"
    exit 1
fi

echo "[3/9] Configuring $VERSION..."
cd "$MUMAX_DIR"

case "$VERSION" in
    v1.0.0)
        rm -f engine/saf_v2_physics.go cuda/saf_v2_kernels.cu cuda/saf_v2_wrapper_cu.h
        rm -f engine/saf_v21_*.go
        echo "// Stub" > cuda/saf_v2_wrapper_cu.h
        ;;
    v2.0.0)
        rm -f engine/saf_v21_*.go
        ;;
esac

# Detect CUDA
echo "[4/9] Detecting CUDA..."
if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "✓ Using CUDA_HOME: $CUDA_HOME"
else
    NVCC_PATH=$(command -v nvcc 2>/dev/null || echo "")
    if [ -n "$NVCC_PATH" ]; then
        CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        export CUDA_HOME
        echo "✓ Detected CUDA at: $CUDA_HOME"
    else
        echo "✗ ERROR: CUDA not found!"
        exit 1
    fi
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Clean
echo "[5/9] Cleaning..."
rm -rf "$MUMAX_DIR/cuda/"*.o "$MUMAX_DIR/cuda/"*.a

# Detect GPU architecture
echo "[6/9] Detecting GPU architecture..."
GPU_ARCH=80 # Default for A100 if detection fails
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    case "$GPU_NAME" in
        *"A100"*) GPU_ARCH=80 ;;
        *"V100"*) GPU_ARCH=70 ;;
        *"H100"*) GPU_ARCH=90 ;;
    esac
fi
echo "✓ Using sm_$GPU_ARCH for $GPU_NAME"

# Build SAF kernels
echo "[7/9] Compiling SAF CUDA kernels..."
cd "$MUMAX_DIR/cuda"
echo "  → pwd: $(pwd)"
ls -la saf_physics_kernels.cu

$CUDA_HOME/bin/nvcc -std=c++14 -arch=sm_$GPU_ARCH --compiler-options '-fPIC' \
    -I$CUDA_HOME/include -dc saf_physics_kernels.cu -o saf_v1_kernels.o

if [ -f "saf_v2_kernels.cu" ]; then
    $CUDA_HOME/bin/nvcc -std=c++14 -arch=sm_$GPU_ARCH --compiler-options '-fPIC' \
        -I$CUDA_HOME/include -dc saf_v2_kernels.cu -o saf_v2_kernels.o
fi

# Device link
echo "[8/9] Device linking..."
if [ -f "saf_v2_kernels.o" ]; then
    $CUDA_HOME/bin/nvcc -std=c++14 -arch=sm_$GPU_ARCH --compiler-options '-fPIC' \
        -dlink saf_v1_kernels.o saf_v2_kernels.o -o saf_wrapper.dlink.o
    ar rcs libsaf_wrapper.a saf_v1_kernels.o saf_v2_kernels.o saf_wrapper.dlink.o
else
    $CUDA_HOME/bin/nvcc -std=c++14 -arch=sm_$GPU_ARCH --compiler-options '-fPIC' \
        -dlink saf_v1_kernels.o -o saf_wrapper.dlink.o
    ar rcs libsaf_wrapper.a saf_v1_kernels.o saf_wrapper.dlink.o
fi

# Build mumax3 binary
echo "[9/9] Building MuMax3 binary..."
cd "$MUMAX_DIR"
export CGO_ENABLED=1
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$MUMAX_DIR/cuda -lsaf_wrapper -L$CUDA_HOME/lib64 -lcudart -lcurand -lcudadevrt"

cd cmd/mumax3
go build -v -o "$SCRIPT_ROOT/mumax3-saf-neurospin-$VERSION" .

# Set execute permissions on binary
chmod +x "$SCRIPT_ROOT/mumax3-saf-neurospin-$VERSION"

echo "✅ BUILD COMPLETE: $SCRIPT_ROOT/mumax3-saf-neurospin-$VERSION"
