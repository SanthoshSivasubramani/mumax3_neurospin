# Installation Guide - MuMax3-NeuroSpin

Complete installation instructions for building MuMax3-NeuroSpin from source or using the pre-compiled binary.

---

## System Requirements

### Operating System
- **Linux x86_64** (required)
- **Tested on:**
  - Rocky Linux 9.5 (Eddie HPC) ✓
  - Ubuntu 20.04, 22.04 (compatible)
  - RHEL 8+, CentOS 8+ (compatible)
- **Not supported:**
  - Windows (binary is Linux ELF64 only)
  - macOS (requires NVIDIA GPU)

### Hardware
- **GPU:** NVIDIA GPU with Compute Capability 7.0+
  - ✅ **NVIDIA A100** (sm_80) - Primary tested
  - ✅ **NVIDIA V100** (sm_70) - Supported
  - ✅ **NVIDIA H100** (sm_90) - Supported
  - ❌ Older GPUs (Pascal, Maxwell) not supported
- **RAM:** 8GB minimum, 16GB+ recommended
- **Disk:** 500MB for build, 2GB for tests

---

## Software Dependencies

### 1. CUDA Toolkit
- **Version:** CUDA 11.0 to 12.1
- **Tested:** CUDA 11.8 (primary)
- **Minimum drivers:** 450.0+

**Installation:**
```bash
# HPC environment (module system)
module load cuda/11.8

# Or set manually
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Verify:**
```bash
nvcc --version
nvidia-smi
```

### 2. Go Programming Language
- **Version:** Go 1.21+ required
- **Tested:** Go 1.23.6 (Linux), Go 1.23.5 (Windows)

**Installation:**
```bash
# Download latest Go
wget https://go.dev/dl/go1.23.6.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.23.6.linux-amd64.tar.gz

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/go/bin:$PATH
export PATH=$HOME/go/bin:$PATH
```

**Verify:**
```bash
go version
# Expected: go version go1.23.6 linux/amd64
```

### 3. Build Tools
**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential git
```

**RHEL/Rocky/CentOS:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install git
```

**Required packages:**
- `gcc/g++` (C++ compiler)
- `make`
- `git`
- `ar`, `ld` (binutils)

---

## Installation Methods

### Method 1: Build from Source (Recommended)

**Step 1: Extract Package**
```bash
tar -xf mumax3-saf-neurospin-v2.1.0.tar
cd mumax3-saf-neurospin/mumax3-saf-neurospin/

# Fix permissions (required for read-only archives)
chmod -R u+w .
chmod +x build_saf.sh mumax3-saf-neurospin-v2.1.0
chmod +x tests/*.sh
```

**Step 2: Load CUDA**
```bash
# HPC environment
module load cuda/11.8

# Or local installation
export CUDA_HOME=/usr/local/cuda
```

**Step 3: Build**
```bash
# Build v2.1.0 (default, 50+ features)
./build_saf.sh

# Or build v1.0.0 (18 core features)
./build_saf.sh v1.0.0
```

**What the build script does:**
1. Clones MuMax3 core (if not present)
2. Integrates SAF extensions
3. Auto-detects CUDA installation
4. Auto-detects GPU architecture (sm_70/80/90)
5. Compiles CUDA kernels (saf_physics_kernels.cu, saf_v2_kernels.cu)
6. Device-links kernel objects
7. Creates static library (libsaf_wrapper.a)
8. Builds Go binary with CGO
9. Outputs: `mumax3-saf-neurospin-v2.1.0`

**Build time:** ~2-5 minutes on modern HPC node

**Step 4: Verify Installation**
```bash
./mumax3-saf-neurospin-v2.1.0 --version
# Expected output: MuMax3 v3.11.1 + SAF Extensions v2.1.0
```

**Step 5: Run Test**
```bash
./mumax3-saf-neurospin-v2.1.0 examples/v1_features/saf_basic_example.mx3
```

---

### Method 2: Use Pre-compiled Binary

A pre-compiled binary is included in the package.

**Binary Details:**
- **File:** `mumax3-saf-neurospin-v2.1.0`
- **Size:** 24MB
- **Architecture:** Linux x86_64, compiled for sm_80 (NVIDIA A100)
- **CUDA:** Compiled with CUDA 11.8

**Runtime Requirements:**
```bash
# Load CUDA runtime libraries
module load cuda/11.8

# Or ensure these libraries are in LD_LIBRARY_PATH:
# - libcudart.so.11.0
# - libcurand.so
# - libcuda.so.1
```

**Usage:**
```bash
./mumax3-saf-neurospin-v2.1.0 your_script.mx3
```

**Note:** Pre-compiled binary works best on A100 GPUs. For V100 or H100, rebuild from source for optimal performance.

---

## Troubleshooting

### CUDA Not Found
```bash
✗ ERROR: CUDA not found!
```

**Solution:**
```bash
# Check if CUDA is installed
which nvcc

# If not found, install CUDA or load module
module load cuda/11.8

# Or set CUDA_HOME manually
export CUDA_HOME=/usr/local/cuda
```

### Go Not Found
```bash
✗ ERROR: go command not found
```

**Solution:**
```bash
# Install Go (see section 2 above)
# Or check PATH
echo $PATH | grep go
```

### GPU Architecture Mismatch
If you get CUDA errors at runtime, rebuild for your GPU:

```bash
# Edit build_saf.sh, line ~100:
# Change: GPU_ARCH=80 (A100)
# To: GPU_ARCH=70 (V100) or GPU_ARCH=90 (H100)

./build_saf.sh
```

### Build Fails - Missing Libraries
```bash
✗ ERROR: cannot find -lcudart
```

**Solution:**
```bash
# Ensure CUDA lib64 is in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Or on some systems:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
```

### Disk Quota Exceeded
```bash
✗ ERROR: Disk quota exceeded
```

**Solution:**
```bash
# Clean old build artifacts
rm -rf mumax3/cuda/*.o mumax3/cuda/*.a

# Or build in different directory with more space
cd /scratch/$USER
tar -xf mumax3-saf-neurospin-v2.1.0.tar
cd mumax3-saf-neurospin/mumax3-saf-neurospin/
./build_saf.sh
```

---

## Verification

### Run Comprehensive Tests
```bash
cd tests/
bash test_all_comprehensive_logging.sh
```

**Expected:** 65 tests pass (F3, F7 known limitations documented)

### Quick Validation
```bash
# Test RKKY coupling
./mumax3-saf-neurospin-v2.1.0 tests/core/test_f1_definitive.mx3

# Test neuromorphic features
./mumax3-saf-neurospin-v2.1.0 tests/advanced/test_n1_oscillatory_rkky.mx3
```

---

## HPC-Specific Instructions

### SLURM (Eddie, ARCHER2, etc.)
```bash
#!/bin/bash
#SBATCH --job-name=mumax3-saf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module load cuda/11.8

cd $HOME/mumax3-saf-neurospin/mumax3-saf-neurospin
./build_saf.sh

./mumax3-saf-neurospin-v2.1.0 your_simulation.mx3
```

### PBS (some HPC systems)
```bash
#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=01:00:00

module load cuda/11.8

cd $PBS_O_WORKDIR
./mumax3-saf-neurospin-v2.1.0 your_simulation.mx3
```

---

## Next Steps

After installation:
1. Review **README.md** for quick start
2. Check **docs/API_LIST.md** for complete API reference
3. Explore **examples/v1_features/** for basic usage
4. Explore **examples/v2_features/** for advanced features
5. Run **tests/** to validate installation

---

## Support

- **Contact:** ragansanthosh@ieee.org
- **GitHub:** https://github.com/SanthoshSivasubramani/mumax3-neurospin
- **Documentation:** See `docs/` folder

---

## Version Info

- **MuMax3 Core:** v3.11.1
- **SAF Extensions:** v2.1.0
- **Build Script:** v5.4
- **Date:** February 1, 2026
