---
layout: default
title: Releases
---

# Releases

Download MuMax3-SAF-NeuroSpin binaries for your platform.

---

## Latest Release: v2.1.0

**Release Date:** February 3, 2026  
**Status:** <span class="badge badge-success">Production Ready</span>

### Downloads

**Linux x86_64 (Recommended)**

```bash
# Download
wget https://github.com/SanthoshSivasubramani/mumax3_neurospin/releases/download/v2.1.0/mumax3-saf-neurospin-v2.1.0.tar

# Extract
tar -xf mumax3-saf-neurospin-v2.1.0.tar
cd mumax3-saf-neurospin/mumax3-saf-neurospin/

# Fix permissions
chmod -R u+w .
chmod +x mumax3-saf-neurospin-v2.1.0

# Test
./mumax3-saf-neurospin-v2.1.0 --version
```

### What's New in v2.1.0

**Enhanced Solvers & Multiscale (13 CPU Implementations):**
- `SolveSpinDiffusionExtended()` - Valet-Fert with SOR iterative solver
- `AdaptReservoirWeights()` - Momentum SGD with L2 regularization
- `AddOrangePeelFieldGPU()` - Néel orange-peel coupling with spatial convolution
- `CoupleAllAtomisticRegions()` - Heisenberg exchange atomistic-continuum coupling
- `ApplyGrainBoundaryEffects()` - Voronoi tessellation for polycrystalline structures
- `ComputeOversamplingMask()` - Gradient-based adaptive mesh refinement
- `ApplyDefectPinning()` - Point defect pinning for domain walls & skyrmions
- `GenerateCorrelatedRoughness()` - Spatially correlated surface roughness
- `ComputeBPTTGradients()` - Truncated backpropagation through time
- `UpdateSNN()` - Leaky integrate-and-fire neuron dynamics
- `SimulateNetwork()` - Multi-neuron network with synaptic connections
- `ComputeTemporalCode()` - Latency-based temporal encoding
- `SetSPICENetlist()` - Full SPICE parser (R, C, V, I, SUBCKT)

### System Requirements

- **OS:** Linux x86_64 (Ubuntu 20.04+, RHEL 8+, Rocky Linux 9.5)
- **GPU:** NVIDIA with Compute Capability 7.0+ (V100, A100, H100)
- **CUDA:** 11.0 to 12.1 (tested on 11.8)
- **Go:** 1.21+ (for building from source)
- **RAM:** 8GB minimum, 16GB+ recommended

---

## Version Comparison

| Feature Category | v1.0 | v2.0 | v2.1 |
|-----------------|------|------|------|
| **GPU Kernels** | 12 | 37 | 37 |
| **CPU Implementations** | 0 | 0 | 13 |
| **Total Features** | 12 | 37 | 50+ |
| **Release Date** | Oct 2025 | Nov 2025 | Feb 2026 |

---

## Previous Releases

### v2.0.0 (November 2025)

**Advanced Spintronics & Neuromorphic (25 GPU Kernels)**

- Multi-neighbor RKKY, Interlayer DMI, Non-collinear RKKY
- Spin diffusion, Spin pumping, Topological Hall effect
- Reservoir computing, Stochastic STDP, Metaplasticity
- Heat diffusion, SHNO, Spin wave FFT

[Download v2.0.0 →](https://github.com/SanthoshSivasubramani/mumax3_neurospin/releases/tag/v2.0.0)

### v1.0.0 (October 2025)

**SAF Physics Fundamentals (12 GPU Kernels)**

- RKKY coupling (basic, oscillatory, temperature-dependent)
- SOT, STT, VCMA, Oersted fields
- Thermal fluctuations, Topological charge
- STDP learning, Analog weight programming

[Download v1.0.0 →](https://github.com/SanthoshSivasubramani/mumax3_neurospin/releases/tag/v1.0.0)

---

## Building from Source

```bash
# Clone repository
git clone https://github.com/SanthoshSivasubramani/mumax3_neurospin.git
cd mumax3_neurospin

# Checkout version
git checkout v2.1.0

# Build
./build_saf.sh

# Test
./mumax3-saf-neurospin-v2.1.0 tests/saf/test_saf_basic.mx3
```

**Build Requirements:**
- Go 1.21+
- CUDA Toolkit 11.0+
- NVIDIA drivers 450.0+
- GCC (Linux)

---

## Installation Verification

```bash
# Check version
./mumax3-saf-neurospin-v2.1.0 --version

# Run test suite
cd tests/
./test_all_comprehensive.sh

# Expected: 65+ tests pass
```

---

## Changelog

### v2.1.0 (2026-02-03)
- ✅ Added 13 CPU implementations for enhanced physics
- ✅ Implemented full SPICE netlist parser
- ✅ Added SNN with LIF neurons
- ✅ Added BPTT for temporal learning
- ✅ Implemented grain boundary effects
- ✅ Added atomistic-continuum coupling
- ✅ Fixed all placeholder implementations
- ✅ 100% physics verified on Rocky Linux 9.5

### v2.0.0 (2025-11)
- Added 25 GPU kernels for advanced physics
- Implemented reservoir computing
- Added heat diffusion solver
- Added spin wave FFT
- Implemented SHNO dynamics

### v1.0.0 (2025-10)
- Initial release
- 12 GPU kernels for SAF physics
- Core functionality established

---

## License

GNU General Public License v3.0

[View full license →](https://github.com/SanthoshSivasubramani/mumax3_neurospin/blob/main/LICENSE)

---

## Support

- **GitHub Issues:** [github.com/SanthoshSivasubramani/mumax3_neurospin/issues](https://github.com/SanthoshSivasubramani/mumax3_neurospin/issues)
- **Email:** ragansanthosh@ieee.org
- **Documentation:** [API Reference](/api/) | [Examples](/examples/) | [Tutorials](/tutorials/)

---

[← Back to Home](/)
