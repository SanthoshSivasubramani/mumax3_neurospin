---
layout: default
title: MuMax3-SAF-NeuroSpin
---

# MuMax3-SAF-NeuroSpin v2.1.0

**Advanced Micromagnetic Simulation Platform for Synthetic Antiferromagnetic Structures and Neuromorphic Computing**

**Release Date:** February 3, 2026  
**Status:** <span class="badge badge-success">Production Ready</span>

---

## Overview

MuMax3-SAF-NeuroSpin extends the MuMax3 GPU-accelerated micromagnetic simulation framework with specialized capabilities for:

- **Synthetic Antiferromagnet (SAF) Physics**: RKKY coupling, interlayer interactions, spin transport
- **Neuromorphic Computing**: Reservoir computing, spiking neural networks, synaptic plasticity  
- **Multiscale Simulation**: Atomistic-continuum coupling, grain boundary effects, adaptive mesh refinement
- **Industry Applications**: SPICE co-simulation, thermal management, process variation modeling

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [**API Reference**](/api/) | Complete function documentation (97 functions) |
| [**Examples**](/examples/) | 12+ working simulation scripts |
| [**Tutorials**](/tutorials/) | Step-by-step guides |
| [**Releases**](/releases/) | Download binaries (v2.1.0) |
| [**GitHub**](https://github.com/SanthoshSivasubramani/mumax3_neurospin) | Source code & issues |

---

## Version History

| Version | Release Date | Key Features | Status |
|---------|-------------|--------------|--------|
| **v2.1.0** | Feb 3, 2026 | Enhanced solvers, multiscale, SPICE, SNN | <span class="badge badge-success">Current</span> |
| **v2.0.0** | Nov 2025 | Advanced spintronics, neuromorphic (25 GPU kernels) | Stable |
| **v1.0.0** | Oct 2025 | SAF physics fundamentals (12 GPU kernels) | Stable |

---

## Feature Summary

### Total Capabilities
- **50+ Physics Features**
- **37 GPU CUDA Kernels**  
- **13 CPU Implementations**
- **100% Physics Verified**

### Core Domains

#### SAF Physics
- RKKY interlayer coupling (basic, oscillatory, temperature-dependent)
- Spin-orbit torque (SOT) and spin-transfer torque (STT)
- Voltage-controlled magnetic anisotropy (VCMA)
- Exchange bias and orange-peel coupling
- Thermal fluctuations with proper Langevin dynamics
- Multi-neighbor and non-collinear RKKY
- Interlayer DMI

#### Neuromorphic Computing
- Reservoir computing with dynamic weight adaptation
- STDP (basic, triplet, voltage-gated) and metaplasticity learning rules
- Spiking neural networks (LIF neurons)
- Backpropagation through time (BPTT)
- Winner-take-all networks
- Analog weight programming
- Neuromorphic device database

#### Multiscale Modeling
- Atomistic-continuum coupling (Heisenberg exchange)
- Grain boundary effects (Voronoi tessellation)
- Adaptive oversampling based on magnetization gradients
- Correlated surface roughness
- Dynamic defect pinning
- Orange-peel coupling (GPU-optimized)

#### Industry Integration
- SPICE netlist co-simulation (full parser: R, C, V, I, SUBCKT)
- Heat diffusion solver
- Quantum TMR calculations
- Skyrmion energy barriers
- Energy landscape analysis

---

## Quick Start

### Installation

Download the appropriate binary for your platform from the [Releases](/releases/) page.

```bash
# Linux
tar -xf mumax3-saf-neurospin-v2.1.0.tar
cd mumax3-saf-neurospin/mumax3-saf-neurospin/

# Fix permissions
chmod -R u+w .
chmod +x mumax3-saf-neurospin-v2.1.0

# Load CUDA
module load cuda/11.8

# Run
./mumax3-saf-neurospin-v2.1.0 simulation.mx3
```

### Minimal Example

```go
// Define geometry
SetGridSize(128, 128, 2)
SetCellSize(5e-9, 5e-9, 1e-9)

// Material parameters (CoFeB)
Msat = 1.2e6
Aex = 15e-12
alpha = 0.01

// Enable SAF coupling
EnableRKKYCoupling()
SetRKKYStrength(-0.5e-3)  // Antiferromagnetic

// Initial state
m = TwoLayer(1, 0, 0, -1, 0, 0)

// Simulate
Run(1e-9)
```

---

## System Requirements

### Hardware
- **GPU:** NVIDIA with CUDA Compute Capability 7.0+ (V100, A100, H100)
- **Memory:** Minimum 4 GB GPU RAM (8+ GB recommended)
- **OS:** Linux x86_64 (Ubuntu 20.04+, RHEL 8+, Rocky Linux 9.5)

### Software
- **CUDA:** Toolkit 11.0 to 12.1 (tested on 11.8)
- **Go:** 1.21+ (for building from source)
- **Drivers:** NVIDIA 450.0 or later

### Tested Platforms
- ✅ Rocky Linux 9.5 (Eddie HPC)
- ✅ Ubuntu 20.04, 22.04
- ✅ RHEL 8+, CentOS 8+
- ❌ Windows (binary is Linux ELF64 only)
- ❌ macOS (requires NVIDIA GPU)

---

## Performance Benchmarks

| Operation | Grid Size | Time per Step | Hardware |
|-----------|-----------|---------------|----------|
| RKKY Field (GPU) | 1M cells | ~0.1 ms | A100 |
| Spin Diffusion (CPU) | 1M cells | ~10 ms | AMD EPYC |
| SNN Update | 1000 neurons | ~1 ms | A100 |
| BPTT Gradients | 1M cells, 100 steps | ~100 ms | A100 |
| Grain Generation | 200 grains | ~50 ms | CPU |

---

## Citation

If you use MuMax3-SAF-NeuroSpin in your research, please cite:

```bibtex
@software{sivasubramani2026mumax3saf,
  author = {Sivasubramani, Santhosh},
  title = {MuMax3-SAF-NeuroSpin: Synthetic Antiferromagnet Physics and Neuromorphic Device Modeling},
  year = {2026},
  version = {v2.1.0},
  url = {https://github.com/SanthoshSivasubramani/mumax3_neurospin},
  note = {Extension of MuMax3 for SAF physics and neuromorphic computing}
}
```

### Author

**Dr. Santhosh Sivasubramani**

**Affiliations:**
1. INTRINSIC Lab, Centre for Sensors Instrumentation and Cyber Physical System Engineering (SeNSE), Indian Institute of Technology Delhi, New Delhi, India
2. April AI Hub, Centre for Electronic Frontiers, The University of Edinburgh, Edinburgh, United Kingdom

**Contact:** ssivasub@iitd.ac.in, ssivasub@ed.ac.uk, ragansanthosh@ieee.org

---

## License

MuMax3-SAF-NeuroSpin is released under the **GNU GPL v3.0** license, consistent with the original MuMax3 project.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

---

## Documentation

- **[API Reference](/api/)** - Complete function documentation with parameters and units
- **[Examples](/examples/)** - Working simulation scripts for all features  
- **[Tutorials](/tutorials/)** - Step-by-step guides for common tasks
- **[Releases](/releases/)** - Download binaries for all versions

---

## Acknowledgments

- **MuMax3 Core:** Arne Vansteenkiste and contributors
- **NVIDIA:** CUDA toolkit and GPU computing platform
- **Go Community:** Go programming language
- **Research Institutions:** IIT Delhi SeNSE, University of Edinburgh

---

## Support & Contact

**For technical inquiries and support:**
- **GitHub Issues:** [github.com/SanthoshSivasubramani/mumax3_neurospin/issues](https://github.com/SanthoshSivasubramani/mumax3_neurospin/issues)
- **Email:** ragansanthosh@ieee.org
- **Documentation:** See repository [docs/](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/docs) folder

---

## News & Updates

**Latest Release: v2.1.0 (February 3, 2026)**
- ✅ 13 new CPU implementations
- ✅ Full SPICE netlist parser  
- ✅ Spiking neural network (LIF) support
- ✅ Backpropagation through time (BPTT)
- ✅ Grain boundary effects with Voronoi tessellation
- ✅ Atomistic-continuum multiscale coupling
- ✅ All placeholder functions implemented
- ✅ 100% physics verified on Rocky Linux 9.5

[View full changelog →](/releases/)

---

**Status:** <span class="badge badge-success">Production Ready</span> | 
**Tested On:** Rocky Linux 9.5, NVIDIA A100, CUDA 11.8, Go 1.23.6
