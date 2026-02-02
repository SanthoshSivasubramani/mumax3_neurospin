# MuMax3-SAF-NeuroSpin v2.1.0

### Synthetic Antiferromagnet (SAF) Physics & Neuromorphic Device Modeling

**Author:** Dr. Santhosh Sivasubramani  
**Version:** v2.1.0  
**Release Date:** February 3, 2026

**Affiliations:**
1. INTRINSIC Lab, Centre for Sensors Instrumentation and Cyber Physical System Engineering (SeNSE), Indian Institute of Technology Delhi, New Delhi, India
2. April AI Hub, Centre for Electronic Frontiers, The University of Edinburgh, Edinburgh, United Kingdom

**Contact:** ssivasub@iitd.ac.in, ssivasub@ed.ac.uk, ragansanthosh@ieee.org  
**Repository:** https://github.com/SanthoshSivasubramani/mumax3-neurospin  
**License:** GNU GPL v3.0

---

## Overview

MuMax3-SAF-NeuroSpin is a comprehensive extension of [MuMax3](https://github.com/mumax/3) for simulating synthetic antiferromagnet (SAF) physics and neuromorphic device modeling. This release includes **50+ features** across three major versions (v1.0, v2.0, v2.1) enabling cutting-edge research in spintronics and neuromorphic computing.

---

## Key Features

### Core SAF Physics (v1.0):
- **RKKY Interlayer Coupling** - Basic, oscillatory, and temperature-dependent
- **Spin-Orbit Torque (SOT)** - Field-like and damping-like components
- **Spin-Transfer Torque (STT)** - Adiabatic and non-adiabatic terms
- **Voltage-Controlled Anisotropy (VCMA)** - Electric field control
- **Thermal Fluctuations** - cuRAND-based stochastic fields
- **Oersted Fields** - Current-induced magnetic fields
- **Topological Charge** - Skyrmion detection
- **Material & Spacer Databases** - 25 magnetic + 17 spacer materials

### Advanced Physics (v2.0):
- **Multi-Neighbor RKKY** - Beyond nearest-neighbor coupling
- **Interlayer DMI** - Interfacial Dzyaloshinskii-Moriya interaction
- **Quantum TMR** - Sub-nanometer tunneling magnetoresistance
- **Orange-Peel Coupling** - Interface roughness effects
- **Spin Wave Analysis** - FFT-based magnon dispersion
- **SHNO** - Spin Hall nano-oscillators
- **Heat Diffusion** - Thermal transport modeling
- **Magnon-Phonon Coupling** - Spin-lattice interactions

### Neuromorphic Computing (v2.1):
- **STDP Learning** - Spike-timing-dependent plasticity (basic, triplet, voltage-gated)
- **Analog Weight Programming** - Synaptic weight control
- **Reservoir Computing** - Echo state networks
- **Metaplasticity** - Learning-to-learn mechanisms
- **Neuromorphic Device Database** - Experimental parameters from literature
- **Skyrmion Energy Barriers** - Activation energy calculations
- **Energy Landscape Analysis** - Multi-dimensional optimization

---

## Quick Start

### Pre-compiled Binary (Fastest):
```bash
tar -xf mumax3-saf-neurospin-v2.1.0.tar
cd mumax3-saf-neurospin/mumax3-saf-neurospin/

# Fix permissions
chmod -R u+w .
chmod +x mumax3-saf-neurospin-v2.1.0

# Load CUDA
module load cuda/11.8

# Run example
./mumax3-saf-neurospin-v2.1.0 examples/v1_features/saf_basic_example.mx3
```

### Build from Source:
```bash
tar -xf mumax3-saf-neurospin-v2.1.0.tar
cd mumax3-saf-neurospin/mumax3-saf-neurospin/

# Fix permissions
chmod -R u+w .
chmod +x build_saf.sh

# Load dependencies
module load cuda/11.8
export PATH=/usr/local/go/bin:$PATH

# Build
./build_saf.sh

# Test
./mumax3-saf-neurospin-v2.1.0 tests/saf/test_saf_basic.mx3
```

**For detailed installation instructions**, see [INSTALL.md](INSTALL.md).

---

## System Requirements

- **OS:** Linux x86_64 (Ubuntu 20.04+, RHEL 8+, Rocky Linux 9.5)
- **GPU:** NVIDIA with Compute Capability 7.0+ (V100, A100, H100)
- **CUDA:** 11.0 to 12.1 (tested on 11.8)
- **Go:** 1.21+ (tested on 1.23.6)
- **RAM:** 8GB minimum, 16GB+ recommended

---

## Documentation

- **Installation Guide:** [INSTALL.md](INSTALL.md) - Comprehensive setup instructions
- **API Reference:** [docs/API_LIST.md](docs/API_LIST.md) - 97 functions documented
- **Examples:** [examples/](examples/) - 9 example scripts (v1 + v2 features)
- **Tests:** [tests/](tests/) - 65+ validation scripts
- **Release Notes:** See GitHub release for detailed changelog

---

## Usage Examples

### Basic SAF Coupling:
```javascript
// Enable SAF extension
EnableSAF()

// Set SAF layers (region 0 and region 1)
SetSAFLayers(0, 1)

// Set RKKY coupling strength (negative = antiferromagnetic)
J_RKKY = -2e-3

// Apply material presets
ApplyMaterial(0, "CoFeB")
ApplyMaterial(1, "NiFe")
ApplySpacerPreset(2, "Ru")

// Get SAF energy
E := SAFEnergy()
print("SAF Energy:", E)
```

### List Available Materials:
```javascript
// List magnetic materials (25 available)
ListMaterials()

// List spacer materials (17 available)
ListSpacers()

// List neuromorphic devices
ListNeuromorphicDevices()
```

### Advanced Features:
```javascript
// Temperature-dependent RKKY
SetTemperatureRKKY(300)  // 300K

// Spin-orbit torque
EnableSOT()
J_SOT = 1e11  // A/m²

// STDP learning
EnableSTDP()
STDP_A_plus = 0.01
STDP_tau_plus = 20e-3  // 20ms
```

---

## Testing

Run the comprehensive test suite:
```bash
cd tests/
./test_all_comprehensive_logging.sh
```

**Expected:** 65+ tests complete in ~15-30 minutes

Individual test categories:
```bash
# Core features
./mumax3-saf-neurospin-v2.1.0 tests/core/*.mx3

# SAF-specific
./mumax3-saf-neurospin-v2.1.0 tests/saf/*.mx3

# Neuromorphic features
./mumax3-saf-neurospin-v2.1.0 tests/advanced/*.mx3
```

---

## Citation

If you use MuMax3-SAF-NeuroSpin in your research, please cite:

```bibtex
@software{sivasubramani2026mumax3saf,
  author = {Sivasubramani, Santhosh},
  title = {MuMax3-SAF-NeuroSpin: Synthetic Antiferromagnet Physics and Neuromorphic Device Modeling},
  year = {2026},
  version = {v2.1.0},
  url = {https://github.com/SanthoshSivasubramani/mumax3-neurospin},
  note = {Extension of MuMax3 for SAF physics and neuromorphic computing}
}
```

**Author:** Dr. Santhosh Sivasubramani  
**Affiliations:** IIT Delhi (INTRINSIC Lab) & University of Edinburgh (April AI Hub)  
**Contact:** ssivasub@iitd.ac.in, ssivasub@ed.ac.uk, ragansanthosh@ieee.org

---

## Version Information

- **Release Version:** v2.1.0
- **Build Script Version:** v5.4
- **MuMax3 Core Version:** v3.11.1
- **Release Date:** February 3, 2026

---

## License

GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

---

## Support

- **Issues:** https://github.com/SanthoshSivasubramani/mumax3-neurospin/issues
- **Email:** ragansanthosh@ieee.org
- **Documentation:** See [docs/](docs/) directory

---

## Acknowledgments

- **MuMax3 Core:** Arne Vansteenkiste and contributors
- **NVIDIA:** CUDA toolkit and GPU computing platform
- **Go Community:** Go programming language
- **Research Groups:** IIT Delhi SeNSE, University of Edinburgh

---

**Status:** ✅ Production Ready  
**Tested On:** Rocky Linux 9.5, NVIDIA A100, CUDA 11.8, Go 1.23.6
