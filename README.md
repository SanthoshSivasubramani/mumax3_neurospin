# MuMax3-NeuroSpin

### Synthetic Antiferromagnet (SAF) Physics & Neuromorphic Device Modeling

**Author:** Dr. Santhosh Sivasubramani
**Contact:** ragansanthosh@ieee.org
**Affiliation:** Centre for Sensors, Instrumentation and Cyber Physical System Engineering (SeNSE), IIT Delhi
**Repository:** [github.com/SanthoshSivasubramani/mumax3-neurospin](https://github.com/SanthoshSivasubramani/mumax3-neurospin)  

This fork extends [MuMax3](https://github.com/mumax/3) with:
- RKKY coupling and oscillatory SAF interactions  
- Spin-orbit and spin-transfer torque kernels  
- Voltage-controlled magnetic anisotropy (VCMA)  
- Thermal stochastic fields  
- Neuromorphic primitives: STDP and analog weight programming  
- Region-wise magnetization accumulation for accurate SAF averages  

**License:** GNU GPL v3  

MuMax3 3.11.1 with Spintronic Activation Functions & Modelling.

## Build

**Quick Start:**
```bash
# Load CUDA (or set CUDA_HOME)
module load cuda/11.8

# Build
./build_saf.sh
```

**For detailed installation instructions**, including system requirements, dependencies, and troubleshooting, see [INSTALL.md](INSTALL.md).

## Usage
```bash
export PATH=$HOME/go/bin:$PATH
mumax3 your_script.mx3
```

## Citation

Author: Dr. Santhosh Sivasubramani
Contact: ragansanthosh@ieee.org
Affiliation: Centre for Sensors, Instrumentation and Cyber Physical System Engineering (SeNSE), IIT Delhi

## Download and Use

### Pre-compiled Binary

Download the latest compiled binary from [GitHub Actions](https://github.com/SanthoshSivasubramani/mumax3-neurospin/actions):

1. Go to Actions tab
2. Click on latest successful workflow run
3. Download `mumax3-neurospin` artifact
4. Extract and run:
```bash
unzip mumax3-neurospin.zip
chmod +x mumax3
./mumax3 your_script.mx3
```

### SAF Extension Features

This build includes:
- ✅ RKKY interlayer coupling
- ✅ Thermal fluctuations (cuRAND)
- ✅ SOT, STT, VCMA, Oersted fields
- ✅ Material database (25 magnetic + 17 spacer materials)
- ✅ Neuromorphic device modeling

### Example Test
```bash
./mumax3 tests/saf/test_saf_basic.mx3
```

Expected output shows SAF banner and "PASS" for tests.

