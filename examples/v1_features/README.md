# MuMax3-SAF-NeuroSpin V1 Feature Examples

**Author:** Dr. Santhosh Sivasubramani  
**Affiliations:**  
1. INTRINSIC Lab, Centre for Sensors Instrumentation and Cyber Physical System Engineering (SeNSE), IIT Delhi, India  
2. April AI Hub, Centre for Electronic Frontiers, The University of Edinburgh, United Kingdom  
**Contact:** ssivasub@iitd.ac.in, ssivasub@ed.ac.uk, ragansanthosh@ieee.org  
**Repository:** [github.com/SanthoshSivasubramani/mumax3-neurospin](https://github.com/SanthoshSivasubramani/mumax3-neurospin)

---

## Overview

This directory contains **4 basic example scripts** demonstrating V1.0.0 features of MuMax3-SAF-NeuroSpin:
- Basic SAF RKKY coupling
- Energy landscape analysis
- Field sweep simulations
- Simple SAF configurations

These examples use the **18 core features** available in v1.0.0, providing foundational tests for synthetic antiferromagnetic structures.

---

## Quick Start

```bash
# Run any example:
mumax3 saf_basic_example.mx3

# All examples save output to current directory
```

---

## Example Scripts

### saf_basic_example.mx3
**Purpose:** Basic SAF RKKY field pairing test  
**Features:** SetSAFLayers, EnableSAF, J_RKKY, SAFEnergy  
**Runtime:** ~1 minute  
**Output:** Validates antiferromagnetic coupling sign convention

### saf_energy_example.mx3
**Purpose:** Energy landscape analysis  
**Features:** SAF energy calculation across different states  
**Runtime:** ~1 minute  
**Output:** Energy vs. configuration plots

### saf_field_sweep_example.mx3
**Purpose:** Field-dependent SAF behavior  
**Features:** External field sweep with RKKY coupling  
**Runtime:** ~2 minutes  
**Output:** Magnetization switching curves

### saf_simple_example.mx3
**Purpose:** Minimal SAF setup  
**Features:** Simplest working SAF configuration  
**Runtime:** <1 minute  
**Output:** Basic validation test

---

## V1.0.0 Features Used

1. **SetSAFLayers(region1, region2)** - Define SAF pair
2. **EnableSAF()** - Activate RKKY coupling
3. **J_RKKY** - Set coupling strength (J/m²)
4. **SAFEnergy()** - Query RKKY energy
5. **ApplyMaterial()** - Material assignment

---

## Next Steps

- **V2 Features:** See `../v2_features/` for advanced examples (50+ features)
- **Tests:** See `../../tests/` for comprehensive validation suite
- **Documentation:** See root README for full feature list

---

## Support

- **Issues:** [GitHub Issues](https://github.com/SanthoshSivasubramani/Mumax3_SAF_Neurospin/issues)
- **Email:** ragansanthosh@ieee.org
