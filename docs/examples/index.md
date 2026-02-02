---
layout: default
title: Examples
---

# Examples

Working simulation scripts demonstrating all MuMax3-SAF-NeuroSpin features.

[View all examples on GitHub →](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples)

---

## Quick Start Examples

### 1. Basic SAF with RKKY Coupling

Antiferromagnetic coupling between two ferromagnetic layers.

```go
SetGridSize(128, 128, 2)
SetCellSize(5e-9, 5e-9, 1e-9)

// CoFeB parameters
Msat = 1.2e6
Aex = 15e-12
alpha = 0.01
Ku1 = 0.8e6
anisU = Vector(0, 0, 1)

// Define layers
DefRegion(1, Layer(0))
DefRegion(2, Layer(1))

// Enable RKKY
EnableRKKYCoupling()
SetRKKYStrength(-0.8e-3)

// Initial antiparallel state
m.SetRegion(1, Uniform(1, 0, 0))
m.SetRegion(2, Uniform(-1, 0, 0))

// Relax
Relax()
```

[Download example →](https://github.com/SanthoshSivasubramani/mumax3_neurospin/blob/main/examples/v1_features/saf_basic_example.mx3)

---

### 2. SOT-Driven Switching

Current-induced magnetization switching via spin-orbit torque.

```go
SetGridSize(64, 64, 1)
SetCellSize(4e-9, 4e-9, 1e-9)

Msat = 800e3
Aex = 13e-12
alpha = 0.05
Ku1 = 0.4e6
anisU = Vector(0, 0, 1)

// Enable SOT
EnableSOT()
SOT_theta_SH = 0.3
SOT_J = Vector(5e11, 0, 0)
SOT_pol = Vector(0, 1, 0)

// Symmetry-breaking field
B_ext = Vector(0.02, 0, 0)

m = Uniform(0, 0, 1)
Run(5e-9)
```

---

### 3. Reservoir Computing

Physical reservoir for time series prediction.

```go
SetGridSize(100, 100, 1)
SetCellSize(10e-9, 10e-9, 2e-9)

Msat = 600e3
Aex = 10e-12
alpha = 0.02

// Configure reservoir
EnableReservoirComputing()
ReservoirSize = 100
ReservoirSpectralRadius = 0.95
ReservoirLeakRate = 0.3

// Input signal
for t := 0.0; t < 100e-9; t += 1e-12 {
    input := Sin(2*pi*1e9*t)
    ReservoirInput(input)
    Steps(1)
}
```

---

### 4. Spiking Neural Network

LIF neurons with STDP learning.

```go
EnableSNN()
SNN_neuronCount = 256
SNN_tau = 20e-3
SNN_Vth = 1.0
SNN_refractoryPeriod = 2e-3

EnableNetworkSimulation()
Network_topology = "small-world"
Network_connectivity = 0.1

EnableStochasticSTDP()
STDP_Aplus = 0.01
STDP_tauPlus = 20e-3

CreateNetwork(256, "small-world")

// Training loop
for epoch := 0; epoch < 100; epoch++ {
    SetSNNCurrent(inputPattern)
    SimulateNetwork(100e-3)
}
```

---

### 5. Polycrystalline Grains

Voronoi grain structure with boundary effects.

```go
SetGridSize(512, 512, 1)
SetCellSize(2e-9, 2e-9, 10e-9)

Msat = 600e3
Aex = 10e-12

EnableGrainBoundaries()
GrainCount = 200
GrainBoundaryReduction = 0.3
GrainAnisotropyVariation = 0.1

GenerateVoronoiGrains(GrainCount)
Save(GrainMap)

// Hysteresis loop
m = Uniform(1, 0, 0)
for B := -0.2; B <= 0.2; B += 0.005 {
    B_ext = Vector(B, 0, 0)
    Relax()
    TableSave()
}
```

---

### 6. SPICE Co-Simulation

MTJ with read circuit.

```go
SetGridSize(32, 32, 3)
SetCellSize(5e-9, 5e-9, 1e-9)

// MTJ stack
DefRegion(1, Layer(0))  // Free layer
DefRegion(2, Layer(1))  // Barrier
DefRegion(3, Layer(2))  // Reference

Msat.SetRegion(2, 0)

EnableQuantumTMR()
QuantumTMR_P1 = 0.65
QuantumTMR_P2 = 0.65

EnableSPICECoSim()
SetSPICENetlist("mtj_circuit.cir")
CoupleSPICEtoRegion("MTJ", 1)

// Read operation
SetSPICEVoltage("VREAD", 0.3)
SPICEStep(10e-9)
```

---

## Example Categories

### SAF Physics
- [Basic RKKY](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v1_features)
- [Thermal stability](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v1_features)
- [VCMA-assisted switching](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v1_features)
- [Multi-layer SAF](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)

### Neuromorphic Computing
- [Reservoir computing](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)
- [Spiking networks](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)
- [BPTT training](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)
- [STDP learning](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v1_features)

### Multiscale
- [Grain boundaries](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)
- [Atomistic coupling](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)
- [Defect pinning](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)

### Industry
- [SPICE co-simulation](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)
- [Heat-assisted switching](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/examples/v2_features)

---

## Running Examples

```bash
# Clone repository
git clone https://github.com/SanthoshSivasubramani/mumax3_neurospin.git
cd mumax3_neurospin

# Run example
./mumax3-saf-neurospin-v2.1.0 examples/v1_features/saf_basic_example.mx3

# With custom output directory
./mumax3-saf-neurospin-v2.1.0 -o results/ examples/v2_features/reservoir_computing.mx3
```

---

## Output Files

| Extension | Description |
|-----------|-------------|
| `.out/table.txt` | Scalar quantities (time series) |
| `.out/m*.ovf` | Magnetization field snapshots |
| `.out/*.png` | Image snapshots |
| `.out/log.txt` | Simulation log |

---

[← Back to Home](/) | [API Reference →](/api/) | [Tutorials →](/tutorials/)
