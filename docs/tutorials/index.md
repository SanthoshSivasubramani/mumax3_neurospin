---
layout: default
title: Tutorials
---

# Tutorials

Step-by-step guides for common simulation tasks.

---

## Getting Started

### Installation & Setup

1. **Download binary** from [Releases](/releases/)
2. **Verify CUDA**: `nvidia-smi` and `nvcc --version`
3. **Make executable**: `chmod +x mumax3-saf-neurospin-v2.1.0`
4. **Test**: `./mumax3-saf-neurospin-v2.1.0 --version`

### First Simulation

**test.mx3:**
```go
SetGridSize(64, 64, 1)
SetCellSize(5e-9, 5e-9, 5e-9)

Msat = 800e3
Aex = 13e-12
alpha = 0.01

m = Uniform(1, 0, 0)
B_ext = Vector(0, 0, 0.1)

Run(1e-9)
```

**Run:**
```bash
./mumax3-saf-neurospin-v2.1.0 test.mx3
```

---

## Tutorial 1: SAF Structure Design

### Step 1: Define Geometry

```go
// Two-layer SAF
SetGridSize(128, 128, 2)
SetCellSize(5e-9, 5e-9, 1e-9)

DefRegion(1, Layer(0))  // Bottom FM
DefRegion(2, Layer(1))  // Top FM
```

### Step 2: Material Parameters

```go
// CoFeB
Msat = 1.2e6
Aex = 15e-12
alpha = 0.01
Ku1 = 0.8e6
anisU = Vector(0, 0, 1)
```

### Step 3: Enable RKKY

```go
EnableRKKYCoupling()
SetRKKYStrength(-0.5e-3)  // Antiferromagnetic
```

**Coupling Guidelines:**
- Ru (0.4 nm): J = -0.5 to -1.0 × 10⁻³ J/m²
- Ru (0.8 nm): J = +0.1 to +0.3 × 10⁻³ J/m²
- Ta: J = -0.1 to -0.3 × 10⁻³ J/m²

### Step 4: Initialize & Relax

```go
m.SetRegion(1, Uniform(1, 0, 0))
m.SetRegion(2, Uniform(-1, 0, 0))
Relax()
```

### Step 5: Field Sweep

```go
TableAdd(E_total)
TableAdd(E_RKKY)

for B := 0.0; B <= 0.5; B += 0.01 {
    B_ext = Vector(B, 0, 0)
    Relax()
    TableSave()
}
```

**Expected Results:**
- E_RKKY < 0 for antiparallel
- Spin-flop at H_sf = |J|/(μ₀ Ms t)
- Saturation at H_sat = 2|J|/(μ₀ Ms t)

---

## Tutorial 2: SOT Switching

### Setup

```go
SetGridSize(64, 64, 1)
SetCellSize(4e-9, 4e-9, 1e-9)

Msat = 800e3
Aex = 13e-12
alpha = 0.05
Ku1 = 0.4e6
anisU = Vector(0, 0, 1)
```

### Enable SOT

```go
EnableSOT()
SOT_theta_SH = 0.3
SOT_pol = Vector(0, 1, 0)
```

**Spin Hall Angles:**
- Pt: 0.07 - 0.12
- β-Ta: -0.12 to -0.15
- β-W: -0.3 to -0.4

### Symmetry-Breaking Field

```go
B_ext = Vector(0.02, 0, 0)  // Small in-plane field
```

### Current Pulse

```go
m = Uniform(0, 0, 1)
SOT_J = Vector(5e11, 0, 0)  // 0.5 TA/m²
Run(5e-9)
```

**Critical Current:**
```
J_c = (2e/ℏ) × (α Ms t / θ_SH) × H_K
```

---

## Tutorial 3: Reservoir Computing

### Design Reservoir

```go
SetGridSize(100, 100, 1)
SetCellSize(10e-9, 10e-9, 2e-9)

Msat = 600e3
Aex = 10e-12
alpha = 0.02
```

### Configure Parameters

```go
EnableReservoirComputing()
ReservoirSize = 100
ReservoirSpectralRadius = 0.95
ReservoirLeakRate = 0.3
ReservoirInputScale = 0.5
```

**Guidelines:**
- Spectral radius: 0.9-0.99 (higher = more memory)
- Leak rate: 0.1-0.5 (lower = more memory)
- Input scale: Task-dependent

### Input & Training

```go
// Generate input
for t := 0.0; t < 100e-9; t += 1e-12 {
    input := Sin(2*pi*1e9*t)
    ReservoirInput(input)
    Steps(1)
}

// Train readout
EnableDynamicReservoir()
ReservoirLearningRate = 0.01
AdaptReservoirWeights(target)
```

---

## Tutorial 4: Spiking Networks

### Network Setup

```go
EnableSNN()
SNN_neuronCount = 256
SNN_tau = 20e-3
SNN_Vth = 1.0
SNN_refractoryPeriod = 2e-3
```

### Topology

```go
EnableNetworkSimulation()
Network_topology = "small-world"
Network_connectivity = 0.1
CreateNetwork(256, "small-world")
```

### STDP Learning

```go
EnableStochasticSTDP()
STDP_Aplus = 0.01
STDP_Aminus = 0.012
STDP_tauPlus = 20e-3
```

### Training Loop

```go
for epoch := 0; epoch < 100; epoch++ {
    current := encodeInput(pattern)
    SetSNNCurrent(current)
    SimulateNetwork(100e-3)
    spikes := GetSNNSpikes()
}
```

---

## Tutorial 5: Grain Boundaries

### Generate Grains

```go
SetGridSize(512, 512, 1)
SetCellSize(2e-9, 2e-9, 10e-9)

EnableGrainBoundaries()
GrainCount = 200
GrainBoundaryReduction = 0.3
GrainAnisotropyVariation = 0.1

GenerateVoronoiGrains(GrainCount)
Save(GrainMap)
```

### Hysteresis Loop

```go
m = Uniform(1, 0, 0)

for B := -0.2; B <= 0.2; B += 0.005 {
    B_ext = Vector(B, 0, 0)
    Relax()
    TableSave()
}
```

**Expected:**
- Increased coercivity vs. single-crystal
- Broader switching field distribution
- Domain wall pinning at boundaries

---

## Best Practices

### Performance
- Use powers of 2 for grid size (FFT efficiency)
- Cell size < exchange length: l_ex = √(2A/(μ₀M²))
- Use adaptive stepping (RK45)
- Minimize autosave frequency

### Stability
- MaxErr: Start with 10⁻⁵, reduce if needed
- Damping: α > 0.001 for stability
- Time step: Let solver adapt automatically

### Physical Accuracy
- Thermal: Verify kT << K_u V
- Include demagnetization for accurate fields
- Use proper boundary conditions

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA error | Check GPU drivers, CUDA version |
| Divergence | Reduce MaxErr, increase α |
| Slow convergence | Increase α for Relax() |
| Memory error | Reduce grid size |

---

## Validation

```go
// Energy decreases during relaxation
TableAdd(E_total)
TableAutosave(1e-12)
Relax()

// Check |m| = 1
TableAdd(MaxAngle)
```

---

## Additional Resources

- **[API Reference](/api/)** - Complete function documentation
- **[Examples](/examples/)** - Working code samples
- **[GitHub](https://github.com/SanthoshSivasubramani/mumax3_neurospin)** - Source & community
- **Email:** ragansanthosh@ieee.org

---

[← Back to Home](/) | [Examples →](/examples/) | [API Reference →](/api/)
