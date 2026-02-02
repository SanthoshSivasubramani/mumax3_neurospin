---
layout: default
title: API Reference
---

# API Reference

Complete documentation for all MuMax3-SAF-NeuroSpin functions, parameters, and physics implementations.

**Current Version:** v2.1.0 | **Total Functions:** 97+ | **Last Updated:** February 3, 2026

---

## Version-Specific Documentation

<table>
<tr>
<th>Version</th>
<th>Features</th>
<th>Implementation</th>
<th>Documentation</th>
</tr>
<tr>
<td><strong>V1.0</strong></td>
<td>SAF Physics Fundamentals</td>
<td>12 GPU Kernels</td>
<td><a href="v1.html">View V1.0 API →</a></td>
</tr>
<tr>
<td><strong>V2.0</strong></td>
<td>Advanced Spintronics & Neuromorphic</td>
<td>25 GPU Kernels</td>
<td><a href="v2.html">View V2.0 API →</a></td>
</tr>
<tr>
<td><strong>V2.1</strong></td>
<td>Enhanced Solvers & Multiscale</td>
<td>13 CPU Implementations</td>
<td><a href="v21.html">View V2.1 API →</a></td>
</tr>
</table>

---

## Global Parameters

### Simulation Control

| Parameter | Type | Unit | Description |
|-----------|------|------|-------------|
| `SetGridSize(Nx, Ny, Nz)` | int, int, int | - | Number of cells in each dimension |
| `SetCellSize(dx, dy, dz)` | float, float, float | m | Cell dimensions |
| `SetPBC(x, y, z)` | int, int, int | - | Periodic boundary conditions (0=off, N=repeat) |

### Material Parameters

| Parameter | Type | Unit | Description |
|-----------|------|------|-------------|
| `Msat` | float | A/m | Saturation magnetization |
| `Aex` | float | J/m | Exchange stiffness |
| `alpha` | float | - | Gilbert damping parameter |
| `Ku1` | float | J/m³ | First-order uniaxial anisotropy |
| `Ku2` | float | J/m³ | Second-order uniaxial anisotropy |
| `Kc1` | float | J/m³ | First-order cubic anisotropy |
| `anisU` | vector | - | Uniaxial anisotropy axis |
| `anisC1` | vector | - | First cubic anisotropy axis |
| `anisC2` | vector | - | Second cubic anisotropy axis |

### Output Control

| Function | Description |
|----------|-------------|
| `TableAdd(quantity)` | Add quantity to output table |
| `TableSave()` | Save current table values |
| `TableAutosave(period)` | Automatic table saving interval |
| `Save(quantity)` | Save spatial data to OVF file |
| `AutoSave(quantity, period)` | Automatic spatial data saving |
| `Snapshot(quantity)` | Save PNG image |
| `SnapshotAs(quantity, filename)` | Save PNG with custom name |

---

## Common Quantities

### Magnetization

| Quantity | Unit | Description |
|----------|------|-------------|
| `m` | - | Normalized magnetization (mx, my, mz) |
| `m.Region(i)` | - | Magnetization in region i |
| `m_full` | A/m | Full magnetization (Ms × m) |

### Fields

| Quantity | Unit | Description |
|----------|------|-------------|
| `B_eff` | T | Total effective field |
| `B_demag` | T | Demagnetizing field |
| `B_exch` | T | Exchange field |
| `B_anis` | T | Anisotropy field |
| `B_ext` | T | External applied field |
| `B_therm` | T | Thermal fluctuation field |

### Energy

| Quantity | Unit | Description |
|----------|------|-------------|
| `E_total` | J | Total magnetic energy |
| `E_demag` | J | Demagnetization energy |
| `E_exch` | J | Exchange energy |
| `E_anis` | J | Anisotropy energy |
| `E_Zeeman` | J | Zeeman energy |
| `Edens_total` | J/m³ | Total energy density |

### Dynamics

| Quantity | Unit | Description |
|----------|------|-------------|
| `torque` | T | Total torque on magnetization |
| `MaxTorque` | T | Maximum torque (convergence criterion) |
| `dt` | s | Current time step |
| `t` | s | Simulation time |

---

## Simulation Functions

### Time Evolution

```go
// Run for specified duration
Run(duration)  // duration in seconds

// Run until condition
RunWhile(condition)

// Relax to equilibrium
Relax()

// Minimize energy
Minimize()

// Single step
Steps(n)
```

### Solvers

```go
// Set ODE solver
SetSolver(solverType)
// Types: 1 (Euler), 2 (Heun), 3 (RK23), 4 (RK4), 5 (RK45), 6 (DormandPrince)

// Adaptive time stepping
MaxDt = 1e-12  // Maximum time step
MinDt = 1e-15  // Minimum time step
MaxErr = 1e-5  // Maximum error per step
```

### Regions

```go
// Define region
DefRegion(id, shape)

// Region-specific parameters
Msat.SetRegion(id, value)
Aex.SetRegion(id, value)
alpha.SetRegion(id, value)

// Inter-region exchange
ext_ScaleExchange(region1, region2, scale)
```

---

## Geometry Functions

### Basic Shapes

```go
// 3D shapes
Cuboid(wx, wy, wz)           // Rectangular prism
Cylinder(d, h)               // Cylinder along z
Ellipsoid(dx, dy, dz)        // Ellipsoid
Cone(d, h)                   // Cone along z
Cell(ix, iy, iz)             // Single cell

// 2D shapes (extruded along z)
Rect(wx, wy)                 // Rectangle
Circle(d)                    // Circle
Ellipse(dx, dy)              // Ellipse
Triangle(x1,y1, x2,y2, x3,y3) // Triangle

// Boolean operations
shape1.Add(shape2)           // Union
shape1.Sub(shape2)           // Subtraction
shape1.Intersect(shape2)     // Intersection
shape1.Xor(shape2)           // Exclusive or
shape.Inverse()              // Complement
```

### Transformations

```go
shape.Transl(dx, dy, dz)     // Translate
shape.Scale(sx, sy, sz)      // Scale
shape.RotX(angle)            // Rotate around X (radians)
shape.RotY(angle)            // Rotate around Y
shape.RotZ(angle)            // Rotate around Z
shape.Repeat(nx, ny, nz)     // Periodic repetition
```

---

## Initial Magnetization States

```go
// Uniform states
m = Uniform(mx, my, mz)

// Vortex
m = Vortex(circulation, polarity)
// circulation: +1 (CCW) or -1 (CW)
// polarity: +1 (up) or -1 (down)

// Random
m = RandomMag()
m = RandomMagSeed(seed)

// From file
m.LoadFile(filename)

// Custom function
m = VectorField(func(x, y, z float64) (float64, float64, float64) {
    return mx, my, mz
})

// Two-layer SAF state
m = TwoLayer(mx1, my1, mz1, mx2, my2, mz2)
```

---

## Units Convention

All parameters use SI units:

| Quantity | Unit | Example |
|----------|------|---------|
| Length | meters (m) | `SetCellSize(5e-9, 5e-9, 1e-9)` |
| Time | seconds (s) | `Run(1e-9)` |
| Magnetic field | Tesla (T) | `B_ext = Vector(0, 0, 0.1)` |
| Magnetization | A/m | `Msat = 1.2e6` |
| Exchange | J/m | `Aex = 15e-12` |
| Anisotropy | J/m³ | `Ku1 = 0.8e6` |
| Energy | Joules (J) | Output: `E_total` |
| Current density | A/m² | `J = Vector(1e12, 0, 0)` |
| Temperature | Kelvin (K) | `Temp = 300` |

---

## Feature Matrix by Version

| Feature | V1.0 | V2.0 | V2.1 |
|---------|------|------|------|
| **SAF Physics** |
| RKKY Coupling | ✅ | ✅ | ✅ |
| SOT/STT | ✅ | ✅ | ✅ |
| VCMA | ✅ | ✅ | ✅ |
| Thermal | ✅ | ✅ | ✅ |
| Multi-neighbor RKKY | ❌ | ✅ | ✅ |
| Interlayer DMI | ❌ | ✅ | ✅ |
| Temperature-dependent RKKY | ❌ | ✅ | ✅ |
| Orange-Peel Coupling | ❌ | ❌ | ✅ |
| **Neuromorphic** |
| STDP | ✅ | ✅ | ✅ |
| Reservoir Computing | ❌ | ✅ | ✅ |
| Metaplasticity | ❌ | ✅ | ✅ |
| SNN (LIF) | ❌ | ❌ | ✅ |
| BPTT | ❌ | ❌ | ✅ |
| **Multiscale** |
| Grain Boundaries | ❌ | ❌ | ✅ |
| Atomistic Coupling | ❌ | ❌ | ✅ |
| Adaptive Oversampling | ❌ | ❌ | ✅ |
| Dynamic Defects | ❌ | ❌ | ✅ |
| **Industry** |
| SPICE Co-simulation | ❌ | ❌ | ✅ |
| Heat Diffusion | ❌ | ✅ | ✅ |

---

## Quick Navigation

### By Physics Domain
- [SAF Physics (V1.0)](v1.html#rkky-interlayer-coupling)
- [Advanced Spintronics (V2.0)](v2.html#multi-neighbor-rkky)
- [Neuromorphic Computing (V2.0)](v2.html#reservoir-computing)
- [Multiscale Modeling (V2.1)](v21.html#grain-boundary-effects)
- [Industry Applications (V2.1)](v21.html#spice-co-simulation)

### By Feature Type
- [Coupling Mechanisms](v1.html#rkky-interlayer-coupling)
- [Spin Torques](v1.html#spin-orbit-torque-sot)
- [Thermal Effects](v1.html#thermal-fluctuations)
- [Learning Rules](v1.html#stdp-learning)
- [Solvers & Algorithms](v21.html#extended-spin-diffusion)

---

## Download Documentation

For offline reference, download the complete API documentation:

- [**PDF Documentation**](https://github.com/SanthoshSivasubramani/mumax3_neurospin/releases/download/v2.1.0/API_Documentation.pdf) (All versions)
- [**Markdown Files**](https://github.com/SanthoshSivasubramani/mumax3_neurospin/tree/main/docs) (GitHub repository)

---

## Contributing

Found an error or want to improve the documentation?

1. Visit our [GitHub repository](https://github.com/SanthoshSivasubramani/mumax3_neurospin)
2. Open an [issue](https://github.com/SanthoshSivasubramani/mumax3_neurospin/issues) or submit a pull request
3. Email corrections to: ragansanthosh@ieee.org

---

[← Back to Home](/) | [Examples →](/examples/) | [Tutorials →](/tutorials/)
