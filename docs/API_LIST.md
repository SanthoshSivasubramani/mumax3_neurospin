# MuMax3-SAF Script API Reference

This document lists all available script commands found in the source code of `mumax3-saf-neurospin-v2.1.0`.

## V2 Features (Latest)
| Feature | Script Command | Go Implementation | Source File |
| :--- | :--- | :--- | :--- |
| **Multi-Neighbor RKKY** | `EnableMultiNeighborRKKY` | `EnableMultiNeighborRKKY` | `engine/saf_v2_physics.go` |
| | `SetRKKYNeighbors` | `SetRKKYNeighbors` | `engine/saf_v2_physics.go` |
| | `SetRKKYCouplings` | `SetRKKYCouplings` | `engine/saf_v2_physics.go` |
| **Interlayer DMI** | `EnableInterlayerDMI` | `EnableInterlayerDMI` | `engine/saf_v2_physics.go` |
| | `SetInterlayerDMI` | `SetInterlayerDMI` | `engine/saf_v2_physics.go` |
| **Non-Collinear RKKY** | `EnableNonCollinearRKKY` | `EnableNonCollinearRKKY` | `engine/saf_v2_physics.go` |
| | `SetNonCollinearRKKY` | `SetNonCollinearRKKY` | `engine/saf_v2_physics.go` |
| **Spin Diffusion** | `EnableSpinDiffusion` | `EnableSpinDiffusion` | `engine/saf_v2_physics.go` |
| | `SetSpinDiffusionLength` | `SetSpinDiffusionLength` | `engine/saf_v2_physics.go` |
| | `GetSpinAccumulation` | `GetSpinAccumulation` | `engine/saf_v2_physics.go` |
| **Stochastic STDP** | `EnableStochasticSTDP` | `EnableStochasticSTDP` | `engine/saf_v2_physics.go` |
| | `ApplyStochasticSTDP` | `ApplyStochasticSTDP` | `engine/saf_v2_physics.go` |
| **Reservoir Computing** | `EnableReservoirComputing` | `EnableReservoirComputing` | `engine/saf_v2_physics.go` |
| | `UpdateReservoirState` | `UpdateReservoirState` | `engine/saf_v2_physics.go` |
| **Metaplasticity** | `EnableMetaplasticity` | `EnableMetaplasticity` | `engine/saf_v2_physics.go` |
| | `ApplyMetaplasticity` | `ApplyMetaplasticity` | `engine/saf_v2_physics.go` |
| **Heat Diffusion** | `EnableHeatDiffusion` | `EnableHeatDiffusion` | `engine/saf_v2_physics.go` |
| | `SolveHeatDiffusion` | `SolveHeatDiffusion` | `engine/saf_v2_physics.go` |
| | `GetTemperatureField` | `GetTemperatureField` | `engine/saf_v2_physics.go` |
| **Nonlinear VCMA** | `EnableNonlinearVCMA` | `EnableNonlinearVCMA` | `engine/saf_v2_physics.go` |
| **Magnon-Phonon** | `EnableMagnonPhononCoupling` | `EnableMagnonPhononCoupling` | `engine/saf_v2_physics.go` |
| **Quantum TMR** | `EnableQuantumTunneling` | `EnableQuantumTunneling` | `engine/saf_v2_physics.go` |
| | `CalculateQuantumTMR` | `CalculateQuantumTMR` | `engine/saf_v2_physics.go` |
| **Orange-Peel** | `EnableOrangePeel` | `EnableOrangePeel` | `engine/saf_v2_physics.go` |
| **Temp-Dep RKKY** | `EnableTemperatureDependentRKKY` | `EnableTemperatureDependentRKKY` | `engine/saf_v2_physics.go` |
| **Spin Wave FFT** | `EnableSpinWaveFFT` | `EnableSpinWaveFFT` | `engine/saf_v2_physics.go` |
| | `ComputeSpinWaveFFT` | `ComputeSpinWaveFFT` | `engine/saf_v2_physics.go` |
| **SHNO** | `EnableSHNO` | `EnableSHNO` | `engine/saf_v2_physics.go` |
| | `UpdateSHNO` | `UpdateSHNO` | `engine/saf_v2_physics.go` |
| **Exchange Bias** | `EnableExchangeBias` | `EnableExchangeBias` | `engine/saf_v2_physics.go` |
| **Voltage RKKY** | `EnableVoltageRKKY` | `EnableVoltageRKKY` | `engine/saf_v2_physics.go` |
| **Atomistic-Continuum** | `EnableAtomisticContinuum` | `EnableAtomisticContinuum` | `engine/saf_v2_physics.go` |
| | `CoupleAtomisticToContinuum` | `CoupleAtomisticToContinuum` | `engine/saf_v2_physics.go` |
| **LL-Bloch** | `EnableLLBloch` | `EnableLLBloch` | `engine/saf_v2_physics.go` |
| | `UpdateLLBloch` | `UpdateLLBloch` | `engine/saf_v2_physics.go` |
| **Dipolar Skyrmions** | `EnableDipolarSkyrmions` | `EnableDipolarSkyrmions` | `engine/saf_v2_physics.go` |
| **Synaptic Homeostasis** | `EnableHomeostasis` | `EnableHomeostasis` | `engine/saf_v2_physics.go` |
| | `ApplyHomeostasis` | `ApplyHomeostasis` | `engine/saf_v2_physics.go` |
| **Dendritic Comp** | `EnableDendritic` | `EnableDendritic` | `engine/saf_v2_physics.go` |
| | `ComputeDendriticOutput` | `ComputeDendriticOutput` | `engine/saf_v2_physics.go` |
| **Winner-Take-All** | `EnableWTA` | `EnableWTA` | `engine/saf_v2_physics.go` |
| | `UpdateWTA` | `UpdateWTA` | `engine/saf_v2_physics.go` |
| **Topological Hall** | `EnableTopologicalHall` | `EnableTopologicalHall` | `engine/saf_v2_physics.go` |
| | `CalculateTopologicalHall` | `CalculateTopologicalHall` | `engine/saf_v2_physics.go` |
| **Spin Pumping** | `EnableSpinPumping` | `EnableSpinPumping` | `engine/saf_v2_physics.go` |
| | `CalculateSpinPumping` | `CalculateSpinPumping` | `engine/saf_v2_physics.go` |

## Core SAF & Materials
| Feature | Script Command | Go Implementation | Source File |
| :--- | :--- | :--- | :--- |
| **SAF Energy** | `SAFEnergy` | `SAFEnergy` | `engine/saf_extension.go` |
| **Materials** | `ApplyMaterialPreset` | `ApplyMaterialPreset` | `engine/saf_extension.go` |
| | `ListMaterials` | `ListMaterials` | `engine/saf_extension.go` |
| | `ApplySpacerPreset` | `ApplySpacerPreset` | `engine/saf_extension.go` |
| | `ListSpacers` | `ListSpacers` | `engine/saf_extension.go` |
| **Oersted Field** | `AddOerstedField` | `AddOerstedField` | `engine/saf_extension.go` |
| **Neuromorphic** | `ApplySTDP` | `ApplySTDP` | `engine/saf_extension.go` |
| | `ProgramSynapticWeights` | `ProgramSynapticWeights` | `engine/saf_extension.go` |

## Shapes & Geometry
| Feature | Script Command | Source File |
| :--- | :--- | :--- |
| `Ellipsoid` | `engine/shape.go` |
| `Superball` | `engine/shape.go` |
| `Ellipse` | `engine/shape.go` |
| `Cone` | `engine/shape.go` |
| `Cylinder` | `engine/shape.go` |
| `Circle` | `engine/shape.go` |
| `Cuboid` | `engine/shape.go` |
| `Rect` | `engine/shape.go` |
| `Square` | `engine/shape.go` |
| `Triangle` | `engine/shape.go` |
| `XRange`, `YRange`, `ZRange` | `engine/shape.go` |
| `Layers`, `Layer`, `Cell` | `engine/shape.go` |
| `Universe` | `engine/shape.go` |
| `GrainRoughness` | `engine/shape.go` |

## Utilities
| Feature | Script Command | Source File |
| :--- | :--- | :--- |
| `Expect`, `ExpectV` (Testing) | `engine/util.go` |
| `Sign` | `engine/util.go` |
| `Vector` | `engine/util.go` |
| `NewSlice` | `engine/util.go` |
| `ThermSeed` | `engine/temperature.go` |
| `Shift` | `engine/shift.go` |
| `TableAdd`, `TableSave` | `engine/table.go` |
| `Save`, `SaveAs`, `Snapshot` | `engine/save.go` |

