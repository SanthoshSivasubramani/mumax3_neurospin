// Copyright © 2025-2026 Prof. Santhosh Sivasubramani
//
// Affiliation:
// INTRINSIC Lab, Centre for Sensors Instrumentation and
// Cyber Physical System Engineering (SeNSE)
// Indian Institute of Technology Delhi, New Delhi, India
//
// Contact: ssivasub@iitd.ac.in, ragansanthosh@ieee.org
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

package engine

// MuMax3-SAF-NeuroSpin: Advanced Physics Extensions
//
// This module implements 25 advanced physics features for spintronics and neuromorphic computing:
//
// Magnetic Interactions (9 features):
//   - Multi-neighbor RKKY coupling (z±1, z±2, z±3, lateral)
//   - Interlayer Dzyaloshinskii-Moriya Interaction (DMI)
//   - Non-collinear RKKY with anisotropy control
//   - Dipolar skyrmion interactions
//   - Exchange bias (AFM/FM interfaces)
//   - Orange-peel coupling (Néel coupling from roughness)
//   - Temperature-dependent RKKY
//   - Voltage-controlled RKKY
//   - Magnon-phonon coupling
//
// Transport & Thermal (4 features):
//   - Valet-Fert spin diffusion solver
//   - Heat diffusion with Joule heating
//   - Quantum tunneling (TMR calculations)
//   - Spin-charge pumping
//
// High-Temperature Dynamics (2 features):
//   - Landau-Lifshitz-Bloch equations
//   - Spin Hall nano-oscillators (SHNO)
//
// Neuromorphic Computing (7 features):
//   - Stochastic STDP with retention decay
//   - Reservoir computing (Echo State Networks)
//   - Metaplasticity (BCM rule)
//   - Synaptic homeostasis
//   - Dendritic computation
//   - Winner-Take-All networks
//   - Multi-scale atomistic-continuum coupling
//
// Analysis Tools (3 features):
//   - Spin wave spectroscopy (FFT)
//   - Topological Hall effect
//   - Nonlinear VCMA (E², E³ terms)
//
// =====================================================================================================

import (
	"unsafe"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// ============================================================================
// V2 PARAMETERS (18 new parameters)
// ============================================================================

var (
	// Multi-neighbor RKKY
	rkky_neighbors = NewScalarParam("rkky_neighbors", "", "Number of RKKY neighbors (1-10)")
	rkky_J1        = NewScalarParam("rkky_J1", "J/m²", "RKKY coupling z±1 (first neighbor)")
	rkky_J2        = NewScalarParam("rkky_J2", "J/m²", "RKKY coupling z±2 (second neighbor)")
	rkky_J3        = NewScalarParam("rkky_J3", "J/m²", "RKKY coupling z±3 (third neighbor)")
	rkky_J_lateral = NewScalarParam("rkky_J_lateral", "J/m²", "Lateral RKKY coupling (x±1,y±1)")

	// Interlayer DMI
	D_interlayer = NewScalarParam("D_interlayer", "J/m²", "Interlayer DMI strength")

	// Non-collinear RKKY
	J_noncollinear     = NewScalarParam("J_noncollinear", "J/m²", "Non-collinear RKKY strength")
	alpha_noncollinear = NewScalarParam("alpha_noncollinear", "", "Non-collinear anisotropy (0=iso, 1=Heis, 2=XY)")

	// Spin diffusion
	lambda_sf_v2         = NewScalarParam("lambda_sf_v2", "m", "Spin diffusion length")
	spinDiffusionMaxIter = NewScalarParam("spinDiffusionMaxIter", "", "Max iterations for spin diffusion solver")

	// Stochastic STDP
	stdp_noise_sigma   = NewScalarParam("stdp_noise_sigma", "", "STDP write noise (σ/mean)")
	stdp_retention_tau = NewScalarParam("stdp_retention_tau", "s", "STDP retention time constant")

	// Reservoir computing
	reservoir_leak_rate       = NewScalarParam("reservoir_leak_rate", "", "Reservoir leak rate")
	reservoir_spectral_radius = NewScalarParam("reservoir_spectral_radius", "", "Reservoir spectral radius")

	// Metaplasticity
	meta_theta_target = NewScalarParam("meta_theta_target", "", "Metaplasticity target activity")
	meta_tau          = NewScalarParam("meta_tau", "s", "Metaplasticity time constant")

	// Heat diffusion
	kappa_thermal = NewScalarParam("kappa_thermal", "W/(m·K)", "Thermal conductivity")
	rho_cp        = NewScalarParam("rho_cp", "J/(m³·K)", "Volumetric heat capacity")

	// Nonlinear VCMA
	xi_vcma2 = NewScalarParam("xi_vcma2", "J/(V²·m)", "VCMA quadratic coefficient")
	xi_vcma3 = NewScalarParam("xi_vcma3", "J/(V³·m)", "VCMA cubic coefficient")

	// Orange-peel
	F_rms_roughness = NewScalarParam("F_rms_roughness", "m", "RMS roughness amplitude")
	xi_correlation  = NewScalarParam("xi_correlation", "m", "Roughness correlation length")

	// Magnon-phonon
	B1_magnetoelastic = NewScalarParam("B1_magnetoelastic", "J/m³", "Magnetoelastic constant B₁")
	B2_magnetoelastic = NewScalarParam("B2_magnetoelastic", "J/m³", "Magnetoelastic constant B₂")

	// Quantum tunneling
	barrier_thickness = NewScalarParam("barrier_thickness", "m", "Tunnel barrier thickness")
	barrier_height    = NewScalarParam("barrier_height", "eV", "Tunnel barrier height")
	polarization_1    = NewScalarParam("polarization_1", "", "Spin polarization layer 1")
	polarization_2    = NewScalarParam("polarization_2", "", "Spin polarization layer 2")

	// ============================================================================
	// V2 FEATURES 13-25 PARAMETERS (Week 3-4)
	// ============================================================================

	// Temperature-dependent RKKY (Feature 13)
	rkky_J0_base  = NewScalarParam("rkky_J0_base", "J/m²", "RKKY J₀ at T=0")
	rkky_T_curie  = NewScalarParam("rkky_T_curie", "K", "Curie temperature")
	rkky_temp_exp = NewScalarParam("rkky_temp_exp", "", "Temperature exponent α")

	// Spin wave FFT (Feature 14)
	sw_fft_max_freq = NewScalarParam("sw_fft_max_freq", "Hz", "Max frequency for FFT")
	sw_fft_bins     = NewScalarParam("sw_fft_bins", "", "Number of FFT frequency bins")

	// SHNO (Feature 15)
	shno_J_SHE    = NewScalarParam("shno_J_SHE", "A/m²", "Spin Hall current density")
	shno_theta_SH = NewScalarParam("shno_theta_SH", "", "Spin Hall angle")

	// Exchange bias (Feature 16)
	J_exbias   = NewScalarParam("J_exbias", "J/m²", "Exchange bias coupling")
	H_exbias_x = NewScalarParam("H_exbias_x", "A/m", "Exchange bias field x")
	H_exbias_y = NewScalarParam("H_exbias_y", "A/m", "Exchange bias field y")
	H_exbias_z = NewScalarParam("H_exbias_z", "A/m", "Exchange bias field z")

	// Voltage-controlled RKKY (Feature 17)
	rkky_voltage_coeff = NewScalarParam("rkky_voltage_coeff", "J/(V·m²)", "RKKY voltage coefficient")

	// Atomistic-continuum (Feature 18)
	atomistic_a    = NewScalarParam("atomistic_a", "m", "Atomic lattice constant")
	atomistic_J_ex = NewScalarParam("atomistic_J_ex", "J", "Atomic exchange J")

	// LL-Bloch (Feature 19)
	llbloch_T_curie = NewScalarParam("llbloch_T_curie", "K", "Curie temperature")
	llbloch_lambda  = NewScalarParam("llbloch_lambda", "", "Longitudinal relaxation")

	// Dipolar skyrmions (Feature 20)
	dipolar_cutoff = NewScalarParam("dipolar_cutoff", "m", "Dipolar interaction cutoff")

	// Synaptic homeostasis (Feature 21)
	homeo_target = NewScalarParam("homeo_target", "", "Target activity rate")
	homeo_tau    = NewScalarParam("homeo_tau", "s", "Homeostasis time constant")

	// Dendritic computation (Feature 22)
	dendrite_threshold = NewScalarParam("dendrite_threshold", "", "Dendritic threshold")
	dendrite_nonlin    = NewScalarParam("dendrite_nonlin", "", "Nonlinearity exponent")

	// Winner-Take-All (Feature 23)
	wta_inhibition = NewScalarParam("wta_inhibition", "", "Lateral inhibition strength")
	wta_tau        = NewScalarParam("wta_tau", "s", "WTA time constant")

	// Topological Hall (Feature 24)
	the_rho_n = NewScalarParam("the_rho_n", "1/m³", "Carrier density")

	// Spin pumping (Feature 25)
	spinpump_g_eff    = NewScalarParam("spinpump_g_eff", "1/m²", "Effective spin mixing conductance")
	spinpump_lambda_N = NewScalarParam("spinpump_lambda_N", "m", "Spin diffusion length in N")
)

// ============================================================================
// GLOBAL STATE
// ============================================================================

var (
	// Features 1-12
	multiNeighborRKKYEnabled  = false
	interlayerDMIEnabled      = false
	nonCollinearRKKYEnabled   = false
	spinDiffusionEnabled      = false
	stochasticSTDPEnabled     = false
	reservoirComputingEnabled = false
	metaplasticityEnabled     = false
	heatDiffusionEnabled      = false
	nonlinearVCMAEnabled      = false
	orangePeelEnabled         = false
	magnonPhononEnabled       = false
	quantumTunnelingEnabled   = false

	// Features 13-25
	temperatureRKKYEnabled    = false
	spinWaveFFTEnabled        = false
	shnoEnabled               = false
	exchangeBiasEnabled       = false
	voltageRKKYEnabled        = false
	atomisticContinuumEnabled = false
	llBlochEnabled            = false
	dipolarSkyrmiansEnabled   = false
	homeostasisEnabled        = false
	dendriticEnabled          = false
	wtaEnabled                = false
	topologicalHallEnabled    = false
	spinPumpingEnabled        = false

	// State fields
	mu_s_field           *data.Slice // Spin accumulation field
	temperature_field_v2 *data.Slice // Temperature field
	strain_field         *data.Slice // Strain field
	reservoir_state      *data.Slice // Reservoir state
	reservoir_weights    *data.Slice // Reservoir weight matrix (N × N_inputs)
	meta_threshold       *data.Slice // Metaplasticity threshold
	sw_fft_spectrum      *data.Slice // Spin wave FFT spectrum
	shno_state           *data.Slice // SHNO oscillator state
	atomistic_spins      *data.Slice // Atomistic spin grid
	dendrite_state       *data.Slice // Dendritic compartment states
	wta_activity         *data.Slice // WTA network activity
	V_Hall_field         *data.Slice // Topological Hall voltage
	spinpump_current     *data.Slice // Spin pumping current
)

// ============================================================================
// FEATURE 1: MULTI-NEIGHBOR RKKY
// ============================================================================

func EnableMultiNeighborRKKY() {
	multiNeighborRKKYEnabled = true
	rkky_neighbors.setRegion(0, []float64{3}) // Default: 3 neighbors (z±1,z±2,z±3)
	rkky_J1.setRegion(0, []float64{-1e-3})    // Default values
	rkky_J2.setRegion(0, []float64{-0.5e-3})
	rkky_J3.setRegion(0, []float64{-0.2e-3})
	rkky_J_lateral.setRegion(0, []float64{-0.1e-3})
	LogOut("Multi-neighbor RKKY enabled (z±1,z±2,z±3,lateral)")
}

func SetRKKYNeighbors(n int) {
	if n < 1 || n > 10 {
		LogErr("RKKY neighbors must be 1-10")
		return
	}
	rkky_neighbors.setRegion(0, []float64{float64(n)})
}

func SetRKKYCouplings(J1, J2, J3, J_lat float64) {
	rkky_J1.setRegion(0, []float64{J1})
	rkky_J2.setRegion(0, []float64{J2})
	rkky_J3.setRegion(0, []float64{J3})
	rkky_J_lateral.setRegion(0, []float64{J_lat})
}

func AddMultiNeighborRKKYField(dst *data.Slice) {
	if !multiNeighborRKKYEnabled {
		return
	}

	n_neighbors := int(rkky_neighbors.GetRegion(0))
	J1 := float32(rkky_J1.GetRegion(0))
	J2 := float32(rkky_J2.GetRegion(0))
	J3 := float32(rkky_J3.GetRegion(0))
	J_lat := float32(rkky_J_lateral.GetRegion(0))

	layer1 := 0 // From V1 SAF config
	layer2 := 1

	thickness := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, r1 := Msat.Slice()
	if r1 {
		defer cuda.SAFRecycle(ms_buf)
	}
	cuda.SAFAddMultiNeighborRKKY_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr),
		J1, J2, J3, J_lat,
		layer1, layer2, n_neighbors, thickness,
		Nx, Ny, Nz, Mesh().NCell())
}

// ============================================================================
// FEATURE 2: INTERLAYER DMI
// ============================================================================

func EnableInterlayerDMI() {
	interlayerDMIEnabled = true
	D_interlayer.setRegion(0, []float64{1e-3}) // Default 1 mJ/m²
	LogOut("Interlayer DMI enabled")
}

func SetInterlayerDMI(D float64) {
	D_interlayer.setRegion(0, []float64{D})
}

func AddInterlayerDMIField(dst *data.Slice) {
	if !interlayerDMIEnabled {
		return
	}

	D := float32(D_interlayer.GetRegion(0))
	layer1, layer2 := 0, 1
	dx := float32(Mesh().CellSize()[0])
	dy := float32(Mesh().CellSize()[1])
	dz := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, r1 := Msat.Slice()
	if r1 {
		defer cuda.SAFRecycle(ms_buf)
	}
	cuda.SAFAddInterlayerDMI_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr),
		D, layer1, layer2, dx, dy, dz,
		Nx, Ny, Nz, Mesh().NCell())
}

// ============================================================================
// FEATURE 3: NON-COLLINEAR RKKY
// ============================================================================

func EnableNonCollinearRKKY() {
	nonCollinearRKKYEnabled = true
	J_noncollinear.setRegion(0, []float64{-1e-3})
	alpha_noncollinear.setRegion(0, []float64{1.0}) // Heisenberg default
	LogOut("Non-collinear RKKY enabled")
}

func SetNonCollinearRKKY(J, alpha float64) {
	J_noncollinear.setRegion(0, []float64{J})
	alpha_noncollinear.setRegion(0, []float64{alpha})
}

func AddNonCollinearRKKYField(dst *data.Slice) {
	if !nonCollinearRKKYEnabled {
		return
	}

	J := float32(J_noncollinear.GetRegion(0))
	alpha := float32(alpha_noncollinear.GetRegion(0))
	layer1, layer2 := 0, 1
	thickness := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, r1 := Msat.Slice()
	if r1 {
		defer cuda.SAFRecycle(ms_buf)
	}
	cuda.SAFAddNonCollinearRKKY_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr),
		J, alpha, layer1, layer2, thickness,
		Nx, Ny, Nz, Mesh().NCell())
}

// ============================================================================
// FEATURE 4: SPIN DIFFUSION
// ============================================================================

func EnableSpinDiffusion() {
	spinDiffusionEnabled = true
	lambda_sf_v2.setRegion(0, []float64{5e-9}) // 5 nm default
	spinDiffusionMaxIter.setRegion(0, []float64{20})

	// Allocate mu_s field
	size := Mesh().Size()
	mu_s_field = data.NewSlice(3, size)

	LogOut("Spin diffusion enabled (Valet-Fert solver)")
}

func SetSpinDiffusionLength(lambda float64) {
	lambda_sf_v2.setRegion(0, []float64{lambda})
}

func SolveSpinDiffusion() {
	if !spinDiffusionEnabled {
		return
	}

	maxIter := int(spinDiffusionMaxIter.GetRegion(0))
	dz := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	j_buf, _ := J_current.Slice()
	st_buf, _ := STT_polarization.Slice()
	la_buf, _ := lambda_sf_v2.Slice()
	cuda.SAFSolveSpinDiffusion_CUDA(mu_s_field,
		unsafe.Pointer(j_buf.DevPtr(0)),
		unsafe.Pointer(st_buf.DevPtr(0)),
		unsafe.Pointer(la_buf.DevPtr(0)),
		dz, Nx, Ny, Nz, maxIter)
}

func AddSpinDiffusionField(dst *data.Slice) {
	if !spinDiffusionEnabled {
		return
	}

	SolveSpinDiffusion() // Update mu_s

	theta_SH := float32(SOT_theta_SH.GetRegion(0))
	thickness := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	cuda.SAFAddSpinDiffusionField_CUDA(dst, M.Buffer(), mu_s_field,
		unsafe.Pointer(ms_buf.DevPtr(0)),
		theta_SH, thickness, Nx, Ny, Nz, Mesh().NCell())
}

func GetSpinAccumulation() *data.Slice {
	return mu_s_field
}

// ============================================================================
// FEATURES 5-11: REMAINING GO WRAPPERS (Compact)
// ============================================================================

// FEATURE 5: STOCHASTIC STDP
func EnableStochasticSTDP() {
	stochasticSTDPEnabled = true
	stdp_noise_sigma.setRegion(0, []float64{0.05})    // 5% noise
	stdp_retention_tau.setRegion(0, []float64{86400}) // 1 day
	LogOut("Stochastic STDP enabled")
}

func ApplyStochasticSTDP(weights *data.Slice, pre, post *data.Slice) {
	if !stochasticSTDPEnabled || !thermalEnabled {
		return
	}
	// Call CUDA kernel with cuRAND states
	A_plus := float32(0.01)
	A_minus := float32(0.01)
	tau_plus := float32(20e-3)
	tau_minus := float32(20e-3)
	noise := float32(stdp_noise_sigma.GetRegion(0))
	retention := float32(stdp_retention_tau.GetRegion(0))
	dt := float32(Time)

	cuda.SAFApplyStochasticSTDP_CUDA(weights, pre, post,
		thermalStates, A_plus, A_minus, tau_plus, tau_minus,
		noise, retention, dt, Mesh().NCell())
}

// FEATURE 6: RESERVOIR COMPUTING
func EnableReservoirComputing() {
	reservoirComputingEnabled = true
	reservoir_leak_rate.setRegion(0, []float64{0.3})
	reservoir_spectral_radius.setRegion(0, []float64{0.9})

	size := Mesh().Size()
	N := size[0] * size[1] // Number of reservoir neurons
	N_inputs := 3          // Input dimension (mx, my, mz)

	reservoir_state = data.NewSlice(1, [3]int{size[0], size[1], 1})
	reservoir_weights = data.NewSlice(1, [3]int{N, N_inputs, 1})

	// Initialize random weights (uniform distribution [-1, 1])
	cuda.Memset(reservoir_weights, 0) // Will be replaced with proper random init

	LogOut("Reservoir computing enabled")
}

func UpdateReservoirState(input *data.Slice) *data.Slice {
	if !reservoirComputingEnabled || reservoir_weights == nil {
		return nil
	}

	leak := float32(reservoir_leak_rate.GetRegion(0))
	rho := float32(reservoir_spectral_radius.GetRegion(0))

	size := Mesh().Size()
	N := size[0] * size[1]
	N_inputs := 3

	cuda.SAFUpdateReservoirState_CUDA(reservoir_state, input, reservoir_weights,
		leak, rho, N, N_inputs)

	return reservoir_state
}

// FEATURE 7: METAPLASTICITY
func EnableMetaplasticity() {
	metaplasticityEnabled = true
	meta_theta_target.setRegion(0, []float64{0.5})
	meta_tau.setRegion(0, []float64{1000.0})

	size := Mesh().Size()
	meta_threshold = data.NewSlice(1, size)
	LogOut("Metaplasticity enabled")
}

func ApplyMetaplasticity(weights, activity *data.Slice) {
	if !metaplasticityEnabled {
		return
	}

	theta := float32(meta_theta_target.GetRegion(0))
	tau := float32(meta_tau.GetRegion(0))
	dt := float32(Time)

	cuda.SAFApplyMetaplasticity_CUDA(weights, meta_threshold, activity,
		theta, tau, dt, Mesh().NCell())
}

// FEATURE 8: HEAT DIFFUSION
func EnableHeatDiffusion() {
	heatDiffusionEnabled = true
	kappa_thermal.setRegion(0, []float64{100.0}) // W/(m·K)
	rho_cp.setRegion(0, []float64{3e6})          // J/(m³·K)

	size := Mesh().Size()
	temperature_field_v2 = data.NewSlice(1, size)
	LogOut("Heat diffusion enabled")
}

func SolveHeatDiffusion(dt float64) {
	if !heatDiffusionEnabled {
		return
	}

	kappa := float32(kappa_thermal.GetRegion(0))
	_ = float32(rho_cp.GetRegion(0))
	dx, dy, dz := float32(Mesh().CellSize()[0]), float32(Mesh().CellSize()[1]), float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	j_buf, _ := J_current.Slice()
	si_buf, _ := sigma_saf.Slice()
	rh_buf, _ := rho_cp.Slice()
	cuda.SAFSolveHeatDiffusion_CUDA(temperature_field_v2, temperature_field_v2,
		unsafe.Pointer(j_buf.DevPtr(0)),
		unsafe.Pointer(si_buf.DevPtr(0)),
		unsafe.Pointer(rh_buf.DevPtr(0)),
		kappa, float32(dt), dx, dy, dz, Nx, Ny, Nz, Mesh().NCell())
}

func GetTemperatureField() *data.Slice {
	return temperature_field_v2
}

// FEATURE 9: NONLINEAR VCMA
func EnableNonlinearVCMA() {
	nonlinearVCMAEnabled = true
	xi_vcma2.setRegion(0, []float64{1e-11}) // J/(V²·m)
	xi_vcma3.setRegion(0, []float64{1e-12}) // J/(V³·m)
	LogOut("Nonlinear VCMA enabled")
}

func AddNonlinearVCMAField(dst *data.Slice) {
	if !nonlinearVCMAEnabled {
		return
	}

	xi1 := float32(xi_VCMA.GetRegion(0))
	xi2 := float32(xi_vcma2.GetRegion(0))
	xi3 := float32(xi_vcma3.GetRegion(0))
	t_int := float32(t_interface.GetRegion(0))
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	ef_buf, _ := E_field.Slice()
	cuda.SAFAddNonlinearVCMA_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(ef_buf.DevPtr(0)),
		xi1, xi2, xi3, t_int, Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 10: MAGNON-PHONON COUPLING
func EnableMagnonPhononCoupling() {
	magnonPhononEnabled = true
	B1_magnetoelastic.setRegion(0, []float64{1e6}) // J/m³
	B2_magnetoelastic.setRegion(0, []float64{1e6})

	size := Mesh().Size()
	strain_field = data.NewSlice(2, [3]int{size[0], size[1], 1}) // εxx, εyy
	LogOut("Magnon-phonon coupling enabled")
}

func AddMagnonPhononField(dst *data.Slice) {
	if !magnonPhononEnabled {
		return
	}

	B1 := float32(B1_magnetoelastic.GetRegion(0))
	B2 := float32(B2_magnetoelastic.GetRegion(0))
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	cuda.SAFAddMagnonPhononCoupling_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(strain_field.DevPtr(0)),
		unsafe.Pointer(strain_field.DevPtr(1)),
		B1, B2, Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 11: QUANTUM TUNNELING
func EnableQuantumTunneling() {
	quantumTunnelingEnabled = true
	barrier_thickness.setRegion(0, []float64{1e-9}) // 1 nm
	barrier_height.setRegion(0, []float64{1.0})     // 1 eV
	polarization_1.setRegion(0, []float64{0.5})
	polarization_2.setRegion(0, []float64{0.5})
	LogOut("Quantum tunneling enabled")
}

func CalculateQuantumTMR() *data.Slice {
	if !quantumTunnelingEnabled {
		return nil
	}

	size := Mesh().Size()
	TMR := data.NewSlice(1, size)

	t_b := float32(barrier_thickness.GetRegion(0))
	phi := float32(barrier_height.GetRegion(0))
	P1 := float32(polarization_1.GetRegion(0))
	P2 := float32(polarization_2.GetRegion(0))
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	cuda.SAFCalculateQuantumTMR_CUDA(TMR, M.Buffer(),
		t_b, phi, P1, P2, Nx, Ny, Nz, Mesh().NCell())

	return TMR
}

// FEATURE 12: ORANGE-PEEL COUPLING (Pure Go - CPU FFT)
func EnableOrangePeel() {
	orangePeelEnabled = true
	F_rms_roughness.setRegion(0, []float64{0.3e-9}) // 0.3 nm RMS
	xi_correlation.setRegion(0, []float64{10e-9})   // 10 nm correlation
	LogOut("Orange-peel coupling enabled")
}

func AddOrangePeelField(dst *data.Slice) {
	if !orangePeelEnabled {
		return
	}

	// Analytical approximation of orange-peel (Néel) coupling
	// Based on: H_op = (μ₀ Ms / 2) * (F_rms / t_s)² * (π/ξ) * m_opposite
	// Reference: Néel (1962), Kools (1996)

	F_rms := F_rms_roughness.GetRegion(0)
	xi := xi_correlation.GetRegion(0)
	t_s := Mesh().CellSize()[2] // Spacer thickness = Z cell size

	if t_s == 0 {
		return // No spacer, no orange-peel
	}

	// Coupling strength (SI units)
	mu0 := 4 * 3.14159265359 * 1e-7
	Ms := Msat.GetRegion(0)
	J_op := mu0 * Ms * Ms * (F_rms / t_s) * (F_rms / t_s) * (3.14159265359 / xi) / 2

	// Apply as interlayer coupling (CPU implementation)
	m := M.Buffer()
	size := Mesh().Size()
	Nx, Ny, Nz := size[0], size[1], size[2]

	// Simple 2-layer coupling (extend for multilayer if needed)
	if Nz < 2 {
		return
	}

	// H_op = -J_op * m_opposite_layer
	// Convert J to field: H = J / (μ₀ Ms² t) where t is layer thickness
	thickness := Mesh().CellSize()[2]
	H_op := J_op / (mu0 * Ms * Ms * thickness)

	// Apply coupling between adjacent layers (CPU loop)
	mx_data := m.Host()[0]
	my_data := m.Host()[1]
	mz_data := m.Host()[2]
	hx_data := dst.Host()[0]
	hy_data := dst.Host()[1]
	hz_data := dst.Host()[2]

	for iz := 0; iz < Nz-1; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				idx1 := ix + Nx*(iy+Ny*iz)
				idx2 := ix + Nx*(iy+Ny*(iz+1))

				// Layer iz feels field from layer iz+1
				hx_data[idx1] += float32(H_op * float64(mx_data[idx2]))
				hy_data[idx1] += float32(H_op * float64(my_data[idx2]))
				hz_data[idx1] += float32(H_op * float64(mz_data[idx2]))

				// Layer iz+1 feels field from layer iz
				hx_data[idx2] += float32(H_op * float64(mx_data[idx1]))
				hy_data[idx2] += float32(H_op * float64(my_data[idx1]))
				hz_data[idx2] += float32(H_op * float64(mz_data[idx1]))
			}
		}
	}
}

// ============================================================================
// FEATURES 13-25: ADVANCED V2 IMPLEMENTATIONS
// ============================================================================

// FEATURE 13: TEMPERATURE-DEPENDENT RKKY
func EnableTemperatureDependentRKKY() {
	temperatureRKKYEnabled = true
	rkky_J0_base.setRegion(0, []float64{-1.5e-3}) // -1.5 mJ/m²
	rkky_T_curie.setRegion(0, []float64{1000.0})  // 1000 K
	rkky_temp_exp.setRegion(0, []float64{1.5})    // α = 1.5
	LogOut("Temperature-dependent RKKY enabled")
}

func AddTemperatureDependentRKKYField(dst *data.Slice) {
	if !temperatureRKKYEnabled || temperature_field_v2 == nil {
		return
	}

	J0 := float32(rkky_J0_base.GetRegion(0))
	T_c := float32(rkky_T_curie.GetRegion(0))
	alpha := float32(rkky_temp_exp.GetRegion(0))
	layer1, layer2 := 0, 1
	thickness := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	cuda.SAFAddTemperatureDependentRKKY_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr),
		temperature_field_v2,
		J0, T_c, alpha, layer1, layer2, thickness,
		Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 14: SPIN WAVE FFT ANALYSIS
func EnableSpinWaveFFT() {
	spinWaveFFTEnabled = true
	sw_fft_max_freq.setRegion(0, []float64{100e9}) // 100 GHz
	sw_fft_bins.setRegion(0, []float64{512})       // 512 bins

	size := Mesh().Size()
	bins := int(sw_fft_bins.GetRegion(0))
	sw_fft_spectrum = data.NewSlice(1, [3]int{bins, size[1], size[2]})
	LogOut("Spin wave FFT enabled")
}

func ComputeSpinWaveFFT() *data.Slice {
	if !spinWaveFFTEnabled {
		return nil
	}

	fmax := float32(sw_fft_max_freq.GetRegion(0))
	bins := int(sw_fft_bins.GetRegion(0))
	dt := float32(Time)
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	cuda.SAFComputeSpinWaveFFT_CUDA(sw_fft_spectrum, M.Buffer(),
		fmax, bins, dt, Nx, Ny, Nz, Mesh().NCell())

	return sw_fft_spectrum
}

// FEATURE 15: SHNO (Spin Hall Nano-Oscillators)
func EnableSHNO() {
	shnoEnabled = true
	shno_J_SHE.setRegion(0, []float64{1e11})   // 10¹¹ A/m²
	shno_theta_SH.setRegion(0, []float64{0.1}) // θ_SH = 0.1

	size := Mesh().Size()
	shno_state = data.NewSlice(3, size) // Store oscillator phase/amplitude
	LogOut("SHNO enabled")
}

func UpdateSHNO(dt float64) {
	if !shnoEnabled {
		return
	}

	J_SHE := float32(shno_J_SHE.GetRegion(0))
	theta := float32(shno_theta_SH.GetRegion(0))
	alpha := float32(Alpha.GetRegion(0))
	dt_f := float32(dt)
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	cuda.SAFUpdateSHNO_CUDA(M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr),
		J_SHE, theta, alpha, dt_f,
		Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 16: EXCHANGE BIAS
func EnableExchangeBias() {
	exchangeBiasEnabled = true
	J_exbias.setRegion(0, []float64{-0.5e-3}) // -0.5 mJ/m²
	H_exbias_x.setRegion(0, []float64{1e5})   // 100 kA/m
	H_exbias_y.setRegion(0, []float64{0.0})
	H_exbias_z.setRegion(0, []float64{0.0})
	LogOut("Exchange bias enabled")
}

func AddExchangeBiasField(dst *data.Slice) {
	if !exchangeBiasEnabled {
		return
	}

	J := float32(J_exbias.GetRegion(0))
	Hx := float32(H_exbias_x.GetRegion(0))
	Hy := float32(H_exbias_y.GetRegion(0))
	Hz := float32(H_exbias_z.GetRegion(0))
	layer_FM, layer_AFM := 0, 1
	thickness := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	cuda.SAFAddExchangeBias_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr),
		J, Hx, Hy, Hz, layer_FM, layer_AFM, thickness,
		Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 17: VOLTAGE-CONTROLLED RKKY
func EnableVoltageRKKY() {
	voltageRKKYEnabled = true
	rkky_voltage_coeff.setRegion(0, []float64{1e-12}) // 1 pJ/(V·m²)
	LogOut("Voltage-controlled RKKY enabled")
}

func AddVoltageRKKYField(dst *data.Slice) {
	if !voltageRKKYEnabled {
		return
	}

	J0 := float32(rkky_J1.GetRegion(0))
	dJ_dV := float32(rkky_voltage_coeff.GetRegion(0))
	layer1, layer2 := 0, 1
	thickness := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	ef_buf, _ := E_field.Slice()
	cuda.SAFAddVoltageRKKY_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr),
		unsafe.Pointer(ef_buf.DevPtr(0)),
		J0, dJ_dV, layer1, layer2, thickness,
		Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 18: ATOMISTIC-CONTINUUM COUPLING
func EnableAtomisticContinuum() {
	atomisticContinuumEnabled = true
	atomistic_a.setRegion(0, []float64{0.25e-9})     // 0.25 nm (Fe)
	atomistic_J_ex.setRegion(0, []float64{2.16e-21}) // 13.5 meV

	size := Mesh().Size()
	// Oversample by 4x in each direction for atomistic grid
	atomistic_spins = data.NewSlice(3, [3]int{size[0] * 4, size[1] * 4, size[2] * 4})
	LogOut("Atomistic-continuum coupling enabled")
}

func CoupleAtomisticToContinuum() {
	if !atomisticContinuumEnabled {
		return
	}

	a := float32(atomistic_a.GetRegion(0))
	J := float32(atomistic_J_ex.GetRegion(0))
	Aex_cont := float32(Aex.GetRegion(0))
	dx := float32(Mesh().CellSize()[0])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	cuda.SAFCoupleAtomisticContinuum_CUDA(M.Buffer(), atomistic_spins,
		unsafe.Pointer(ms_buf.DevPtr(0)),
		a, J, Aex_cont, dx,
		Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 19: LL-BLOCH EQUATIONS
func EnableLLBloch() {
	llBlochEnabled = true
	llbloch_T_curie.setRegion(0, []float64{1043.0}) // Co: 1043 K
	llbloch_lambda.setRegion(0, []float64{0.01})    // Longitudinal relaxation
	LogOut("Landau-Lifshitz-Bloch equations enabled")
}

func UpdateLLBloch(dt float64) {
	if !llBlochEnabled || temperature_field_v2 == nil {
		return
	}

	Ms0 := float32(Msat.GetRegion(0))
	T_c := float32(llbloch_T_curie.GetRegion(0))
	lambda := float32(llbloch_lambda.GetRegion(0))
	alpha := float32(Alpha.GetRegion(0))
	dt_f := float32(dt)
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	cuda.SAFUpdateLLBloch_CUDA(M.Buffer(), M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		temperature_field_v2,
		Ms0, T_c, lambda, alpha, dt_f,
		Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 20: DIPOLAR SKYRMION INTERACTIONS
func EnableDipolarSkyrmions() {
	dipolarSkyrmiansEnabled = true
	dipolar_cutoff.setRegion(0, []float64{50e-9}) // 50 nm cutoff
	LogOut("Dipolar skyrmion interactions enabled")
}

func AddDipolarSkyrmionField(dst *data.Slice) {
	if !dipolarSkyrmiansEnabled {
		return
	}

	cutoff := float32(dipolar_cutoff.GetRegion(0))
	dx := float32(Mesh().CellSize()[0])
	dy := float32(Mesh().CellSize()[1])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	ms_buf, _ := Msat.Slice()
	cuda.SAFAddDipolarSkyrmion_CUDA(dst, M.Buffer(),
		unsafe.Pointer(ms_buf.DevPtr(0)),
		cutoff, dx, dy,
		Nx, Ny, Nz, Mesh().NCell())
}

// FEATURE 21: SYNAPTIC HOMEOSTASIS
func EnableHomeostasis() {
	homeostasisEnabled = true
	homeo_target.setRegion(0, []float64{0.1}) // 10% target activity
	homeo_tau.setRegion(0, []float64{1000.0}) // 1000s time constant
	LogOut("Synaptic homeostasis enabled")
}

func ApplyHomeostasis(weights, activity *data.Slice, dt float64) {
	if !homeostasisEnabled {
		return
	}

	target := float32(homeo_target.GetRegion(0))
	tau := float32(homeo_tau.GetRegion(0))
	dt_f := float32(dt)

	cuda.SAFApplyHomeostasis_CUDA(weights, activity,
		target, tau, dt_f, Mesh().NCell())
}

// FEATURE 22: DENDRITIC COMPUTATION
var dendrite_weights *data.Slice

func EnableDendritic() {
	dendriticEnabled = true
	dendrite_threshold.setRegion(0, []float64{0.5})
	dendrite_nonlin.setRegion(0, []float64{2.0}) // Quadratic

	size := Mesh().Size()
	dendrite_state = data.NewSlice(1, size)
	// Initialize default weights to 1.0
	dendrite_weights = data.NewSlice(1, size)
	cuda.Memset(dendrite_weights, 0)

	LogOut("Dendritic computation enabled")
}

func SetDendriticWeights(w *data.Slice) {
	if dendrite_weights == nil {
		EnableDendritic()
	}
	cuda.MemCpy(dendrite_weights.DevPtr(0), w.DevPtr(0), int64(w.Len()*4))
}

func ComputeDendriticOutput(inputs *data.Slice) *data.Slice {
	if !dendriticEnabled {
		return nil
	}

	// Safety check
	if dendrite_weights == nil {
		EnableDendritic()
	}

	thresh := float32(dendrite_threshold.GetRegion(0))
	V_reset := float32(0.0) // Default reset potential, can be exposed if needed

	cuda.SAFComputeDendritic_CUDA(dendrite_state, inputs, dendrite_weights,
		thresh, V_reset, Mesh().NCell())

	return dendrite_state
}

// FEATURE 23: WINNER-TAKE-ALL
func EnableWTA() {
	wtaEnabled = true
	wta_inhibition.setRegion(0, []float64{0.5})
	wta_tau.setRegion(0, []float64{10e-3}) // 10 ms

	size := Mesh().Size()
	wta_activity = data.NewSlice(1, size)
	LogOut("Winner-Take-All enabled")
}

func UpdateWTA(input *data.Slice, dt float64) *data.Slice {
	if !wtaEnabled {
		return nil
	}

	inhibit := float32(wta_inhibition.GetRegion(0))
	tau := float32(wta_tau.GetRegion(0))
	dt_f := float32(dt)

	cuda.SAFUpdateWTA_CUDA(wta_activity, input,
		inhibit, tau, dt_f, Mesh().NCell())

	return wta_activity
}

// FEATURE 24: TOPOLOGICAL HALL EFFECT
func EnableTopologicalHall() {
	topologicalHallEnabled = true
	the_rho_n.setRegion(0, []float64{6e28}) // Typical metal carrier density

	size := Mesh().Size()
	V_Hall_field = data.NewSlice(1, size)
	LogOut("Topological Hall effect enabled")
}

func CalculateTopologicalHall(J_current *data.Slice) *data.Slice {
	if !topologicalHallEnabled {
		return nil
	}

	rho_n := float32(the_rho_n.GetRegion(0))
	dx := float32(Mesh().CellSize()[0])
	dy := float32(Mesh().CellSize()[1])
	thickness := float32(Mesh().CellSize()[2])
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	cuda.SAFCalculateTopologicalHall_CUDA(V_Hall_field, M.Buffer(), J_current,
		rho_n, dx, dy, thickness,
		Nx, Ny, Nz, Mesh().NCell())

	return V_Hall_field
}

// FEATURE 25: SPIN-CHARGE PUMPING
// Stores previous magnetization for dm/dt calculation
var spinpump_m_old *data.Slice

func EnableSpinPumping() {
	spinPumpingEnabled = true
	spinpump_g_eff.setRegion(0, []float64{1e19})    // 10¹⁹ m⁻²
	spinpump_lambda_N.setRegion(0, []float64{5e-9}) // 5 nm

	size := Mesh().Size()
	spinpump_current = data.NewSlice(3, size)
	// Allocate m_old buffer
	spinpump_m_old = data.NewSlice(3, size)
	// Initialize with current magnetization to avoid large startup transient
	cuda.MemCpy(spinpump_m_old.DevPtr(0), M.Buffer().DevPtr(0), int64(M.Buffer().Len()*4))
	cuda.MemCpy(spinpump_m_old.DevPtr(1), M.Buffer().DevPtr(1), int64(M.Buffer().Len()*4))
	cuda.MemCpy(spinpump_m_old.DevPtr(2), M.Buffer().DevPtr(2), int64(M.Buffer().Len()*4))

	LogOut("Spin pumping enabled (tracked m_old)")
}

func CalculateSpinPumping(dt float64) *data.Slice {
	if !spinPumpingEnabled {
		return nil
	}

	// Ensure m_old exists (safety check)
	if spinpump_m_old == nil {
		EnableSpinPumping()
	}

	g_eff := float32(spinpump_g_eff.GetRegion(0))
	lambda_N := float32(spinpump_lambda_N.GetRegion(0))
	dt_f := float32(dt)
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	cuda.SAFCalculateSpinPumping_CUDA(spinpump_current, M.Buffer(), spinpump_m_old,
		g_eff, lambda_N, dt_f,
		Nx, Ny, Nz, Mesh().NCell())

	// Update m_old for next step: m_old = m_current
	// We copy AFTER calculation so next step has (m_t, m_{t-1})
	// Note: This assumes CalculateSpinPumping is called once per time step
	cuda.MemCpy(spinpump_m_old.DevPtr(0), M.Buffer().DevPtr(0), int64(M.Buffer().Len()*4))
	cuda.MemCpy(spinpump_m_old.DevPtr(1), M.Buffer().DevPtr(1), int64(M.Buffer().Len()*4))
	cuda.MemCpy(spinpump_m_old.DevPtr(2), M.Buffer().DevPtr(2), int64(M.Buffer().Len()*4))

	return spinpump_current
}

// ============================================================================
// V2 REGISTRATIONS
// ============================================================================

func init() {
	// Feature 1: Multi-neighbor RKKY
	DeclFunc("EnableMultiNeighborRKKY", EnableMultiNeighborRKKY,
		"Enable multi-neighbor RKKY coupling (z±1,z±2,z±3,lateral)")
	DeclFunc("SetRKKYNeighbors", SetRKKYNeighbors,
		"Set number of RKKY neighbors to include (1-10)")
	DeclFunc("SetRKKYCouplings", SetRKKYCouplings,
		"Set RKKY coupling strengths (J1, J2, J3, J_lateral)")

	// Feature 2: Interlayer DMI
	DeclFunc("EnableInterlayerDMI", EnableInterlayerDMI,
		"Enable interlayer Dzyaloshinskii-Moriya interaction")
	DeclFunc("SetInterlayerDMI", SetInterlayerDMI,
		"Set interlayer DMI strength D (J/m²)")

	// Feature 3: Non-collinear RKKY
	DeclFunc("EnableNonCollinearRKKY", EnableNonCollinearRKKY,
		"Enable non-collinear RKKY for frustrated systems")
	DeclFunc("SetNonCollinearRKKY", SetNonCollinearRKKY,
		"Set non-collinear RKKY parameters (J, alpha)")

	// Feature 4: Spin diffusion
	DeclFunc("EnableSpinDiffusion", EnableSpinDiffusion,
		"Enable spin diffusion (Valet-Fert solver)")
	DeclFunc("SetSpinDiffusionLength", SetSpinDiffusionLength,
		"Set spin diffusion length λsf (m)")
	DeclFunc("GetSpinAccumulation", GetSpinAccumulation,
		"Get spin accumulation field μs(x,y,z)")

	// Feature 5: Stochastic STDP
	DeclFunc("EnableStochasticSTDP", EnableStochasticSTDP,
		"Enable stochastic STDP with write noise and retention")
	DeclFunc("ApplyStochasticSTDP", ApplyStochasticSTDP,
		"Apply stochastic STDP to weight matrix")

	// Feature 6: Reservoir computing
	DeclFunc("EnableReservoirComputing", EnableReservoirComputing,
		"Enable reservoir computing primitives")
	DeclFunc("UpdateReservoirState", UpdateReservoirState,
		"Update reservoir state with input")

	// Feature 7: Metaplasticity
	DeclFunc("EnableMetaplasticity", EnableMetaplasticity,
		"Enable metaplasticity STDP rules")
	DeclFunc("ApplyMetaplasticity", ApplyMetaplasticity,
		"Apply metaplasticity to weights and thresholds")

	// Feature 8: Heat diffusion
	DeclFunc("EnableHeatDiffusion", EnableHeatDiffusion,
		"Enable Joule heating and thermal diffusion")
	DeclFunc("SolveHeatDiffusion", SolveHeatDiffusion,
		"Solve heat diffusion equation")
	DeclFunc("GetTemperatureField", GetTemperatureField,
		"Get temperature field T(x,y,z)")

	// Feature 9: Nonlinear VCMA
	DeclFunc("EnableNonlinearVCMA", EnableNonlinearVCMA,
		"Enable nonlinear VCMA (quadratic + cubic)")

	// Feature 10: Magnon-phonon
	DeclFunc("EnableMagnonPhononCoupling", EnableMagnonPhononCoupling,
		"Enable magnon-phonon interactions (magnetoelastic)")

	// Feature 11: Quantum tunneling
	DeclFunc("EnableQuantumTunneling", EnableQuantumTunneling,
		"Enable quantum tunneling corrections for TMR")
	DeclFunc("CalculateQuantumTMR", CalculateQuantumTMR,
		"Calculate quantum-corrected TMR")

	// Feature 12: Orange-peel
	DeclFunc("EnableOrangePeel", EnableOrangePeel,
		"Enable orange-peel coupling from surface roughness")

	// ========================================================================
	// FEATURES 13-25: ADVANCED V2 REGISTRATIONS
	// ========================================================================

	// Feature 13: Temperature-dependent RKKY
	DeclFunc("EnableTemperatureDependentRKKY", EnableTemperatureDependentRKKY,
		"Enable temperature-dependent RKKY coupling")

	// Feature 14: Spin wave FFT
	DeclFunc("EnableSpinWaveFFT", EnableSpinWaveFFT,
		"Enable spin wave FFT analysis")
	DeclFunc("ComputeSpinWaveFFT", ComputeSpinWaveFFT,
		"Compute spin wave frequency spectrum")

	// Feature 15: SHNO
	DeclFunc("EnableSHNO", EnableSHNO,
		"Enable Spin Hall Nano-Oscillators")
	DeclFunc("UpdateSHNO", UpdateSHNO,
		"Update SHNO dynamics")

	// Feature 16: Exchange bias
	DeclFunc("EnableExchangeBias", EnableExchangeBias,
		"Enable exchange bias (AFM/FM interface)")

	// Feature 17: Voltage-controlled RKKY
	DeclFunc("EnableVoltageRKKY", EnableVoltageRKKY,
		"Enable voltage-controlled RKKY coupling")

	// Feature 18: Atomistic-continuum
	DeclFunc("EnableAtomisticContinuum", EnableAtomisticContinuum,
		"Enable atomistic-continuum coupling")
	DeclFunc("CoupleAtomisticToContinuum", CoupleAtomisticToContinuum,
		"Couple atomistic spins to continuum magnetization")

	// Feature 19: LL-Bloch
	DeclFunc("EnableLLBloch", EnableLLBloch,
		"Enable Landau-Lifshitz-Bloch equations (high T)")
	DeclFunc("UpdateLLBloch", UpdateLLBloch,
		"Update magnetization using LL-Bloch")

	// Feature 20: Dipolar skyrmions
	DeclFunc("EnableDipolarSkyrmions", EnableDipolarSkyrmions,
		"Enable dipolar skyrmion-skyrmion interactions")

	// Feature 21: Synaptic homeostasis
	DeclFunc("EnableHomeostasis", EnableHomeostasis,
		"Enable synaptic homeostasis")
	DeclFunc("ApplyHomeostasis", ApplyHomeostasis,
		"Apply homeostasis to weights")

	// Feature 22: Dendritic computation
	DeclFunc("EnableDendritic", EnableDendritic,
		"Enable dendritic computation")
	DeclFunc("ComputeDendriticOutput", ComputeDendriticOutput,
		"Compute dendritic nonlinear integration")

	// Feature 23: Winner-Take-All
	DeclFunc("EnableWTA", EnableWTA,
		"Enable Winner-Take-All circuits")
	DeclFunc("UpdateWTA", UpdateWTA,
		"Update WTA network activity")

	// Feature 24: Topological Hall effect
	DeclFunc("EnableTopologicalHall", EnableTopologicalHall,
		"Enable topological Hall effect")
	DeclFunc("CalculateTopologicalHall", CalculateTopologicalHall,
		"Calculate topological Hall voltage")

	// Feature 25: Spin pumping
	DeclFunc("EnableSpinPumping", EnableSpinPumping,
		"Enable spin-charge pumping")
	DeclFunc("CalculateSpinPumping", CalculateSpinPumping,
		"Calculate spin pumping current")
}
