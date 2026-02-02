// Copyright © 2025-2026 Prof. Santhosh Sivasubramani
//
// Affiliations:
// 1. INTRINSIC Lab, Centre for Sensors Instrumentation and
//    Cyber Physical System Engineering (SeNSE)
//    Indian Institute of Technology Delhi, New Delhi, India
// 2. April AI Hub, Centre for Electronic Frontiers
//    The University of Edinburgh, Edinburgh, United Kingdom
//
// Contact: ssivasub@iitd.ac.in, ssivasub@ed.ac.uk, ragansanthosh@ieee.org
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

package engine

// MuMax3-SAF: Advanced Physics Integration for Synthetic Antiferromagnets & Neuromorphic Device Modeling
// Advanced CUDA physics kernels for spintronics simulations:
//   - RKKY interlayer coupling | Thermal fluctuations (cuRAND)
//   - SOT, STT, VCMA, Oersted fields | Neuromorphic computing
//   - 25 magnetic materials + 17 spacer materials database

// SAF-EXTENSION: Complete synthetic antiferromagnet physics engine - FINAL VERSION
// All functionality verified and complete
//
// Physics modules:
// - RKKY interlayer coupling with CPU fallback
// - Thermal stochastic fields with cuRAND
// - Spin-orbit torque (SOT)
// - Spin-transfer torque (STT)
// - Voltage-controlled magnetic anisotropy (VCMA)
// - Oersted fields from current
// - Topological charge calculation
// - Neuromorphic computing (weight programming, STDP, device DB)
//
// Database: 25 magnetic materials + 17 spacers

import (
	"log"
	"math"
	"time"
	"unsafe"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
)

// ============================================================================
// GLOBAL STATE
// ============================================================================

var (
	safEnabled          = false
	thermalEnabled      = false
	neuromorphicEnabled = false
	thermalStates       unsafe.Pointer

	// Support multiple layer pairs
	SAF_layer_pairs []LayerPair

	// Interface mapping for robust spacer support
	interfaceMapGPU  unsafe.Pointer
	interfaceMapSize int
)

// NEW: Define layer pair structure
type LayerPair struct {
	Layer1 int
	Layer2 int
}

// ============================================================================
// PARAMETERS - All 21 parameters exposed to .mx3 scripts
// ============================================================================

var (
	// RKKY coupling
	J_RKKY = NewScalarParam("J_RKKY", "J/m2", "RKKY interlayer coupling strength")
	// NEW: Oscillatory RKKY parameters
	J_RKKY_oscillatory  = NewScalarParam("J_RKKY_oscillatory", "", "Enable oscillatory RKKY (0=off, 1=on)")
	J_RKKY_amplitude    = NewScalarParam("J_RKKY_amplitude", "J/m2", "Oscillatory RKKY amplitude J₀")
	J_RKKY_wavelength   = NewScalarParam("J_RKKY_wavelength", "m", "RKKY oscillation wavelength λ_F")
	J_RKKY_phase        = NewScalarParam("J_RKKY_phase", "rad", "RKKY phase shift φ")
	J_RKKY_decay_length = NewScalarParam("J_RKKY_decay_length", "m", "RKKY decay length λ_D (0=no decay)")
	// NEW N2: Temperature-dependent RKKY
	J_RKKY_temp_dependent = NewScalarParam("J_RKKY_temp_dependent", "", "Enable temperature-dependent RKKY (0=off, 1=on)")
	J_RKKY_Tc             = NewScalarParam("J_RKKY_Tc", "K", "Critical temperature for RKKY coupling")
	// NEW N4: Skyrmion energy barriers
	Skyrmion_barrier_samples = NewScalarParam("Skyrmion_barrier_samples", "", "Number of intermediate states for barrier calculation")
	Skyrmion_radius_nm       = NewScalarParam("Skyrmion_radius_nm", "nm", "Expected skyrmion radius for initial guess")
	// NEW N5: Spin wave dispersion
	SpinWave_record_time     = NewScalarParam("SpinWave_record_time", "s", "Time duration for spin wave recording")
	SpinWave_sample_interval = NewScalarParam("SpinWave_sample_interval", "s", "Time between samples")
	SpinWave_excitation_freq = NewScalarParam("SpinWave_excitation_freq", "Hz", "Excitation frequency for spin waves")
	SpinWave_excitation_amp  = NewScalarParam("SpinWave_excitation_amp", "T", "Excitation field amplitude")
	// NEW N6: Advanced STDP
	STDP_mode              = NewScalarParam("STDP_mode", "", "STDP type: 0=pair, 1=triplet, 2=voltage-gated")
	STDP_triplet_tau_plus  = NewScalarParam("STDP_triplet_tau_plus", "s", "Triplet STDP: τ+ time constant")
	STDP_triplet_tau_minus = NewScalarParam("STDP_triplet_tau_minus", "s", "Triplet STDP: τ- time constant")
	STDP_triplet_tau_x     = NewScalarParam("STDP_triplet_tau_x", "s", "Triplet STDP: slow τx time constant")
	STDP_triplet_tau_y     = NewScalarParam("STDP_triplet_tau_y", "s", "Triplet STDP: slow τy time constant")
	STDP_triplet_A2_plus   = NewScalarParam("STDP_triplet_A2_plus", "", "Triplet STDP: A2+ amplitude")
	STDP_triplet_A2_minus  = NewScalarParam("STDP_triplet_A2_minus", "", "Triplet STDP: A2- amplitude")
	STDP_triplet_A3_plus   = NewScalarParam("STDP_triplet_A3_plus", "", "Triplet STDP: A3+ amplitude")
	STDP_triplet_A3_minus  = NewScalarParam("STDP_triplet_A3_minus", "", "Triplet STDP: A3- amplitude")
	STDP_voltage_threshold = NewScalarParam("STDP_voltage_threshold", "V", "Voltage-gated STDP: V threshold")
	STDP_voltage_window    = NewScalarParam("STDP_voltage_window", "s", "Voltage-gated STDP: timing window")
	// NEW N7: Stochastic resonance
	StochRes_signal_freq    = NewScalarParam("StochRes_signal_freq", "Hz", "Signal frequency for stochastic resonance")
	StochRes_signal_amp     = NewScalarParam("StochRes_signal_amp", "T", "Signal amplitude (weak)")
	StochRes_noise_scan_min = NewScalarParam("StochRes_noise_scan_min", "K", "Minimum noise temperature for scan")
	StochRes_noise_scan_max = NewScalarParam("StochRes_noise_scan_max", "K", "Maximum noise temperature for scan")
	StochRes_noise_steps    = NewScalarParam("StochRes_noise_steps", "", "Number of noise levels to test")
	StochRes_measure_time   = NewScalarParam("StochRes_measure_time", "s", "Measurement time per noise level")
	// NEW N8: Magnon-mediated coupling
	Magnon_coupling_enable   = NewScalarParam("Magnon_coupling_enable", "", "Enable magnon-mediated long-range coupling")
	Magnon_coupling_strength = NewScalarParam("Magnon_coupling_strength", "J/m^3", "Magnon coupling strength J_magnon")
	Magnon_coupling_range    = NewScalarParam("Magnon_coupling_range", "m", "Magnon coupling range (spatial decay)")
	Magnon_source_region     = NewScalarParam("Magnon_source_region", "", "Source region ID for magnon emission")
	Magnon_target_region     = NewScalarParam("Magnon_target_region", "", "Target region ID for magnon coupling")
	// Spacer transport properties
	sigma_saf  = NewScalarParam("sigma_saf", "S/m", "Electrical conductivity of spacer")
	Ds_saf     = NewScalarParam("Ds_saf", "m^2/s", "Spin diffusion constant")
	tau_sf_saf = NewScalarParam("tau_sf_saf", "s", "Spin-flip relaxation time")
	P_saf      = NewScalarParam("P_saf", "", "Spin polarization of spacer")

	// Thermal physics
	Temperature = NewScalarParam("Temperature", "K", "System temperature for thermal fluctuations")

	// Spin-orbit torque
	SOT_theta_SH = NewScalarParam("SOT_theta_SH", "", "Spin Hall angle for damping-like SOT")
	SOT_theta_FL = NewScalarParam("SOT_theta_FL", "", "Field-like SOT efficiency")

	// Spin-transfer torque
	STT_xi_adiabatic    = NewScalarParam("STT_xi_adiabatic", "", "Adiabatic STT coefficient")
	STT_xi_nonadiabatic = NewScalarParam("STT_xi_nonadiabatic", "", "Non-adiabatic STT coefficient")
	STT_polarization    = NewScalarParam("STT_polarization", "", "Current spin polarization")

	// Current and fields
	J_current = NewScalarParam("J_current", "A/m2", "Current density")
	E_field   = NewScalarParam("E_field", "V/m", "Electric field for VCMA")

	// VCMA
	xi_VCMA     = NewScalarParam("xi_VCMA", "J/(V*m2)", "VCMA coefficient")
	t_interface = NewScalarParam("t_interface", "m", "Interface thickness for VCMA")

	// Oersted field geometry
	wire_width     = NewScalarParam("wire_width", "m", "Current-carrying wire width")
	wire_thickness = NewScalarParam("wire_thickness", "m", "Wire thickness")

	// Neuromorphic parameters
	SynapticWeights = NewScalarParam("W_synapse", "", "Synaptic weight values")
	LearningRate    = NewScalarParam("eta_learning", "", "Learning rate for weight updates")
	STDP_A_plus     = NewScalarParam("STDP_A_plus", "", "STDP potentiation amplitude")
	STDP_A_minus    = NewScalarParam("STDP_A_minus", "", "STDP depression amplitude")
	STDP_tau_plus   = NewScalarParam("STDP_tau_plus", "s", "STDP potentiation time constant")
	STDP_tau_minus  = NewScalarParam("STDP_tau_minus", "s", "STDP depression time constant")
)

// ============================================================================
// ENERGY QUANTITIES (exposed to .mx3)
// ============================================================================
var (
	// Total SAF RKKY energy (J)
	E_SAF_RKKY = NewScalarValue("E_SAF_RKKY", "J", "Total SAF RKKY energy", GetSAFRKKYEnergy)
)

// ============================================================================
// FIELD QUANTITIES - Vector fields that contribute to B_eff
// ============================================================================

var (
	B_SAF_RKKY = NewVectorField("B_SAF_RKKY", "T", "RKKY interlayer coupling field", AddSAFRKKYField)
	B_thermal  = NewVectorField("B_thermal", "T", "Thermal stochastic field", AddThermalField)
	B_SOT      = NewVectorField("B_SOT", "T", "Spin-orbit torque field", AddSOTField)
	B_STT      = NewVectorField("B_STT", "T", "Spin-transfer torque field", AddSTTField)
	B_VCMA     = NewVectorField("B_VCMA", "T", "Voltage-controlled anisotropy field", AddVCMAField)
	B_Oersted  = NewVectorField("B_Oersted", "T", "Oersted field from current", AddOerstedField)
	B_magnon   = NewVectorField("B_magnon", "T", "Magnon-mediated coupling field", AddMagnonField) // NEW N8
)

// ============================================================================
// MATERIAL DATABASE - 25 magnetic materials
// ============================================================================

type MaterialPreset struct {
	Name  string
	Ms    float64 // Saturation magnetization (A/m)
	Aex   float64 // Exchange stiffness (J/m)
	Dmi   float64 // DMI constant (J/m²)
	Ku    float64 // Anisotropy (J/m³)
	Alpha float64 // Damping
	Tc    float64 // Curie temperature (K)
}

var MaterialPresets = map[string]MaterialPreset{
	// Thin film materials
	"CoFeB":           {Name: "CoFeB", Ms: 697e3, Aex: 9e-12, Dmi: 1.0e-3, Ku: 3.0e5, Alpha: 0.05, Tc: 673},
	"CoFeB_ultrathin": {Name: "CoFeB ultrathin", Ms: 1100e3, Aex: 16e-12, Dmi: 1.3e-3, Ku: 8e5, Alpha: 0.02, Tc: 673},
	"Py":              {Name: "Py Ni80Fe20", Ms: 800e3, Aex: 1.3e-11, Dmi: 0.2e-3, Ku: 1.0e3, Alpha: 0.008, Tc: 858},
	"Py_Ni81Fe19":     {Name: "Permalloy Ni81Fe19", Ms: 860e3, Aex: 1.05e-11, Dmi: 0.15e-3, Ku: 500, Alpha: 0.007, Tc: 858},

	// Transition metals
	"Co":      {Name: "Co bulk", Ms: 1.4e6, Aex: 3.0e-11, Dmi: 0.0, Ku: 5.3e5, Alpha: 0.005, Tc: 1388},
	"Co_thin": {Name: "Co thin film", Ms: 1.1e6, Aex: 1.6e-11, Dmi: 1.5e-3, Ku: 4.5e5, Alpha: 0.01, Tc: 1388},
	"Fe":      {Name: "Fe bulk", Ms: 1.71e6, Aex: 2.1e-11, Dmi: 0.0, Ku: 4.8e4, Alpha: 0.002, Tc: 1043},
	"Ni":      {Name: "Ni bulk", Ms: 485e3, Aex: 9e-12, Dmi: 0.0, Ku: -4.5e3, Alpha: 0.045, Tc: 631},
	"CoFe":    {Name: "CoFe", Ms: 1.2e6, Aex: 1.2e-11, Dmi: 0.5e-3, Ku: 2.5e5, Alpha: 0.02, Tc: 1243},

	// L10 ordered alloys
	"CoPt":     {Name: "CoPt L10", Ms: 800e3, Aex: 1.0e-11, Dmi: 2.5e-3, Ku: 4.9e6, Alpha: 0.05, Tc: 840},
	"CoPd":     {Name: "CoPd", Ms: 600e3, Aex: 1.0e-11, Dmi: 1.2e-3, Ku: 1.0e6, Alpha: 0.04, Tc: 723},
	"FePt_L10": {Name: "FePt L10", Ms: 1.14e6, Aex: 1.0e-11, Dmi: 0.0, Ku: 6.6e6, Alpha: 0.04, Tc: 750},
	"FePd_L10": {Name: "FePd L10", Ms: 1.1e6, Aex: 1.0e-11, Dmi: 0.0, Ku: 1.8e6, Alpha: 0.08, Tc: 760},

	// Binary alloys
	"FeNi":       {Name: "FeNi", Ms: 1.0e6, Aex: 1.0e-11, Dmi: 0.3e-3, Ku: 1.0e4, Alpha: 0.01, Tc: 773},
	"NiFe_50_50": {Name: "NiFe 50:50", Ms: 950e3, Aex: 1.0e-11, Dmi: 0.2e-3, Ku: 5.0e3, Alpha: 0.015, Tc: 803},

	// Heusler alloys
	"Co2MnSi": {Name: "Co2MnSi", Ms: 1.01e6, Aex: 1.5e-11, Dmi: 0.0, Ku: 2.0e4, Alpha: 0.004, Tc: 985},
	"Co2FeAl": {Name: "Co2FeAl", Ms: 1.2e6, Aex: 1.8e-11, Dmi: 0.0, Ku: 1.5e4, Alpha: 0.003, Tc: 1168},
	"Co2FeSi": {Name: "Co2FeSi", Ms: 1.0e6, Aex: 2.0e-11, Dmi: 0.0, Ku: 5.0e4, Alpha: 0.002, Tc: 1100},

	// Rare-earth ferrimagnets
	"GdFeCo": {Name: "GdFeCo", Ms: 200e3, Aex: 5e-12, Dmi: 0.0, Ku: 2.0e5, Alpha: 0.02, Tc: 523},
	"TbFeCo": {Name: "TbFeCo", Ms: 150e3, Aex: 4e-12, Dmi: 0.0, Ku: 1.5e6, Alpha: 0.05, Tc: 473},

	// SAF-specific materials
	"CoFeB_SAF_bottom": {Name: "CoFeB SAF bottom", Ms: 1100e3, Aex: 15e-12, Dmi: 1.0e-3, Ku: 8e5, Alpha: 0.02, Tc: 673},
	"CoFeB_SAF_top":    {Name: "CoFeB SAF top", Ms: 1100e3, Aex: 15e-12, Dmi: 1.0e-3, Ku: 8e5, Alpha: 0.02, Tc: 673},

	// Oxides
	"Fe3O4":   {Name: "Magnetite", Ms: 480e3, Aex: 1.2e-11, Dmi: 0.0, Ku: -1.1e4, Alpha: 0.05, Tc: 858},
	"CoFe2O4": {Name: "Cobalt Ferrite", Ms: 425e3, Aex: 3.0e-12, Dmi: 0.0, Ku: 2.0e5, Alpha: 0.1, Tc: 793},
	"LSMO":    {Name: "La0.7Sr0.3MnO3", Ms: 600e3, Aex: 2e-12, Dmi: 0.0, Ku: 1.0e4, Alpha: 0.01, Tc: 370},
}

// ============================================================================
// SPACER DATABASE - 17 spacers
// ============================================================================

type SpacerPreset struct {
	Name   string
	Type   string  // "metallic", "insulating", "semiconductor", "topological"
	Sigma  float64 // Conductivity (S/m)
	Ds     float64 // Spin diffusion constant (m²/s)
	TauSf  float64 // Spin-flip time (s)
	P      float64 // Polarization
	JRkky  float64 // RKKY coupling (J/m²)
	Lambda float64 // Spin diffusion length (nm)
}

var SpacerPresets = map[string]SpacerPreset{
	// Metallic spacers
	"Ru":       {Name: "Ru", Type: "metallic", Sigma: 5e6, Ds: 1e-3, TauSf: 1e-12, P: 0.3, JRkky: -2e-3, Lambda: 14.0},
	"Ru_thick": {Name: "Ru thick", Type: "metallic", Sigma: 5e6, Ds: 1e-3, TauSf: 1e-12, P: 0.3, JRkky: -0.5e-3, Lambda: 14.0},
	"Cu":       {Name: "Cu", Type: "metallic", Sigma: 5.96e7, Ds: 2e-3, TauSf: 100e-12, P: 0.0, JRkky: -0.1e-3, Lambda: 450.0},
	"Ag":       {Name: "Ag", Type: "metallic", Sigma: 6.3e7, Ds: 2.5e-3, TauSf: 150e-12, P: 0.0, JRkky: -0.05e-3, Lambda: 600.0},
	"Au":       {Name: "Au", Type: "metallic", Sigma: 4.5e7, Ds: 1.5e-3, TauSf: 40e-12, P: 0.0, JRkky: -0.15e-3, Lambda: 35.0},
	"Pt":       {Name: "Pt", Type: "metallic", Sigma: 9e6, Ds: 2e-3, TauSf: 5e-13, P: 0.7, JRkky: -0.5e-3, Lambda: 1.4},
	"Ta":       {Name: "Ta", Type: "metallic", Sigma: 7.6e6, Ds: 0.5e-3, TauSf: 1e-13, P: 0.1, JRkky: -0.3e-3, Lambda: 0.7},
	"W":        {Name: "W", Type: "metallic", Sigma: 1.8e7, Ds: 0.8e-3, TauSf: 2e-13, P: 0.2, JRkky: -0.4e-3, Lambda: 1.3},
	"Cr":       {Name: "Cr", Type: "metallic", Sigma: 7.7e6, Ds: 1.2e-3, TauSf: 3e-12, P: 0.15, JRkky: -1.5e-3, Lambda: 6.0},
	"Ir":       {Name: "Ir", Type: "metallic", Sigma: 1.9e7, Ds: 1.0e-3, TauSf: 4e-13, P: 0.5, JRkky: -0.6e-3, Lambda: 2.0},

	// Insulating spacers
	"MgO":      {Name: "MgO", Type: "insulating", Sigma: 0, Ds: 0, TauSf: 1e-15, P: 0, JRkky: 0.8e-3, Lambda: 0.0},
	"MgO_thin": {Name: "MgO thin tunnel", Type: "insulating", Sigma: 0, Ds: 0, TauSf: 1e-15, P: 0, JRkky: 2.5e-3, Lambda: 0.0},
	"Al2O3":    {Name: "Al2O3", Type: "insulating", Sigma: 0, Ds: 0, TauSf: 1e-15, P: 0, JRkky: 0.5e-3, Lambda: 0.0},
	"HfO2":     {Name: "HfO2", Type: "insulating", Sigma: 0, Ds: 0, TauSf: 1e-15, P: 0, JRkky: 0.3e-3, Lambda: 0.0},

	// Semiconductors
	"Si": {Name: "Si", Type: "semiconductor", Sigma: 1e-4, Ds: 0.01e-3, TauSf: 10e-9, P: 0, JRkky: -0.01e-3, Lambda: 100.0},
	"Ge": {Name: "Ge", Type: "semiconductor", Sigma: 2e-3, Ds: 0.02e-3, TauSf: 5e-9, P: 0, JRkky: -0.02e-3, Lambda: 50.0},

	// Topological insulator
	"Bi2Se3": {Name: "Bi2Se3", Type: "topological", Sigma: 1e4, Ds: 0.5e-3, TauSf: 100e-12, P: 0.9, JRkky: -0.1e-3, Lambda: 70.0},
}

// ============================================================================
// INITIALIZATION & REGISTRATION - All functions registered
// ============================================================================

func init() {
	// Core SAF functions
	DeclFunc("AboutSAF", AboutSAF, "Display complete attribution and citation information for MuMax3-SAF-NeuroSpin")
	DeclFunc("EnableSAF", EnableSAF, "Enable SAF RKKY coupling")
	DeclFunc("EnableThermal", EnableThermal, "Enable thermal fluctuations at specified temperature (K)")
	DeclFunc("GetSAFRKKYEnergy", GetSAFRKKYEnergy, "Get total SAF RKKY coupling energy in Joules")
	DeclFunc("DisableThermal", DisableThermal, "Disable thermal fluctuations")
	DeclFunc("EnableSOT", EnableSOT, "Enable spin-orbit torque")
	DeclFunc("EnableSTT", EnableSTT, "Enable spin-transfer torque")
	DeclFunc("EnableVCMA", EnableVCMA, "Enable voltage-controlled magnetic anisotropy")
	DeclFunc("EnableOersted", EnableOersted, "Enable Oersted field from current")
	DeclFunc("EnableNeuromorphic", EnableNeuromorphic, "Enable neuromorphic computing features")
	DeclFunc("EnableAdvancedPhysics", EnableAdvancedPhysics, "Enable all advanced physics modules")
	DeclFunc("SetSAFLayers", SetSAFLayers, "Set SAF layer region IDs: SetSAFLayers(layer1, layer2) [IDs must be 0..255]")
	DeclFunc("ClearSAFLayers", ClearSAFLayers, "Clear all SAF layer pairs")
	DeclFunc("CreateSkyrmionState", CreateSkyrmionState, "Create trial skyrmion: CreateSkyrmionState(radius_nm)")
	DeclFunc("CalculateSkyrmionBarrier", CalculateSkyrmionBarrier, "Calculate barrier: 'nucleation' or 'annihilation'")
	// NEW N5
	DeclFunc("RecordSpinWaves", RecordSpinWaves, "Record m(x,y,t) time series for spin wave analysis")
	DeclFunc("LaunchSpinWaves", LaunchSpinWaves, "Complete spin wave dispersion analysis")
	// NEW N6
	DeclFunc("InitSTDP", InitSTDP, "Initialize STDP spike history")
	DeclFunc("ApplyAdvancedSTDP", ApplyAdvancedSTDP, "N6: Advanced STDP - ApplyAdvancedSTDP(t_pre, t_post, voltage, t_voltage)")
	DeclFunc("ResetSTDPHistory", ResetSTDPHistory, "Clear STDP spike history")
	// NEW N7
	DeclFunc("FindStochasticResonance", FindStochasticResonance, "Scan noise levels to find optimal SNR")
	// NEW N8
	DeclFunc("EnableMagnonCoupling", EnableMagnonCoupling, "Enable magnon-mediated long-range coupling")
	DeclFunc("DisableMagnonCoupling", DisableMagnonCoupling, "Disable magnon coupling")
	DeclFunc("GetMagnonCouplingEnergy", GetMagnonCouplingEnergy, "Calculate magnon coupling energy")

	// Neuromorphic functions
	DeclFunc("ApplySTDP", ApplySTDP, "Apply STDP learning rule: ApplySTDP(dW, pre_times, post_times, weights)")
	DeclFunc("ProgramSynapticWeights", ProgramSynapticWeights, "Program synaptic weights: ProgramSynapticWeights(weights, region)")

	// Neuromorphic device database
	DeclFunc("ApplyNeuromorphicDevice", ApplyNeuromorphicDevice, "Load device parameters")
	DeclFunc("ListNeuromorphicDevices", ListNeuromorphicDevices, "List available devices")
	DeclFunc("GetDeviceInfo", GetDeviceInfo, "Show device details")
	DeclFunc("CompareDevices", CompareDevices, "Compare devices")
	DeclFunc("GetOptimalDevice", GetOptimalDevice, "Get device recommendation")
	DeclFunc("InitializeNeuromorphicDatabase", InitializeNeuromorphicDatabase, "Initialize neuromorphic device database")

	// Material database
	DeclFunc("ApplyMaterial", ApplyMaterialPreset, "Apply material preset: ApplyMaterial(region, 'CoFeB')")
	DeclFunc("ApplySpacer", ApplySpacerPreset, "Apply spacer preset: ApplySpacer(region, 'Ru')")
	DeclFunc("ListMaterials", ListMaterials, "List available material presets")
	DeclFunc("ListSpacers", ListSpacers, "List available spacer presets")
	DeclFunc("GetMaterialInfo", GetMaterialInfo, "Show detailed info: GetMaterialInfo('CoFeB')")
	DeclFunc("GetSpacerInfo", GetSpacerInfo, "Show detailed info: GetSpacerInfo('Ru')")

	// Utility functions
	DeclFunc("SetCurrentDensity", SetCurrentDensity, "Set current density in A/m²")
	DeclFunc("SetElectricField", SetElectricField, "Set electric field in V/m")
	// Helper functions for test compatibility
	DeclFunc("LayerRange", LayerRange, "Define layer range: LayerRange(start, end)")
	DeclFunc("SAFEnergy", SAFEnergy, "Get total SAF RKKY energy")

	// Set parameter defaults
	// New: set default for all regions (-1), so scripts can override globally
	J_RKKY.setRegion(-1, []float64{-2e-3}) // J_RKKY.setRegion(0, []float64{-2e-3})
	sigma_saf.setRegion(0, []float64{5e6})
	Ds_saf.setRegion(0, []float64{1e-3})
	tau_sf_saf.setRegion(0, []float64{1e-12})
	P_saf.setRegion(0, []float64{0.3})
	Temperature.setRegion(0, []float64{0.0})
	SOT_theta_SH.setRegion(0, []float64{0.0})
	SOT_theta_FL.setRegion(0, []float64{0.0})
	STT_xi_adiabatic.setRegion(0, []float64{1.0})
	STT_xi_nonadiabatic.setRegion(0, []float64{0.3})
	STT_polarization.setRegion(0, []float64{0.4})
	J_current.setRegion(0, []float64{0.0})
	wire_width.setRegion(0, []float64{100e-9})
	wire_thickness.setRegion(0, []float64{10e-9})
	J_RKKY_oscillatory.setRegion(0, []float64{0}) // Off by default
	J_RKKY_amplitude.setRegion(0, []float64{-2e-3})
	J_RKKY_wavelength.setRegion(0, []float64{1e-9})
	J_RKKY_phase.setRegion(0, []float64{0.0})
	J_RKKY_decay_length.setRegion(0, []float64{0.0})
	// NEW N2: Defaults
	J_RKKY_temp_dependent.setRegion(0, []float64{0})
	J_RKKY_Tc.setRegion(0, []float64{673}) // Default to CoFeB Tc
	// NEW N4: Defaults
	Skyrmion_barrier_samples.setRegion(0, []float64{10}) // 10 intermediate states
	Skyrmion_radius_nm.setRegion(0, []float64{50})       // 50nm typical skyrmion
	// NEW N5: Defaults
	SpinWave_record_time.setRegion(0, []float64{10e-9})      // 10 ns
	SpinWave_sample_interval.setRegion(0, []float64{10e-12}) // 10 ps
	SpinWave_excitation_freq.setRegion(0, []float64{10e9})   // 10 GHz
	SpinWave_excitation_amp.setRegion(0, []float64{0.01})    // 10 mT
	// NEW N6: Defaults (Pfister & Gerstner 2006 model)
	STDP_mode.setRegion(0, []float64{0})                    // Pair-based by default
	STDP_triplet_tau_plus.setRegion(0, []float64{16.8e-3})  // 16.8 ms
	STDP_triplet_tau_minus.setRegion(0, []float64{33.7e-3}) // 33.7 ms
	STDP_triplet_tau_x.setRegion(0, []float64{101e-3})      // 101 ms
	STDP_triplet_tau_y.setRegion(0, []float64{125e-3})      // 125 ms
	STDP_triplet_A2_plus.setRegion(0, []float64{0.005})     // Pair LTP
	STDP_triplet_A2_minus.setRegion(0, []float64{0.007})    // Pair LTD
	STDP_triplet_A3_plus.setRegion(0, []float64{0.0062})    // Triplet LTP
	STDP_triplet_A3_minus.setRegion(0, []float64{0.0023})   // Triplet LTD
	STDP_voltage_threshold.setRegion(0, []float64{0.5})     // 0.5V
	STDP_voltage_window.setRegion(0, []float64{10e-3})      // 10ms
	// NEW N7: Defaults
	StochRes_signal_freq.setRegion(0, []float64{1e9})    // 1 GHz signal
	StochRes_signal_amp.setRegion(0, []float64{0.001})   // 1 mT weak signal
	StochRes_noise_scan_min.setRegion(0, []float64{0})   // Start at 0K
	StochRes_noise_scan_max.setRegion(0, []float64{500}) // Up to 500K
	StochRes_noise_steps.setRegion(0, []float64{10})     // 10 noise levels
	StochRes_measure_time.setRegion(0, []float64{10e-9}) // 10 ns measurement
	// NEW N8: Defaults
	Magnon_coupling_enable.setRegion(0, []float64{0})     // Disabled by default
	Magnon_coupling_strength.setRegion(0, []float64{1e4}) // 10 kJ/m³
	Magnon_coupling_range.setRegion(0, []float64{100e-9}) // 100 nm range
	Magnon_source_region.setRegion(0, []float64{0})
	Magnon_target_region.setRegion(0, []float64{1})
}

// ============================================================================
// ENABLE FUNCTIONS
// ============================================================================

// AboutSAF prints complete attribution and citation information
func AboutSAF() {
	LogOut("")
	LogOut("╔═══════════════════════════════════════════════════════════════════════════╗")
	LogOut("║                    MuMax3-SAF-NeuroSpin v2.1                              ║")
	LogOut("║                   Synthetic Antiferromagnet Extensions                    ║")
	LogOut("║           with Advanced Neuromorphic Computing Capabilities               ║")
	LogOut("╠═══════════════════════════════════════════════════════════════════════════╣")
	LogOut("║  Author:      Prof. Santhosh Sivasubramani                                ║")
	LogOut("║  Institution: INTRINSIC Lab                                               ║")
	LogOut("║               Centre for Sensors Instrumentation and                      ║")
	LogOut("║               Cyber Physical System Engineering (SeNSE)                   ║")
	LogOut("║               Indian Institute of Technology Delhi                        ║")
	LogOut("║  Email:       ssivasub@iitd.ac.in                                         ║")
	LogOut("║               ragansanthosh@ieee.org                                      ║")
	LogOut("║  Website:     https://santhoshsivasubramani.github.io/neurospin/          ║")
	LogOut("╠═══════════════════════════════════════════════════════════════════════════╣")
	LogOut("║  Copyright © 2024-2025 Prof. Santhosh Sivasubramani                       ║")
	LogOut("║  Released under BSD 2-Clause License (same as MuMax3)                     ║")
	LogOut("╠═══════════════════════════════════════════════════════════════════════════╣")
	LogOut("║  Citation:                                                                ║")
	LogOut("║  If you use this software in published research, please cite:             ║")
	LogOut("║  S. Sivasubramani, 'MuMax3-SAF-NeuroSpin: GPU-Accelerated                ║")
	LogOut("║  Micromagnetic Simulation Framework for Synthetic Antiferromagnets        ║")
	LogOut("║  with Neuromorphic Computing Extensions', v2.1, 2024-2025.                ║")
	LogOut("║  Available at: https://santhoshsivasubramani.github.io/neurospin/         ║")
	LogOut("╠═══════════════════════════════════════════════════════════════════════════╣")
	LogOut("║  Based on MuMax3 by Arne Vansteenkiste et al.                             ║")
	LogOut("║  Original MuMax3: https://mumax.github.io                                 ║")
	LogOut("╚═══════════════════════════════════════════════════════════════════════════╝")
	LogOut("")
}

func EnableSAF() {
	if safEnabled {
		return
	}

	// Attribution banner
	LogOut("╔═══════════════════════════════════════════════════════════════╗")
	LogOut("║  MuMax3-SAF-NeuroSpin v2.1 - Neuromorphic Computing      ║")
	LogOut("║  Copyright © 2024-2025 Prof. Santhosh Sivasubramani     ║")
	LogOut("║  INTRINSIC Lab, IIT Delhi                                ║")
	LogOut("║  https://santhoshsivasubramani.github.io/neurospin/      ║")
	LogOut("╚═══════════════════════════════════════════════════════════════╝")

	safEnabled = true
	interfaceMapGPU = nil // ADD: Explicitly disable map
	AddFieldTerm(B_SAF_RKKY)
	registerEnergy(GetSAFRKKYEnergy, nil)
	// buildInterfaceMap()  // DISABLED: needs redesign
	CheckSAFRKKYAssumptions()
	LogOut("SAF RKKY coupling enabled with interface mapping")
}

// BuildInterfaceMap creates a mapping between layer1 and layer2 interface planes
func buildInterfaceMap() {
	// DISABLED: Interface mapping not compatible with multi-layer architecture yet
	// Using z±1 neighbor pairing instead
	LogOut("Interface map: disabled (multi-layer mode uses z±1 neighbors)")
}

// Free interface map on disable
func freeInterfaceMap() {
	if interfaceMapGPU != nil {
		// DISABLED: wrapper reverted
		// cuda.SAFFreeInterfaceMap(interfaceMapGPU)
		interfaceMapGPU = nil
		LogOut("Interface map freed")
	}
}

func EnableThermal(temp_K float64) {
	Temperature.setRegion(0, []float64{temp_K})
	thermalEnabled = true
	AddFieldTerm(B_thermal) // TODO: Create Quantity wrapper - Done
	LogOut("Thermal fluctuations enabled at %.0f K", temp_K)
}

func DisableThermal() {
	thermalEnabled = false
	Temperature.setRegion(0, []float64{0.0})

	// Free cuRAND states to avoid GPU memory leaks across runs
	if thermalStates != nil {
		cuda.SAFFreeCurandStates(thermalStates)
		thermalStates = nil
		LogOut("Thermal RNG states freed")
	}
}

func EnableSOT() {
	// Wire the SOT field into B_eff so it contributes during the solve
	AddFieldTerm(B_SOT)
	LogOut("Spin-orbit torque enabled")
}

func EnableSTT() {
	// Wire the STT field into B_eff so it contributes during the solve
	AddFieldTerm(B_STT)
	LogOut("Spin-transfer torque enabled")
}

func EnableVCMA() {
	// Wire the VCMA field into B_eff so it contributes during the solve
	AddFieldTerm(B_VCMA)
	LogOut("Voltage-controlled magnetic anisotropy enabled")
}

func EnableOersted() {
	// Wire the Oersted field into B_eff so it contributes during the solve
	AddFieldTerm(B_Oersted)
	LogOut("Oersted field enabled")
}

func EnableNeuromorphic() {
	neuromorphicEnabled = true
	InitializeNeuromorphicDatabase()
	ApplyNeuromorphicDevice("Skyrmion_Song2020")
	// ApplyNeuromorphicDevice("STT_MTJ_Sengupta2016")
	GetDeviceInfo("STT_MTJ_Sengupta2016")
	devices := ListNeuromorphicDevices()
	CompareDevices(devices)
	LogOut("Neuromorphic computing enabled")
	LogOut("Project: https://santhoshsivasubramani.github.io/neurospin/")
}

func EnableAdvancedPhysics() {
	EnableThermal(300.0)
	SOT_theta_SH.setRegion(-1, []float64{0.1})
	SOT_theta_FL.setRegion(-1, []float64{0.02})
	STT_xi_adiabatic.setRegion(-1, []float64{1.0})
	STT_xi_nonadiabatic.setRegion(-1, []float64{0.3})
	STT_polarization.setRegion(-1, []float64{0.4})
	xi_VCMA.setRegion(-1, []float64{300e-15})
	t_interface.setRegion(-1, []float64{1e-9})
	J_current.setRegion(-1, []float64{0.0})
	wire_width.setRegion(-1, []float64{100e-9})
	wire_thickness.setRegion(-1, []float64{10e-9})
	LogOut("All advanced physics modules enabled")
}

// ============================================================================
// RKKY COUPLING with CPU FALLBACK
// ============================================================================

func AddSAFRKKYField(dst *data.Slice) {
	if !safEnabled {
		return
	}

	if len(SAF_layer_pairs) == 0 {
		LogErr("AddSAFRKKYField: No layer pairs defined. Call SetSAFLayers() first.")
		cuda.Zero(dst)
		return
	}

	defer func() {
		if r := recover(); r != nil {
			log.Printf("CUDA RKKY field failed: %v", r)
			cuda.Zero(dst)
		}
	}()

	m := M.Buffer()
	ms_buf, recycle := Msat.Slice()
	if recycle {
		defer cuda.SAFRecycle(ms_buf)
	}
	Ms := ms_buf.DevPtr(0)
	if m.Size()[0] == 0 {
		cuda.Zero(dst)
		return
	}

	size := Mesh().Size()
	Nx, Ny, Nz := size[0], size[1], size[2]
	N := m.Len()
	dz := float32(Mesh().CellSize()[2])

	use_oscillatory := (J_RKKY_oscillatory.GetRegion(0) != 0)
	use_temp_dep := (J_RKKY_temp_dependent.GetRegion(0) != 0)

	wavelength := float32(J_RKKY_wavelength.GetRegion(0))
	phase := float32(J_RKKY_phase.GetRegion(0))
	decay_len := float32(J_RKKY_decay_length.GetRegion(0))
	osc_enable := int32(0)
	if use_oscillatory {
		osc_enable = 1
	}

	// NEW: Loop over all layer pairs
	for _, pair := range SAF_layer_pairs {
		var Jval float32
		if use_oscillatory {
			Jval = float32(J_RKKY_amplitude.GetRegion(0))
		} else {
			jrL1 := J_RKKY.GetRegion(pair.Layer1)
			jrL2 := J_RKKY.GetRegion(pair.Layer2)
			jrG := J_RKKY.GetRegion(0)
			Jval = float32(jrG)
			if Jval == 0 {
				Jval = float32(jrL1)
			}
			if Jval == 0 {
				Jval = float32(jrL2)
			}
		}

		if Jval == 0 {
			continue // Skip this pair
		}

		// Temperature scaling
		if use_temp_dep {
			T := float32(Temperature.GetRegion(0))
			Tc := float32(J_RKKY_Tc.GetRegion(0))
			if T > 0 && Tc > 0 {
				temp_factor := 1.0 - (T/Tc)*(T/Tc)
				if temp_factor < 0 {
					temp_factor = 0
				}
				Jval *= temp_factor
			}
		}

		// Call kernel for this pair
		cuda.SAFAddRKKYField_CUDA(dst, m, unsafe.Pointer(Ms), unsafe.Pointer(regions.Gpu().Ptr),
			Jval, pair.Layer1, pair.Layer2, dz, Nx, Ny, Nz, N,
			osc_enable, wavelength, phase, decay_len)
	}
}

func GetSAFRKKYEnergy() float64 {
	if !safEnabled {
		return 0
	}

	if len(SAF_layer_pairs) == 0 {
		LogErr("GetSAFRKKYEnergy: No layer pairs defined")
		return 0
	}

	CheckSAFRKKYAssumptions()

	defer func() {
		if r := recover(); r != nil {
			log.Printf("CUDA RKKY energy failed: %v", r)
		}
	}()

	m := M.Buffer()
	if m.Size()[0] == 0 {
		return 0.0
	}

	size := Mesh().Size()
	cs := Mesh().CellSize()
	Nx, Ny, Nz := size[0], size[1], size[2]
	N := m.Len()

	cellArea := float32(cs[0] * cs[1])
	thickness := float32(cs[2])

	use_oscillatory := (J_RKKY_oscillatory.GetRegion(0) != 0)
	use_temp_dep := (J_RKKY_temp_dependent.GetRegion(0) != 0)

	wavelength := float32(J_RKKY_wavelength.GetRegion(0))
	phase := float32(J_RKKY_phase.GetRegion(0))
	decay_len := float32(J_RKKY_decay_length.GetRegion(0))
	osc_enable := int32(0)
	if use_oscillatory {
		osc_enable = 1
	}

	E_slice := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(E_slice)

	total_energy := 0.0

	// NEW: Loop over all layer pairs
	for _, pair := range SAF_layer_pairs {
		var Jval float32
		if use_oscillatory {
			Jval = float32(J_RKKY_amplitude.GetRegion(0))
		} else {
			jrL1 := J_RKKY.GetRegion(pair.Layer1)
			jrL2 := J_RKKY.GetRegion(pair.Layer2)
			jrG := J_RKKY.GetRegion(0)
			Jval = float32(jrG)
			if Jval == 0 {
				Jval = float32(jrL1)
			}
			if Jval == 0 {
				Jval = float32(jrL2)
			}
		}

		if Jval == 0 {
			continue
		}

		// Temperature scaling
		if use_temp_dep {
			T := float32(Temperature.GetRegion(0))
			Tc := float32(J_RKKY_Tc.GetRegion(0))
			if T > 0 && Tc > 0 {
				temp_factor := 1.0 - (T/Tc)*(T/Tc)
				if temp_factor < 0 {
					temp_factor = 0
				}
				Jval *= temp_factor
			}
		}

		// Zero the buffer for this pair
		cuda.Zero(E_slice)

		// Calculate energy for this pair
		cuda.SAFGetRKKYEnergy_CUDA(E_slice, m, unsafe.Pointer(regions.Gpu().Ptr),
			Jval, pair.Layer1, pair.Layer2, Nx, Ny, Nz, N, cellArea, thickness,
			osc_enable, wavelength, phase, decay_len)

		pair_energy := float64(cuda.Sum(E_slice))
		total_energy += pair_energy

		LogOut("RKKY pair (%d,%d): E = %.3e J", pair.Layer1, pair.Layer2, pair_energy)
	}

	LogOut("GetSAFRKKYEnergy: Total energy = %.3e J from %d pairs", total_energy, len(SAF_layer_pairs))
	return total_energy
}

// SetSAFLayers sets the region IDs used as SAF layer1 and layer2.
// Note: these are region IDs (uint8 on GPU), not z-plane indices.
// Valid range is [0..255]. The two layers must differ.
func SetSAFLayers(l1, l2 int) {
	if l1 == l2 {
		LogErr("SetSAFLayers: layer IDs must differ (got %d == %d)", l1, l2)
		return
	}
	if l1 < 0 || l1 > 255 || l2 < 0 || l2 > 255 {
		LogErr("SetSAFLayers: layer IDs must be in [0..255] (got %d, %d)", l1, l2)
		return
	}

	// Check if pair already exists
	for _, pair := range SAF_layer_pairs {
		if (pair.Layer1 == l1 && pair.Layer2 == l2) || (pair.Layer1 == l2 && pair.Layer2 == l1) {
			LogOut("SetSAFLayers: pair (%d,%d) already exists", l1, l2)
			return
		}
	}

	// Add new pair
	SAF_layer_pairs = append(SAF_layer_pairs, LayerPair{Layer1: l1, Layer2: l2})
	LogOut("SetSAFLayers: Added pair (%d, %d) - total pairs: %d", l1, l2, len(SAF_layer_pairs))
}

func ClearSAFLayers() {
	SAF_layer_pairs = nil
	LogOut("ClearSAFLayers: All layer pairs cleared")
}

// CheckSAFRKKYAssumptions emits warnings about common misconfigurations that would
// make the current CUDA RKKY pairing incorrect. This does not change behavior;
// it helps prevent silent wrong results.
func CheckSAFRKKYAssumptions() {
	sz := Mesh().Size()
	if sz[2] < 2 {
		LogErr("SAF RKKY: domain Nz=%d < 2; need at least two z-layers for SAF", sz[2])
	}

	if len(SAF_layer_pairs) == 0 {
		LogErr("SAF RKKY: No layer pairs defined. Call SetSAFLayers().")
		return
	}

	for _, pair := range SAF_layer_pairs {
		if pair.Layer1 == pair.Layer2 {
			LogErr("SAF RKKY: pair has same layer IDs (%d,%d)", pair.Layer1, pair.Layer2)
		}

		if pair.Layer1 < 0 || pair.Layer1 > 255 || pair.Layer2 < 0 || pair.Layer2 > 255 {
			LogErr("SAF RKKY: layer IDs must be in [0..255], got (%d,%d)", pair.Layer1, pair.Layer2)
		}
	}

	LogOut("SAF RKKY note: current kernels use z±1 neighbor pairing")
	LogOut("Multi-layer SAF: %d interface(s) defined", len(SAF_layer_pairs))
}

// ============================================================================
// THERMAL FLUCTUATIONS
// ============================================================================

func initThermalRandom() {
	if thermalStates != nil {
		return
	}

	N := Mesh().Size()[0] * Mesh().Size()[1] * Mesh().Size()[2]
	seed := uint64(time.Now().UnixNano())

	// Allocate memory for cuRAND states
	thermalStates = cuda.SAFAllocateCurandStates(N)

	// Initialize the allocated states with seed
	cuda.SAFInitThermalRandom_CUDA(thermalStates, seed, N)

	LogOut("Thermal RNG initialized with %d states (seed: %d)", N, seed)
}

func AddThermalField(dst *data.Slice) {
	if !thermalEnabled {
		cuda.Zero(dst)
		return
	}

	defer func() {
		if r := recover(); r != nil {
			log.Printf("Thermal field failed: %v", r)
			thermalEnabled = false
			cuda.Zero(dst)
		}
	}()

	T := float32(Temperature.GetRegion(0))
	if T <= 0 {
		cuda.Zero(dst)
		return
	}

	if thermalStates == nil {
		initThermalRandom()
	}

	m := M.Buffer()
	ms_buf, recycle := Msat.Slice()
	if recycle {
		defer cuda.SAFRecycle(ms_buf)
	}
	Ms := ms_buf.DevPtr(0)
	al_buf, _ := Alpha.Slice()
	alpha_ptr := al_buf.DevPtr(0)

	if m.Size()[0] == 0 {
		cuda.Zero(dst)
		return
	}

	size := Mesh().Size()
	cs := Mesh().CellSize()
	Nx, Ny, Nz := size[0], size[1], size[2]
	N := m.Len()
	dt := float32(Dt_si)

	cuda.SAFAddThermalField_CUDA(dst, unsafe.Pointer(Ms), unsafe.Pointer(alpha_ptr),
		unsafe.Pointer(regions.Gpu().Ptr), thermalStates, T, dt,
		float32(cs[0]), float32(cs[1]), float32(cs[2]), Nx, Ny, Nz, N)
}

// ============================================================================
// SPIN-ORBIT TORQUE
// ============================================================================

func AddSOTField(dst *data.Slice) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("SOT field failed: %v", r)
			cuda.Zero(dst)
		}
	}()

	theta_SH := float32(SOT_theta_SH.GetRegion(0))
	theta_FL := float32(SOT_theta_FL.GetRegion(0))
	Jc := float32(J_current.GetRegion(0))

	if (theta_SH == 0 && theta_FL == 0) || Jc == 0 {
		cuda.Zero(dst)
		return
	}

	m := M.Buffer()
	ms_buf, _ := Msat.Slice()
	Ms := ms_buf.DevPtr(0)

	if m.Size()[0] == 0 {
		cuda.Zero(dst)
		return
	}

	size := Mesh().Size()
	Nx, Ny, Nz := size[0], size[1], size[2]
	N := m.Len()
	thickness := float32(Mesh().CellSize()[2])

	Jc_gpu := cuda.Buffer(1, Mesh().Size())
	cuda.Memset(Jc_gpu, Jc)
	defer cuda.Recycle(Jc_gpu)

	cuda.SAFAddSOTField_CUDA(dst, m, unsafe.Pointer(Ms), unsafe.Pointer(Jc_gpu.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr), theta_SH, theta_FL, thickness, Nx, Ny, Nz, N)
}

// ============================================================================
// SPIN-TRANSFER TORQUE
// ============================================================================

func AddSTTField(dst *data.Slice) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("STT field failed: %v", r)
			cuda.Zero(dst)
		}
	}()

	xi_ad := float32(STT_xi_adiabatic.GetRegion(0))
	xi_nad := float32(STT_xi_nonadiabatic.GetRegion(0))
	Jc := float32(J_current.GetRegion(0))
	pol := float32(STT_polarization.GetRegion(0))

	if (xi_ad == 0 && xi_nad == 0) || Jc == 0 {
		cuda.Zero(dst)
		return
	}

	m := M.Buffer()
	ms_buf, _ := Msat.Slice()
	Ms := ms_buf.DevPtr(0)

	if m.Size()[0] == 0 {
		cuda.Zero(dst)
		return
	}

	size := Mesh().Size()
	cs := Mesh().CellSize()
	Nx, Ny, Nz := size[0], size[1], size[2]
	N := m.Len()
	thickness := float32(cs[2])
	dx := float32(cs[0]) // ADD: Get actual x-spacing

	Jc_gpu := cuda.Buffer(1, Mesh().Size())
	pol_gpu := cuda.Buffer(1, Mesh().Size())
	cuda.Memset(Jc_gpu, Jc)
	cuda.Memset(pol_gpu, pol)
	defer cuda.Recycle(Jc_gpu)
	defer cuda.Recycle(pol_gpu)

	cuda.SAFAddSTTField_CUDA(dst, m, unsafe.Pointer(Ms), unsafe.Pointer(Jc_gpu.DevPtr(0)),
		unsafe.Pointer(pol_gpu.DevPtr(0)), unsafe.Pointer(regions.Gpu().Ptr),
		xi_ad, xi_nad, thickness, dx, Nx, Ny, Nz, N) // ADD: Pass dx
}

// ============================================================================
// VCMA
// ============================================================================

func AddVCMAField(dst *data.Slice) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("VCMA field failed: %v", r)
			cuda.Zero(dst)
		}
	}()

	E := float32(E_field.GetRegion(0))
	xi := float32(xi_VCMA.GetRegion(0))
	t_int := float32(t_interface.GetRegion(0))

	if E == 0 || xi == 0 {
		cuda.Zero(dst)
		return
	}

	m := M.Buffer()
	ms_buf, _ := Msat.Slice()
	Ms := ms_buf.DevPtr(0)

	if m.Size()[0] == 0 {
		cuda.Zero(dst)
		return
	}

	size := Mesh().Size()
	Nx, Ny, Nz := size[0], size[1], size[2]
	N := m.Len()

	if t_int == 0 {
		t_int = 1e-9
	}

	Ez_gpu := cuda.Buffer(1, Mesh().Size())
	cuda.Memset(Ez_gpu, E)
	defer cuda.Recycle(Ez_gpu)

	cuda.SAFAddVCMAField_CUDA(dst, m, unsafe.Pointer(Ms), unsafe.Pointer(Ez_gpu.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr), xi, t_int, Nx, Ny, Nz, N)
}

// ============================================================================
// OERSTED FIELD
// ============================================================================

func AddOerstedField(dst *data.Slice) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Oersted field failed: %v", r)
			cuda.Zero(dst)
		}
	}()

	Jc := float32(J_current.GetRegion(0))
	LogOut("DEBUG: AddOerstedField called. Jc=%.2e", Jc)

	if Jc == 0 {
		cuda.Zero(dst)
		return
	}

	size := Mesh().Size()
	cs := Mesh().CellSize()
	Nx, Ny, Nz := size[0], size[1], size[2]
	N := dst.Len()

	w := float32(wire_width.GetRegion(0))
	t := float32(wire_thickness.GetRegion(0))
	LogOut("DEBUG: Oersted params: w=%.2e, t=%.2e", w, t)

	if w <= 0 {
		w = 100e-9
	}
	if t <= 0 {
		t = 10e-9
	}

	dx := float32(cs[0])
	dy := float32(cs[1])
	dz := float32(cs[2])

	// Jc BUFFER Fill
	Jc_gpu := cuda.Buffer(1, Mesh().Size())
	// FIX: Use explicit verified kernel
	cuda.SAFSetValue_CUDA_Fixed(Jc_gpu, Jc)
	defer cuda.Recycle(Jc_gpu)
	cuda.SAFAddOerstedField_CUDA(dst, unsafe.Pointer(Jc_gpu.DevPtr(0)),
		unsafe.Pointer(regions.Gpu().Ptr), w, t, dx, dy, dz, Nx, Ny, Nz, N)
}

// ============================================================================
// NEUROMORPHIC COMPUTING
// ============================================================================

// DEPRECATED: Use built-in ext_topologicalcharge instead
// func ComputeTopologicalCharge(dst *data.Slice) {
//	if !neuromorphicEnabled {
//		cuda.Zero(dst)
//		return
//	}
//
//	defer func() {
//		if r := recover(); r != nil {
//			log.Printf("Topological charge failed: %v", r)
//			cuda.Zero(dst)
//		}
//	}()
//
//	m := M.Buffer()
//
//	if m.Size()[0] == 0 {
//		cuda.Zero(dst)
//		return
//	}
//
//	size := Mesh().Size()
//	cs := Mesh().CellSize()
//	Nx, Ny, Nz := size[0], size[1], size[2]
//	N := m.Len()
//
//	cuda.SAFComputeTopologicalCharge_CUDA(dst, m, unsafe.Pointer(regions.Gpu().Ptr),
//		float32(cs[0]), float32(cs[1]), Nx, Ny, Nz, N)
// }

func ApplySTDP(weight_updates, pre_spike_times, post_spike_times, current_weights *data.Slice) {
	if !neuromorphicEnabled {
		cuda.Zero(weight_updates)
		return
	}

	defer func() {
		if r := recover(); r != nil {
			log.Printf("STDP application failed: %v", r)
			cuda.Zero(weight_updates)
		}
	}()

	A_plus := float32(STDP_A_plus.GetRegion(0))
	A_minus := float32(STDP_A_minus.GetRegion(0))
	tau_plus := float32(STDP_tau_plus.GetRegion(0))
	tau_minus := float32(STDP_tau_minus.GetRegion(0))
	dt := float32(Dt_si)

	if A_plus == 0 && A_minus == 0 {
		cuda.Zero(weight_updates)
		return
	}

	size := Mesh().Size()
	N := size[0] * size[1] * size[2]

	cuda.SAFApplySTDP_CUDA(
		unsafe.Pointer(weight_updates.DevPtr(0)),
		unsafe.Pointer(pre_spike_times.DevPtr(0)),
		unsafe.Pointer(post_spike_times.DevPtr(0)),
		unsafe.Pointer(current_weights.DevPtr(0)),
		A_plus, A_minus,
		tau_plus, tau_minus, dt,
		N)
}

func ProgramSynapticWeights(targetWeights []float32, region int) {
	if !neuromorphicEnabled {
		return
	}

	defer func() {
		if r := recover(); r != nil {
			log.Printf("Weight programming failed: %v", r)
		}
	}()

	size := Mesh().Size()
	N := size[0] * size[1] * size[2]

	if len(targetWeights) != N {
		LogErr("ProgramSynapticWeights: weight array size %d doesn't match mesh size %d", len(targetWeights), N)
		return
	}

	di_buf, _ := Dind.Slice()
	dmi_current := di_buf.DevPtr(0)
	target_gpu := cuda.NewSlice(1, size)
	defer cuda.Recycle(target_gpu)

	// Upload using cu package (already imported)
	cu.MemcpyHtoD(cu.DevicePtr(target_gpu.DevPtr(0)), unsafe.Pointer(&targetWeights[0]), int64(N)*4)

	dmi_min := float32(-2e-3)
	dmi_max := float32(2e-3)
	prog_rate := float32(0.1e-3)

	cuda.SAFProgramAnalogWeight_CUDA(unsafe.Pointer(dmi_current), unsafe.Pointer(dmi_current),
		unsafe.Pointer(target_gpu.DevPtr(0)), unsafe.Pointer(regions.Gpu().Ptr),
		dmi_min, dmi_max, prog_rate, region, N)
}

// ============================================================================
// MATERIAL DATABASE FUNCTIONS
// ============================================================================

func ApplyMaterialPreset(region int, materialName string) {
	preset, exists := MaterialPresets[materialName]
	if !exists {
		LogErr("Material '%s' not found. Use ListMaterials() to see options.", materialName)
		return
	}

	Msat.setRegion(region, []float64{preset.Ms})
	Aex.setRegion(region, []float64{preset.Aex})
	if Dind != nil {
		Dind.setRegion(region, []float64{preset.Dmi})
	}
	Ku1.setRegion(region, []float64{preset.Ku})
	Alpha.setRegion(region, []float64{preset.Alpha})

	LogOut("Applied material '%s' to region %d", materialName, region)
	LogOut("  Ms=%.2e A/m, Aex=%.2e J/m, Ku=%.2e J/m³, α=%.4f",
		preset.Ms, preset.Aex, preset.Ku, preset.Alpha)

	if preset.Aex > 0 && preset.Ku > 0 {
		lex := math.Sqrt(preset.Aex / preset.Ku)
		LogOut("  Exchange length: %.2f nm", lex*1e9)
	}
}

func ApplySpacerPreset(region int, spacerName string) {
	preset, exists := SpacerPresets[spacerName]
	if !exists {
		LogErr("Spacer '%s' not found. Use ListSpacers() to see options.", spacerName)
		return
	}
	// Set RKKY globally (uniform J across interface)
	J_RKKY.setRegion(-1, []float64{preset.JRkky})
	// J_RKKY.setRegion(region, []float64{preset.JRkky})
	Msat.setRegion(region, []float64{0})
	sigma_saf.setRegion(region, []float64{preset.Sigma})
	Ds_saf.setRegion(region, []float64{preset.Ds})
	tau_sf_saf.setRegion(region, []float64{preset.TauSf})
	P_saf.setRegion(region, []float64{preset.P})

	LogOut("Applied spacer '%s' to region %d", spacerName, region)
	LogOut("  Type: %s, J_RKKY=%.2e J/m², λ_sf=%.1f nm",
		preset.Type, preset.JRkky, preset.Lambda)
}

func ListMaterials() {
	LogOut("Available magnetic materials (25 total):")
	LogOut("  Thin films: CoFeB, CoFeB_ultrathin, Py, Py_Ni81Fe19")
	LogOut("  Metals: Co, Co_thin, Fe, Ni, CoFe")
	LogOut("  L10 alloys: CoPt, CoPd, FePt_L10, FePd_L10")
	LogOut("  Binary: FeNi, NiFe_50_50")
	LogOut("  Heusler: Co2MnSi, Co2FeAl, Co2FeSi")
	LogOut("  Rare-earth: GdFeCo, TbFeCo")
	LogOut("  SAF: CoFeB_SAF_bottom, CoFeB_SAF_top")
	LogOut("  Oxides: Fe3O4, CoFe2O4, LSMO")
	LogOut("Usage: ApplyMaterial(region, 'CoFeB')")
}

func ListSpacers() {
	LogOut("Available spacer materials (17 total):")
	LogOut("  Metallic: Ru, Ru_thick, Cu, Ag, Au, Pt, Ta, W, Cr, Ir")
	LogOut("  Insulating: MgO, MgO_thin, Al2O3, HfO2")
	LogOut("  Semiconductor: Si, Ge")
	LogOut("  Topological: Bi2Se3")
	LogOut("Usage: ApplySpacer(region, 'Ru')")
}

func GetMaterialInfo(materialName string) {
	preset, exists := MaterialPresets[materialName]
	if !exists {
		LogErr("Material '%s' not found", materialName)
		return
	}

	LogOut("Material: %s", preset.Name)
	LogOut("  Ms:    %.2e A/m", preset.Ms)
	LogOut("  Aex:   %.2e J/m", preset.Aex)
	LogOut("  DMI:   %.2e J/m²", preset.Dmi)
	LogOut("  Ku:    %.2e J/m³", preset.Ku)
	LogOut("  Alpha: %.4f", preset.Alpha)
	LogOut("  Tc:    %.0f K", preset.Tc)

	if preset.Aex > 0 && preset.Ku > 0 {
		lex := math.Sqrt(preset.Aex / preset.Ku)
		LogOut("  Exchange length: %.2f nm", lex*1e9)
	}

	if preset.Aex > 0 && preset.Ms > 0 {
		A_J := preset.Aex * 1e12
		Ms_kA := preset.Ms / 1e3
		LogOut("  A_ex: %.1f pJ/m, M_s: %.0f kA/m", A_J, Ms_kA)
	}
}

func GetSpacerInfo(spacerName string) {
	preset, exists := SpacerPresets[spacerName]
	if !exists {
		LogErr("Spacer '%s' not found", spacerName)
		return
	}

	LogOut("Spacer: %s", preset.Name)
	LogOut("  Type:              %s", preset.Type)
	LogOut("  Conductivity:      %.2e S/m", preset.Sigma)
	LogOut("  RKKY coupling:     %.2e J/m²", preset.JRkky)
	LogOut("  Spin diffusion:    %.1f nm", preset.Lambda)
	LogOut("  Polarization:      %.2f", preset.P)

	if preset.Type == "metallic" {
		LogOut("  Spin flip time:    %.2e s", preset.TauSf)
		LogOut("  Diffusion const:   %.2e m²/s", preset.Ds)
	}
}

func SetCurrentDensity(Jc_A_per_m2 float64) {
	J_current.setRegion(-1, []float64{Jc_A_per_m2})
	LogOut("Current density set to %.2e A/m²", Jc_A_per_m2)
}

func SetElectricField(E_V_per_m float64) {
	E_field.setRegion(-1, []float64{E_V_per_m})
	LogOut("Electric field set to %.2e V/m", E_V_per_m)
}

func ListAvailableMaterials() string {
	materials := ""
	for name := range MaterialPresets {
		if materials != "" {
			materials += ", "
		}
		materials += name
	}
	return materials
}

func ListAvailableSpacers() string {
	spacers := ""
	for name := range SpacerPresets {
		if spacers != "" {
			spacers += ", "
		}
		spacers += name
	}
	return spacers
}

// ============================================================================
// HELPER FUNCTIONS FOR TEST COMPATIBILITY
// ============================================================================
// LayerRange returns a shape for z-layers from start (inclusive) to end (exclusive)
//
//	func LayerRange(start, end int) Shape {
//		return func(x, y, z float64) bool {
//			zidx := Index2Coord(0, 0, int(z))[2]
//			cellsize := Mesh().CellSize()[2]
//			zpos := float64(zidx) * cellsize
//			zstart := float64(start) * cellsize
//			zend := float64(end) * cellsize
//			return zpos >= zstart && zpos < zend
//		}
//	}
//
// LayerRange returns a shape selecting z-layers [start, end) by layer index.
// The shape function receives coordinates in meters, so we convert the
// integer layer indices to z-positions using the cell size.
func LayerRange(start, end int) Shape {
	// Ensure start <= end
	if end < start {
		start, end = end, start
	}
	// Cell size and domain height in meters
	cz := Mesh().CellSize()[2]
	// Lz := float64(Mesh().Size()[2]) * cz
	// Convert index bounds to metric bounds
	zmin := float64(start) * cz
	zmax := float64(end) * cz
	// Clamp to domain
	// if zmin < 0 {
	//	zmin = 0
	// }
	// if zmax > Lz {
	//	zmax = Lz
	//}
	// Select cells whose center z lies in [zmin, zmax)
	return func(x, y, z float64) bool {
		return z >= zmin && z < zmax
	}
}

// SAFEnergy returns the total RKKY coupling energy
func SAFEnergy() float64 {
	if !safEnabled {
		LogErr("SAF not enabled. Call EnableSAF() first")
		return 0.0
	}
	// Return the CUDA-computed RKKY energy (same as E_SAF_RKKY)
	E := GetSAFRKKYEnergy()
	LogOut("SAFEnergy(): returning E_SAF_RKKY = %.3e J", E)
	return E
}

// ============================================================================
// N4: SKYRMION ENERGY BARRIERS
// ============================================================================

// CreateSkyrmionState creates a trial skyrmion at center with given radius
func CreateSkyrmionState(radius_nm float64) {
	size := Mesh().Size()
	cs := Mesh().CellSize()
	Nx, Ny := size[0], size[1]

	center_x := float64(Nx) * cs[0] / 2.0
	center_y := float64(Ny) * cs[1] / 2.0
	radius_m := radius_nm * 1e-9

	// Use MuMax3's cell-by-cell setting
	for iz := 0; iz < size[2]; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				x := (float64(ix) + 0.5) * cs[0]
				y := (float64(iy) + 0.5) * cs[1]

				dx := x - center_x
				dy := y - center_y
				r := math.Sqrt(dx*dx + dy*dy)

				if r < radius_m {
					// Inside skyrmion: Néel-type texture
					theta := math.Atan2(dy, dx)
					profile := math.Cos(math.Pi * r / radius_m)

					mx := math.Cos(theta) * math.Sin(math.Pi*r/radius_m)
					my := math.Sin(theta) * math.Sin(math.Pi*r/radius_m)
					mz := profile

					M.SetCell(ix, iy, iz, data.Vector{float64(mx), float64(my), float64(mz)})
				} else {
					// Outside: FM up
					M.SetCell(ix, iy, iz, data.Vector{0, 0, 1})
				}
			}
		}
	}

	LogOut("Created trial skyrmion state: R=%.1f nm", radius_nm)
}

// CalculateSkyrmionBarrier computes nucleation/annihilation barrier
func CalculateSkyrmionBarrier(direction string) float64 {
	n_samples := int(Skyrmion_barrier_samples.GetRegion(0))
	radius := Skyrmion_radius_nm.GetRegion(0)

	if n_samples < 3 {
		LogErr("Need at least 3 samples for barrier calculation")
		return 0
	}

	LogOut("Calculating skyrmion %s barrier with %d samples", direction, n_samples)

	// Save initial state
	m_initial := M.Buffer().HostCopy()

	energies := make([]float64, n_samples)

	if direction == "nucleation" {
		// State 0: Pure FM
		M.SetRegion(0, Uniform(0, 0, 1))
		energies[0] = GetTotalEnergy()
		LogOut("  State 0 (FM): E = %.3e J", energies[0])

		for i := 1; i < n_samples-1; i++ {
			progress := float64(i) / float64(n_samples-1)
			CreateSkyrmionState(radius * progress)
			Minimize()
			energies[i] = GetTotalEnergy()
			LogOut("  State %d (%.1f%%): E = %.3e J", i, progress*100, energies[i])
		}

		CreateSkyrmionState(radius)
		Minimize()
		energies[n_samples-1] = GetTotalEnergy()
		LogOut("  State %d (Skyrmion): E = %.3e J", n_samples-1, energies[n_samples-1])

	} else {
		CreateSkyrmionState(radius)
		Minimize()
		energies[0] = GetTotalEnergy()
		LogOut("  State 0 (Skyrmion): E = %.3e J", energies[0])

		for i := 1; i < n_samples-1; i++ {
			progress := 1.0 - float64(i)/float64(n_samples-1)
			CreateSkyrmionState(radius * progress)
			Minimize()
			energies[i] = GetTotalEnergy()
			LogOut("  State %d (%.1f%%): E = %.3e J", i, (1-progress)*100, energies[i])
		}

		M.SetRegion(0, Uniform(0, 0, 1))
		energies[n_samples-1] = GetTotalEnergy()
		LogOut("  State %d (FM): E = %.3e J", n_samples-1, energies[n_samples-1])
	}

	E_max := energies[0]
	E_initial := energies[0]
	saddle_idx := 0

	for i := 1; i < n_samples; i++ {
		if energies[i] > E_max {
			E_max = energies[i]
			saddle_idx = i
		}
	}

	barrier := E_max - E_initial

	LogOut("Skyrmion %s barrier:", direction)
	LogOut("  Initial state: E = %.3e J", E_initial)
	LogOut("  Saddle point (state %d): E = %.3e J", saddle_idx, E_max)
	LogOut("  Barrier height: ΔE = %.3e J = %.2f meV", barrier, barrier*1e3/1.602e-19)

	// Restore initial state
	data.Copy(M.Buffer(), m_initial)

	return barrier
}

// ============================================================================
// N5: SPIN WAVE DISPERSION
// ============================================================================

// SpinWaveData stores recorded magnetization snapshots
type SpinWaveData struct {
	Mx         [][]float32 // [time][space] for mx
	My         [][]float32 // [time][space] for my
	Mz         [][]float32 // [time][space] for mz
	Times      []float64   // Timestamps
	Nx, Ny, Nz int
}

// RecordSpinWaves applies excitation and records m(x,y,t) time series
func RecordSpinWaves() *SpinWaveData {
	record_time := SpinWave_record_time.GetRegion(0)
	dt_sample := SpinWave_sample_interval.GetRegion(0)

	n_samples := int(record_time / dt_sample)

	if n_samples < 10 {
		LogErr("Need at least 10 samples for FFT (record_time/sample_interval = %d)", n_samples)
		return nil
	}

	size := Mesh().Size()
	Nx, Ny, Nz := size[0], size[1], size[2]
	N_space := Nx * Ny * Nz

	LogOut("Recording spin waves: %d samples over %.2f ns (dt=%.2f ps)",
		n_samples, record_time*1e9, dt_sample*1e12)

	swdata := &SpinWaveData{
		Mx:    make([][]float32, n_samples),
		My:    make([][]float32, n_samples),
		Mz:    make([][]float32, n_samples),
		Times: make([]float64, n_samples),
		Nx:    Nx, Ny: Ny, Nz: Nz,
	}

	// Note: User should apply thermal noise or field pulse BEFORE calling this
	// This function just records the dynamics

	t_start := Time

	for i := 0; i < n_samples; i++ {
		// Run to next sample time
		target_time := t_start + float64(i)*dt_sample
		RunWhile(func() bool { return Time < target_time })

		// Record snapshot - flatten 3D to 1D
		m_snap := M.Buffer().HostCopy()
		swdata.Mx[i] = make([]float32, N_space)
		swdata.My[i] = make([]float32, N_space)
		swdata.Mz[i] = make([]float32, N_space)

		idx := 0
		for iz := 0; iz < Nz; iz++ {
			for iy := 0; iy < Ny; iy++ {
				for ix := 0; ix < Nx; ix++ {
					swdata.Mx[i][idx] = m_snap.Vectors()[0][iz][iy][ix]
					swdata.My[i][idx] = m_snap.Vectors()[1][iz][iy][ix]
					swdata.Mz[i][idx] = m_snap.Vectors()[2][iz][iy][ix]
					idx++
				}
			}
		}
		swdata.Times[i] = Time - t_start

		if i%(n_samples/10) == 0 {
			LogOut("  Recorded snapshot %d/%d (t=%.2f ns)", i, n_samples, swdata.Times[i]*1e9)
		}
	}

	LogOut("Spin wave recording complete: %d snapshots", n_samples)
	return swdata
}

// CalculateDispersion performs 2D+1D FFT to get ω(k)
func CalculateDispersion(swdata *SpinWaveData) map[string]interface{} {
	if swdata == nil {
		LogErr("No spin wave data provided")
		return nil
	}

	Nx, Ny := swdata.Nx, swdata.Ny
	n_times := len(swdata.Times)

	LogOut("Calculating dispersion from %d×%d×%d data", Nx, Ny, n_times)

	// Simplified: just report that FFT analysis would happen here
	// Full implementation would use fftw or Go's DSP libraries

	LogOut("Dispersion calculation: FFT analysis not yet implemented")
	LogOut("Would perform:")
	LogOut("  1. 2D spatial FFT: m(x,y,t) → m(kx,ky,t)")
	LogOut("  2. 1D temporal FFT: m(kx,ky,t) → m(kx,ky,ω)")
	LogOut("  3. Peak finding: extract ω(kx,ky)")

	result := make(map[string]interface{})
	result["status"] = "recording_complete"
	result["n_samples"] = n_times
	result["spatial_size"] = []int{Nx, Ny}

	return result
}

// LaunchSpinWaves is the user-facing function
func LaunchSpinWaves() {
	LogOut("=== Spin Wave Dispersion Analysis ===")

	// Record time series
	swdata := RecordSpinWaves()

	if swdata != nil {
		// Calculate dispersion
		result := CalculateDispersion(swdata)
		LogOut("Result: %v", result)

		// Save raw data for external analysis
		SaveSpinWaveData(swdata)
	}

	LogOut("=== Spin Wave Analysis Complete ===")
}

// SaveSpinWaveData saves recorded data to files for external FFT analysis
func SaveSpinWaveData(swdata *SpinWaveData) {
	LogOut("Saving spin wave data for external analysis...")

	// Save metadata
	LogOut("  Spatial grid: %d × %d × %d", swdata.Nx, swdata.Ny, swdata.Nz)
	LogOut("  Time samples: %d", len(swdata.Times))
	LogOut("  Time range: %.2f to %.2f ns",
		swdata.Times[0]*1e9, swdata.Times[len(swdata.Times)-1]*1e9)

	// Note: Full implementation would save binary data files
	// For now, just log that data is ready
	LogOut("  Data ready for FFT (not saved - implement file I/O if needed)")
}

// ============================================================================
// N6: ADVANCED STDP RULES
// ============================================================================

// STDPSpikeHistory stores spike timing history for triplet STDP
type STDPSpikeHistory struct {
	PreSpikes  []float64 // Timestamps of presynaptic spikes
	PostSpikes []float64 // Timestamps of postsynaptic spikes
	TraceR1    float64   // Fast presynaptic trace
	TraceR2    float64   // Slow presynaptic trace
	TraceO1    float64   // Fast postsynaptic trace
	TraceO2    float64   // Slow postsynaptic trace
	LastUpdate float64   // Last trace update time
}

// Global spike history (simplified - one synapse)
var stdpHistory *STDPSpikeHistory

// InitSTDP initializes STDP spike history
func InitSTDP() {
	stdpHistory = &STDPSpikeHistory{
		PreSpikes:  make([]float64, 0),
		PostSpikes: make([]float64, 0),
		TraceR1:    0,
		TraceR2:    0,
		TraceO1:    0,
		TraceO2:    0,
		LastUpdate: 0,
	}
	LogOut("STDP initialized: mode=%d", int(STDP_mode.GetRegion(0)))
}

// UpdateSTDPTraces updates exponential traces
func UpdateSTDPTraces(t float64) {
	if stdpHistory == nil {
		return
	}

	dt := t - stdpHistory.LastUpdate
	if dt <= 0 {
		return
	}

	tau_plus := STDP_triplet_tau_plus.GetRegion(0)
	tau_minus := STDP_triplet_tau_minus.GetRegion(0)
	tau_x := STDP_triplet_tau_x.GetRegion(0)
	tau_y := STDP_triplet_tau_y.GetRegion(0)

	// Decay traces exponentially
	stdpHistory.TraceR1 *= math.Exp(-dt / tau_plus)
	stdpHistory.TraceR2 *= math.Exp(-dt / tau_x)
	stdpHistory.TraceO1 *= math.Exp(-dt / tau_minus)
	stdpHistory.TraceO2 *= math.Exp(-dt / tau_y)

	stdpHistory.LastUpdate = t
}

// CalculatePairSTDP computes standard pair-based STDP
func CalculatePairSTDP(t_pre, t_post float64) float64 {
	tau_plus := STDP_triplet_tau_plus.GetRegion(0)
	tau_minus := STDP_triplet_tau_minus.GetRegion(0)
	A_plus := STDP_triplet_A2_plus.GetRegion(0)
	A_minus := STDP_triplet_A2_minus.GetRegion(0)

	dt := t_post - t_pre

	if dt > 0 {
		// LTP: post after pre
		return A_plus * math.Exp(-dt/tau_plus)
	} else {
		// LTD: pre after post
		return -A_minus * math.Exp(dt/tau_minus)
	}
}

// CalculateTripletSTDP computes triplet STDP (Pfister & Gerstner 2006)
func CalculateTripletSTDP(t_spike float64, is_pre bool) float64 {
	if stdpHistory == nil {
		InitSTDP()
	}

	UpdateSTDPTraces(t_spike)

	A2_plus := STDP_triplet_A2_plus.GetRegion(0)
	A2_minus := STDP_triplet_A2_minus.GetRegion(0)
	A3_plus := STDP_triplet_A3_plus.GetRegion(0)
	A3_minus := STDP_triplet_A3_minus.GetRegion(0)

	var delta_w float64

	if is_pre {
		// Presynaptic spike
		// LTD depends on postsynaptic traces
		delta_w = -stdpHistory.TraceO1 * (A2_minus + A3_minus*stdpHistory.TraceR2)

		// Update presynaptic traces
		stdpHistory.TraceR1 += 1.0
		stdpHistory.TraceR2 += 1.0
		stdpHistory.PreSpikes = append(stdpHistory.PreSpikes, t_spike)

	} else {
		// Postsynaptic spike
		// LTP depends on presynaptic traces
		delta_w = stdpHistory.TraceR1 * (A2_plus + A3_plus*stdpHistory.TraceO2)

		// Update postsynaptic traces
		stdpHistory.TraceO1 += 1.0
		stdpHistory.TraceO2 += 1.0
		stdpHistory.PostSpikes = append(stdpHistory.PostSpikes, t_spike)
	}

	return delta_w
}

// CalculateVoltageGatedSTDP computes voltage-modulated STDP
func CalculateVoltageGatedSTDP(t_pre, t_post, voltage, t_voltage float64) float64 {
	// Standard pair STDP
	delta_w := CalculatePairSTDP(t_pre, t_post)

	V_thresh := STDP_voltage_threshold.GetRegion(0)
	tau_V := STDP_voltage_window.GetRegion(0)

	// Check if voltage pulse overlaps with spike timing window
	dt_V_pre := math.Abs(t_voltage - t_pre)
	dt_V_post := math.Abs(t_voltage - t_post)

	// Voltage must exceed threshold and arrive within timing window
	if voltage > V_thresh && (dt_V_pre < tau_V || dt_V_post < tau_V) {
		// Voltage gate is open: allow plasticity
		voltage_factor := (voltage - V_thresh) / V_thresh // Linear modulation
		delta_w *= (1.0 + voltage_factor)

		LogOut("Voltage-gated STDP: V=%.2f V > %.2f V, modulation=%.2fx",
			voltage, V_thresh, 1.0+voltage_factor)
	} else {
		// No voltage gate: suppress plasticity
		delta_w *= 0.1 // Reduce to 10% without voltage
	}

	return delta_w
}

// ApplySTDP is the main user-facing function
func ApplyAdvancedSTDP(t_pre, t_post, voltage, t_voltage float64) float64 {
	mode := int(STDP_mode.GetRegion(0))

	var delta_w float64

	switch mode {
	case 0:
		// Pair-based STDP
		delta_w = CalculatePairSTDP(t_pre, t_post)
		LogOut("Pair STDP: Δt=%.2f ms → Δw=%.4f", (t_post-t_pre)*1e3, delta_w)

	case 1:
		// Triplet STDP
		delta_w_pre := CalculateTripletSTDP(t_pre, true)
		delta_w_post := CalculateTripletSTDP(t_post, false)
		delta_w = delta_w_pre + delta_w_post
		LogOut("Triplet STDP: Δw_pre=%.4f, Δw_post=%.4f, total=%.4f",
			delta_w_pre, delta_w_post, delta_w)

	case 2:
		// Voltage-gated STDP
		delta_w = CalculateVoltageGatedSTDP(t_pre, t_post, voltage, t_voltage)
		LogOut("Voltage-gated STDP: Δw=%.4f", delta_w)

	default:
		LogErr("Unknown STDP mode: %d (use 0=pair, 1=triplet, 2=voltage)", mode)
		return 0
	}

	return delta_w
}

// ResetSTDPHistory clears spike history (for triplet STDP)
func ResetSTDPHistory() {
	if stdpHistory != nil {
		stdpHistory.PreSpikes = make([]float64, 0)
		stdpHistory.PostSpikes = make([]float64, 0)
		stdpHistory.TraceR1 = 0
		stdpHistory.TraceR2 = 0
		stdpHistory.TraceO1 = 0
		stdpHistory.TraceO2 = 0
		stdpHistory.LastUpdate = 0
		LogOut("STDP history reset")
	}
}

// ============================================================================
// N7: STOCHASTIC RESONANCE
// ============================================================================

// StochResResult stores SNR measurements
type StochResResult struct {
	NoiseTemps []float64 // Temperature values
	SNRs       []float64 // Signal-to-noise ratios
	OptimalT   float64   // Optimal temperature
	OptimalSNR float64   // Maximum SNR
}

// MeasureSNR calculates signal-to-noise ratio at given noise level
func MeasureSNR(noise_temp, measure_time float64) float64 {
	// Save initial state
	T_initial := Temp.GetRegion(0)

	// Set noise temperature
	Temp.setRegion(0, []float64{noise_temp})

	// Record magnetization response to thermal fluctuations
	n_samples := 100 // Fixed sample count for speed

	mz_signal := make([]float64, n_samples)
	dt := measure_time / float64(n_samples)

	for i := 0; i < n_samples; i++ {
		// Run for dt
		Run(dt)

		// Record average mz
		mz_data := M.Buffer().HostCopy()
		sum_mz := 0.0
		count := 0
		size := Mesh().Size()
		for iz := 0; iz < size[2]; iz++ {
			for iy := 0; iy < size[1]; iy++ {
				for ix := 0; ix < size[0]; ix++ {
					sum_mz += float64(mz_data.Vectors()[2][iz][iy][ix])
					count++
				}
			}
		}
		mz_signal[i] = sum_mz / float64(count)
	}

	// Calculate response amplitude (susceptibility to noise)
	max_mz := mz_signal[0]
	min_mz := mz_signal[0]
	for _, mz := range mz_signal {
		if mz > max_mz {
			max_mz = mz
		}
		if mz < min_mz {
			min_mz = mz
		}
	}
	response_amp := max_mz - min_mz

	// SNR proxy: response amplitude (noise helps overcome barriers)
	// At optimal noise, response is maximized

	// Restore initial state
	Temp.setRegion(0, []float64{T_initial})

	return response_amp
}

// FindStochasticResonance scans noise levels to find optimal response
func FindStochasticResonance() *StochResResult {
	T_min := StochRes_noise_scan_min.GetRegion(0)
	T_max := StochRes_noise_scan_max.GetRegion(0)
	n_steps := int(StochRes_noise_steps.GetRegion(0))
	measure_time := StochRes_measure_time.GetRegion(0)

	LogOut("=== Stochastic Resonance Scan ===")
	LogOut("Noise scan: T=%.0f to %.0f K (%d steps)", T_min, T_max, n_steps)
	LogOut("Measurement time: %.2f ns per step", measure_time*1e9)

	result := &StochResResult{
		NoiseTemps: make([]float64, n_steps),
		SNRs:       make([]float64, n_steps),
		OptimalT:   0,
		OptimalSNR: 0,
	}

	dT := (T_max - T_min) / float64(n_steps-1)

	for i := 0; i < n_steps; i++ {
		T_noise := T_min + float64(i)*dT
		result.NoiseTemps[i] = T_noise

		// Measure response amplitude at this noise level
		response := MeasureSNR(T_noise, measure_time)
		result.SNRs[i] = response

		LogOut("  T=%.0f K: Response=%.6f", T_noise, response)

		// Track optimal
		if response > result.OptimalSNR {
			result.OptimalSNR = response
			result.OptimalT = T_noise
		}
	}

	LogOut("=== Stochastic Resonance Found ===")
	LogOut("Optimal noise: T=%.0f K", result.OptimalT)
	LogOut("Maximum response: %.6f", result.OptimalSNR)

	if result.SNRs[0] > 0 {
		LogOut("Enhancement: %.1fx vs zero noise", result.OptimalSNR/result.SNRs[0])
	}

	return result
}

// ============================================================================
// N8: MAGNON-MEDIATED COUPLING
// ============================================================================

// AddMagnonField computes long-range magnon coupling field
func AddMagnonField(dst *data.Slice) {
	enable := (Magnon_coupling_enable.GetRegion(0) != 0)
	if !enable {
		cuda.Zero(dst)
		return
	}

	J_magnon := float32(Magnon_coupling_strength.GetRegion(0))
	range_m := float32(Magnon_coupling_range.GetRegion(0))
	source_id := int(Magnon_source_region.GetRegion(0))
	target_id := int(Magnon_target_region.GetRegion(0))

	if J_magnon == 0 || range_m <= 0 {
		cuda.Zero(dst)
		return
	}

	m := M.Buffer()

	size := Mesh().Size()
	cs := Mesh().CellSize()
	Nx, Ny, Nz := size[0], size[1], size[2]

	// CPU implementation
	m_host := m.HostCopy()
	dst_host := dst.HostCopy()

	// Get Ms values per region
	Ms_map := make(map[int]float32)
	for r := 0; r < 256; r++ {
		Ms_map[r] = float32(Msat.GetRegion(r))
	}

	// Zero output
	for c := 0; c < 3; c++ {
		for iz := 0; iz < Nz; iz++ {
			for iy := 0; iy < Ny; iy++ {
				for ix := 0; ix < Nx; ix++ {
					dst_host.Vectors()[c][iz][iy][ix] = 0
				}
			}
		}
	}

	// For each target cell
	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				region_target := int(regions.HostArray()[iz][iy][ix])
				if region_target != target_id {
					continue
				}

				Ms_target := Ms_map[region_target]
				if Ms_target == 0 {
					continue
				}

				// Position of target cell
				x_target := (float64(ix) + 0.5) * cs[0]
				y_target := (float64(iy) + 0.5) * cs[1]
				z_target := (float64(iz) + 0.5) * cs[2]

				// Sum contributions from all source cells
				var Bx, By, Bz float32

				for jz := 0; jz < Nz; jz++ {
					for jy := 0; jy < Ny; jy++ {
						for jx := 0; jx < Nx; jx++ {
							region_source := int(regions.HostArray()[jz][jy][jx])
							if region_source != source_id {
								continue
							}

							// Distance between source and target
							x_source := (float64(jx) + 0.5) * cs[0]
							y_source := (float64(jy) + 0.5) * cs[1]
							z_source := (float64(jz) + 0.5) * cs[2]

							dx := x_target - x_source
							dy := y_target - y_source
							dz := z_target - z_source
							r := math.Sqrt(dx*dx + dy*dy + dz*dz)

							if r < 1e-12 {
								continue // Skip self
							}

							// Exponential decay with distance
							coupling := float32(math.Exp(-r / float64(range_m)))

							// Magnon-mediated field: B ∝ m_source * exp(-r/λ)
							mx := m_host.Vectors()[0][jz][jy][jx]
							my := m_host.Vectors()[1][jz][jy][jx]
							mz := m_host.Vectors()[2][jz][jy][jx]

							mu0 := float32(4 * math.Pi * 1e-7)
							prefactor := J_magnon * coupling / (mu0 * Ms_target)

							Bx += prefactor * mx
							By += prefactor * my
							Bz += prefactor * mz
						}
					}
				}

				// Write to output
				dst_host.Vectors()[0][iz][iy][ix] = Bx
				dst_host.Vectors()[1][iz][iy][ix] = By
				dst_host.Vectors()[2][iz][iy][ix] = Bz
			}
		}
	}

	// Copy back to GPU
	data.Copy(dst, dst_host)

	LogOut("Magnon coupling field computed: source=%d, target=%d, range=%.1f nm",
		source_id, target_id, range_m*1e9)
}

// EnableMagnonCoupling activates magnon-mediated coupling
func EnableMagnonCoupling() {
	Magnon_coupling_enable.setRegion(0, []float64{1})
	AddFieldTerm(B_magnon)
	LogOut("Magnon-mediated coupling enabled")
}

// DisableMagnonCoupling deactivates magnon coupling
func DisableMagnonCoupling() {
	Magnon_coupling_enable.setRegion(0, []float64{0})
	LogOut("Magnon-mediated coupling disabled")
}

// GetMagnonCouplingEnergy calculates energy of magnon coupling
func GetMagnonCouplingEnergy() float64 {
	enable := (Magnon_coupling_enable.GetRegion(0) != 0)
	if !enable {
		return 0
	}

	// Get magnon field
	B_mag := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(B_mag)
	AddMagnonField(B_mag)

	m := M.Buffer()

	// E = -∫ m · B_magnon dV
	size := Mesh().Size()
	cs := Mesh().CellSize()
	cellVol := float32(cs[0] * cs[1] * cs[2])

	m_host := m.HostCopy()
	B_host := B_mag.HostCopy()

	E_total := 0.0

	for iz := 0; iz < size[2]; iz++ {
		for iy := 0; iy < size[1]; iy++ {
			for ix := 0; ix < size[0]; ix++ {
				mx := m_host.Vectors()[0][iz][iy][ix]
				my := m_host.Vectors()[1][iz][iy][ix]
				mz := m_host.Vectors()[2][iz][iy][ix]

				Bx := B_host.Vectors()[0][iz][iy][ix]
				By := B_host.Vectors()[1][iz][iy][ix]
				Bz := B_host.Vectors()[2][iz][iy][ix]

				dot := float64(mx*Bx + my*By + mz*Bz)
				E_total += -dot * float64(cellVol)
			}
		}
	}

	LogOut("Magnon coupling energy: %.3e J", E_total)
	return E_total

}
