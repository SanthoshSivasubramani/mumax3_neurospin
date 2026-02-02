// MuMax3-SAF-NeuroSpin Go CUDA Bindings
//
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
// Provides Go ↔ CUDA bridge for SAF and neuromorphic kernels.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L${SRCDIR} -lsaf_wrapper -L/usr/local/cuda/lib64 -lcudart -lcurand

#include "saf_wrapper_cu.h"
#include "saf_v2_wrapper_cu.h"
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// SAFAddRKKYField_CUDA calls RKKY coupling kernel
func SAFAddRKKYField_CUDA(dst, m *data.Slice, Ms unsafe.Pointer, regions unsafe.Pointer,
	J_rkky float32, layer1, layer2 int, thickness float32,
	Nx, Ny, Nz, N int,
	osc_enable int32, wavelength, phase, decay_len float32) {

	C.k_addRKKYField_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.uint8_t)(regions),
		C.float(J_rkky),
		C.int(layer1), C.int(layer2),
		C.float(thickness),
		C.int(Nx), C.int(Ny), C.int(Nz), C.int(N),
		C.int(osc_enable), C.float(wavelength), C.float(phase), C.float(decay_len),
		nil)
}

func SAFGetRKKYEnergy_CUDA(E_out, m *data.Slice, regions unsafe.Pointer,
	J_rkky float32, layer1, layer2 int,
	Nx, Ny, Nz, N int, cellArea, thickness float32,
	osc_enable int32, wavelength, phase, decay_len float32) {

	C.k_getRKKYEnergy_async(
		(*C.float)(unsafe.Pointer(E_out.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.uint8_t)(regions),
		C.float(J_rkky),
		C.int(layer1), C.int(layer2),
		C.int(Nx), C.int(Ny), C.int(Nz), C.int(N),
		C.float(cellArea), C.float(thickness),
		C.int(osc_enable), C.float(wavelength), C.float(phase), C.float(decay_len),
		nil)
}

// SAFInitThermalRandom_CUDA initializes cuRAND states
func SAFInitThermalRandom_CUDA(states unsafe.Pointer, seed uint64, N int) {
	C.k_initThermalRandom_async(
		(*C.curandState)(states),
		C.ulong(seed),
		C.int(N),
		nil)
}

// SAFAddThermalField_CUDA calls thermal field kernel
func SAFAddThermalField_CUDA(dst *data.Slice, Ms, alpha unsafe.Pointer, regions unsafe.Pointer,
	states unsafe.Pointer, temperature, dt float32,
	dx, dy, dz float32, Nx, Ny, Nz, N int) {

	C.k_addThermalField_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(Ms),
		(*C.float)(alpha),
		(*C.uint8_t)(regions),
		(*C.curandState)(states),
		C.float(temperature), C.float(dt),
		C.float(dx), C.float(dy), C.float(dz),
		C.int(Nx), C.int(Ny), C.int(Nz), C.int(N),
		nil)
}

// SAFAddSOTField_CUDA calls spin-orbit torque kernel
func SAFAddSOTField_CUDA(dst, m *data.Slice, Ms, Jc unsafe.Pointer, regions unsafe.Pointer,
	theta_SH, theta_FL, thickness float32, Nx, Ny, Nz, N int) {

	C.k_addSOTField_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.float)(Jc),
		(*C.uint8_t)(regions),
		C.float(theta_SH), C.float(theta_FL), C.float(thickness),
		C.int(Nx), C.int(Ny), C.int(Nz), C.int(N),
		nil)
}

// SAFAddSTTField_CUDA calls spin-transfer torque kernel
func SAFAddSTTField_CUDA(dst, m *data.Slice, Ms, Jc, pol unsafe.Pointer, regions unsafe.Pointer,
	xi_ad, xi_nad, thickness, dx float32, Nx, Ny, Nz, N int) {

	C.k_addSTTField_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.float)(Jc),
		(*C.float)(pol),
		(*C.uint8_t)(regions),
		C.float(xi_ad), C.float(xi_nad), C.float(thickness), C.float(dx),
		C.int(Nx), C.int(Ny), C.int(Nz), C.int(N),
		nil)
}

// SAFAddVCMAField_CUDA calls VCMA kernel
func SAFAddVCMAField_CUDA(dst, m *data.Slice, Ms, Ez unsafe.Pointer, regions unsafe.Pointer,
	xi_vcma, t_interface float32, Nx, Ny, Nz, N int) {

	C.k_addVCMAField_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.float)(Ez),
		(*C.uint8_t)(regions),
		C.float(xi_vcma), C.float(t_interface),
		C.int(Nx), C.int(Ny), C.int(Nz), C.int(N),
		nil)
}

// SAFAddOerstedField_CUDA calls Oersted field kernel
func SAFAddOerstedField_CUDA(dst *data.Slice, Jc unsafe.Pointer, regions unsafe.Pointer,
	wire_width, wire_thickness, dx, dy, dz float32, Nx, Ny, Nz, N int) {

	C.k_addOerstedField_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(Jc),
		(*C.uint8_t)(regions),
		C.float(wire_width), C.float(wire_thickness),
		C.float(dx), C.float(dy), C.float(dz),
		C.int(Nx), C.int(Ny), C.int(Nz), C.int(N),
		nil)
}

// SAFSetValue_CUDA sets float array to value
func SAFSetValue_CUDA(dst *data.Slice, val float32) {
	N := dst.Len()
	C.k_setValue_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		C.float(val),
		C.int(N),
		nil)
}

// SAFComputeTopologicalCharge_CUDA calls topological charge kernel
func SAFComputeTopologicalCharge_CUDA(Q_out, m *data.Slice, regions unsafe.Pointer,
	dx, dy float32, Nx, Ny, Nz, N int) {

	C.k_computeTopologicalCharge_async(
		(*C.float)(unsafe.Pointer(Q_out.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.uint8_t)(regions),
		C.float(dx), C.float(dy),
		C.int(Nx), C.int(Ny), C.int(Nz), C.int(N),
		nil)
}

// SAFProgramAnalogWeight_CUDA calls weight programming kernel
func SAFProgramAnalogWeight_CUDA(target_param, current_param, target_weights unsafe.Pointer,
	regions unsafe.Pointer, param_min, param_max, programming_rate float32,
	region_id, N int) {

	C.k_programAnalogWeight_async(
		(*C.float)(target_param),
		(*C.float)(current_param),
		(*C.float)(target_weights),
		(*C.uint8_t)(regions),
		C.float(param_min), C.float(param_max), C.float(programming_rate),
		C.int(region_id), C.int(N),
		nil)
}

// SAFApplySTDP_CUDA calls STDP learning kernel
func SAFApplySTDP_CUDA(weight_updates, pre_spikes, post_spikes, current_weights unsafe.Pointer,
	A_plus, A_minus, tau_plus, tau_minus, dt float32, N int) {

	C.k_applySTDP_async(
		(*C.float)(weight_updates),
		(*C.float)(pre_spikes),
		(*C.float)(post_spikes),
		(*C.float)(current_weights),
		C.float(A_plus), C.float(A_minus),
		C.float(tau_plus), C.float(tau_minus), C.float(dt),
		C.int(N),
		nil)
}

// SAFAccumulateRegionSums_CUDA calls region-wise magnetization accumulation kernel
// This computes total magnetization components and spin counts per region.
func SAFAccumulateRegionSums_CUDA(mx, my, mz *data.Slice, regions unsafe.Pointer,
	mxSum, mySum, mzSum, count unsafe.Pointer, N, regionID int) {

	C.k_accumulateRegionSums_async(
		(*C.float)(unsafe.Pointer(mx.DevPtr(0))),
		(*C.float)(unsafe.Pointer(my.DevPtr(0))),
		(*C.float)(unsafe.Pointer(mz.DevPtr(0))),
		(*C.uint8_t)(regions),
		C.int(N), C.int(regionID),
		(*C.float)(mxSum), (*C.float)(mySum), (*C.float)(mzSum), (*C.int)(count),
		nil)
}

// SAFAllocateCurandStates allocates device memory for cuRAND states
func SAFAllocateCurandStates(N int) unsafe.Pointer {
	return unsafe.Pointer(C.allocate_curand_states(C.int(N)))
}

// SAFFreeCurandStates frees cuRAND state memory
func SAFFreeCurandStates(states unsafe.Pointer) {
	C.free_curand_states((*C.curandState)(states))
}

// ============================================================================
// V2 CUDA KERNEL WRAPPERS (12 New Functions)
// ============================================================================

// V2.1: Multi-neighbor RKKY
func SAFAddMultiNeighborRKKY_CUDA(dst, m *data.Slice, Ms, regions unsafe.Pointer,
	J1, J2, J3, J_lateral float32, layer1, layer2, n_neighbors int, thickness float32,
	Nx, Ny, Nz, N int) {

	C.k_addMultiNeighborRKKY_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.uint8_t)(regions),
		C.float(J1), C.float(J2), C.float(J3), C.float(J_lateral),
		C.int(layer1), C.int(layer2), C.int(n_neighbors),
		C.float(thickness), C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.2: Interlayer DMI
func SAFAddInterlayerDMI_CUDA(dst, m *data.Slice, Ms, regions unsafe.Pointer,
	D_inter float32, layer1, layer2 int, dx, dy, dz float32,
	Nx, Ny, Nz, N int) {

	C.k_addInterlayerDMI_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.uint8_t)(regions),
		C.float(D_inter), C.int(layer1), C.int(layer2),
		C.float(dx), C.float(dy), C.float(dz),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.3: Non-collinear RKKY
func SAFAddNonCollinearRKKY_CUDA(dst, m *data.Slice, Ms, regions unsafe.Pointer,
	J_ncol, alpha_ncol float32, layer1, layer2 int, thickness float32,
	Nx, Ny, Nz, N int) {

	C.k_addNonCollinearRKKY_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.uint8_t)(regions),
		C.float(J_ncol), C.float(alpha_ncol),
		C.int(layer1), C.int(layer2), C.float(thickness),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.4: Spin diffusion solver
func SAFSolveSpinDiffusion_CUDA(mu_s *data.Slice, Jc, P, lambda_sf unsafe.Pointer,
	dz float32, Nx, Ny, Nz, maxIter int) {

	C.k_solveSpinDiffusion_async(
		(*C.float)(unsafe.Pointer(mu_s.DevPtr(0))),
		(*C.float)(unsafe.Pointer(mu_s.DevPtr(1))),
		(*C.float)(unsafe.Pointer(mu_s.DevPtr(2))),
		(*C.float)(Jc),
		(*C.float)(P),
		(*C.float)(lambda_sf),
		C.float(dz), C.int(Nx), C.int(Ny), C.int(Nz), C.int(maxIter), nil)
}

// V2.4b: Spin diffusion field
func SAFAddSpinDiffusionField_CUDA(dst, m, mu_s *data.Slice, Ms unsafe.Pointer,
	theta_SH, thickness float32, Nx, Ny, Nz, N int) {

	C.k_addSpinDiffusionField_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(unsafe.Pointer(mu_s.DevPtr(0))),
		(*C.float)(unsafe.Pointer(mu_s.DevPtr(1))),
		(*C.float)(unsafe.Pointer(mu_s.DevPtr(2))),
		(*C.float)(Ms),
		C.float(theta_SH), C.float(thickness),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.5: Stochastic STDP
func SAFApplyStochasticSTDP_CUDA(weights, pre, post *data.Slice, states unsafe.Pointer,
	A_plus, A_minus, tau_plus, tau_minus, noise_sigma, retention_tau, dt float32, N int) {

	C.k_applyStochasticSTDP_async(
		(*C.float)(unsafe.Pointer(weights.DevPtr(0))),
		(*C.float)(unsafe.Pointer(pre.DevPtr(0))),
		(*C.float)(unsafe.Pointer(post.DevPtr(0))),
		(*C.curandState)(states),
		C.float(A_plus), C.float(A_minus),
		C.float(tau_plus), C.float(tau_minus),
		C.float(noise_sigma), C.float(retention_tau), C.float(dt),
		C.int(N), nil)
}

// V2.6: Reservoir computing
func SAFUpdateReservoirState_CUDA(state, input, W_res *data.Slice,
	leak_rate, spectral_radius float32, N, N_inputs int) {

	C.k_updateReservoirState_async(
		(*C.float)(unsafe.Pointer(state.DevPtr(0))),
		(*C.float)(unsafe.Pointer(input.DevPtr(0))),
		(*C.float)(unsafe.Pointer(W_res.DevPtr(0))),
		C.float(leak_rate), C.float(spectral_radius),
		C.int(N), C.int(N_inputs), nil)
}

// V2.7: Metaplasticity
func SAFApplyMetaplasticity_CUDA(weights, thresh, activity *data.Slice,
	theta_target, tau_meta, dt float32, N int) {

	C.k_applyMetaplasticity_async(
		(*C.float)(unsafe.Pointer(weights.DevPtr(0))),
		(*C.float)(unsafe.Pointer(thresh.DevPtr(0))),
		(*C.float)(unsafe.Pointer(activity.DevPtr(0))),
		C.float(theta_target), C.float(tau_meta), C.float(dt),
		C.int(N), nil)
}

// V2.8: Heat diffusion
func SAFSolveHeatDiffusion_CUDA(T_new, T_old *data.Slice, Jc, sigma, rho_cp unsafe.Pointer,
	kappa, dt, dx, dy, dz float32, Nx, Ny, Nz, N int) {

	C.k_solveHeatDiffusion_async(
		(*C.float)(unsafe.Pointer(T_new.DevPtr(0))),
		(*C.float)(unsafe.Pointer(T_old.DevPtr(0))),
		(*C.float)(Jc),
		(*C.float)(sigma),
		(*C.float)(rho_cp),
		C.float(kappa), C.float(dt), C.float(dx), C.float(dy), C.float(dz),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.9: Nonlinear VCMA
func SAFAddNonlinearVCMA_CUDA(dst, m *data.Slice, Ms, Ez unsafe.Pointer,
	xi1, xi2, xi3, t_int float32, Nx, Ny, Nz, N int) {

	C.k_addNonlinearVCMA_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.float)(Ez),
		C.float(xi1), C.float(xi2), C.float(xi3), C.float(t_int),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.10: Magnon-phonon coupling
func SAFAddMagnonPhononCoupling_CUDA(dst, m *data.Slice, Ms, strain_xx, strain_yy unsafe.Pointer,
	B1, B2 float32, Nx, Ny, Nz, N int) {

	C.k_addMagnonPhononCoupling_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.float)(strain_xx),
		(*C.float)(strain_yy),
		C.float(B1), C.float(B2),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.11: Quantum tunneling
func SAFCalculateQuantumTMR_CUDA(TMR, m *data.Slice,
	t_barrier, phi_barrier, P1, P2 float32, Nx, Ny, Nz, N int) {

	C.k_calculateQuantumTMR_async(
		(*C.float)(unsafe.Pointer(TMR.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		C.float(t_barrier), C.float(phi_barrier), C.float(P1), C.float(P2),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// ============================================================================
// V2 FEATURES 13-25 WRAPPERS
// ============================================================================

// V2.13: Temperature-dependent RKKY
func SAFAddTemperatureDependentRKKY_CUDA(dst, m *data.Slice, Ms, regions unsafe.Pointer,
	T_field *data.Slice, J0_base, T_curie, temp_exponent float32,
	layer1, layer2 int, thickness float32, Nx, Ny, Nz, N int) {

	C.k_addTemperatureDependentRKKY_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.uchar)(regions),
		(*C.float)(unsafe.Pointer(T_field.DevPtr(0))),
		C.float(J0_base), C.float(T_curie), C.float(temp_exponent),
		C.int(layer1), C.int(layer2), C.float(thickness),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.14: Spin wave FFT
func SAFComputeSpinWaveFFT_CUDA(spectrum, m *data.Slice,
	f_max float32, n_bins int, dt float32, Nx, Ny, Nz, N int) {

	C.k_computeSpinWaveFFT_async(
		(*C.float)(unsafe.Pointer(spectrum.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		C.float(f_max), C.int(n_bins), C.float(dt),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.15: SHNO
func SAFUpdateSHNO_CUDA(m *data.Slice, Ms, regions unsafe.Pointer,
	J_SHE, theta_SH, alpha_damping, dt float32, Nx, Ny, Nz, N int) {

	C.k_updateSHNO_async(
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.uchar)(regions),
		C.float(J_SHE), C.float(theta_SH), C.float(alpha_damping), C.float(dt),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.16: Exchange bias
func SAFAddExchangeBias_CUDA(dst, m *data.Slice, Ms, regions unsafe.Pointer,
	J_exbias, Hx_exbias, Hy_exbias, Hz_exbias float32,
	layer_FM, layer_AFM int, thickness float32, Nx, Ny, Nz, N int) {

	C.k_addExchangeBias_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.uchar)(regions),
		C.float(J_exbias), C.float(Hx_exbias), C.float(Hy_exbias), C.float(Hz_exbias),
		C.int(layer_FM), C.int(layer_AFM), C.float(thickness),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.17: Voltage-controlled RKKY
func SAFAddVoltageRKKY_CUDA(dst, m *data.Slice, Ms, regions, Ez unsafe.Pointer,
	J0, dJ_dV float32, layer1, layer2 int, thickness float32, Nx, Ny, Nz, N int) {

	C.k_addVoltageRKKY_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		(*C.uchar)(regions),
		(*C.float)(Ez),
		C.float(J0), C.float(dJ_dV),
		C.int(layer1), C.int(layer2), C.float(thickness),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.18: Atomistic-continuum coupling
func SAFCoupleAtomisticContinuum_CUDA(m_cont, S_atom *data.Slice, Ms unsafe.Pointer,
	a_lattice, J_exchange, Aex_continuum, dx_cont float32, Nx, Ny, Nz, N int) {

	C.k_coupleAtomisticContinuum_async(
		(*C.float)(unsafe.Pointer(m_cont.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m_cont.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m_cont.DevPtr(2))),
		(*C.float)(unsafe.Pointer(S_atom.DevPtr(0))),
		(*C.float)(unsafe.Pointer(S_atom.DevPtr(1))),
		(*C.float)(unsafe.Pointer(S_atom.DevPtr(2))),
		(*C.float)(Ms),
		C.float(a_lattice), C.float(J_exchange), C.float(Aex_continuum), C.float(dx_cont),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.19: Landau-Lifshitz-Bloch
func SAFUpdateLLBloch_CUDA(m, Ms_effective *data.Slice, Ms unsafe.Pointer, T_field *data.Slice,
	Ms0, T_curie, lambda_long, alpha, dt float32, Nx, Ny, Nz, N int) {

	C.k_updateLLBloch_async(
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(unsafe.Pointer(Ms_effective.DevPtr(0))),
		(*C.float)(Ms),
		(*C.float)(unsafe.Pointer(T_field.DevPtr(0))),
		C.float(Ms0), C.float(T_curie), C.float(lambda_long), C.float(alpha), C.float(dt),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.20: Dipolar skyrmion interactions
func SAFAddDipolarSkyrmion_CUDA(dst, m *data.Slice, Ms unsafe.Pointer,
	r_cutoff, dx, dy float32, Nx, Ny, Nz, N int) {

	C.k_addDipolarSkyrmion_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(1))),
		(*C.float)(unsafe.Pointer(dst.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(Ms),
		C.float(r_cutoff), C.float(dx), C.float(dy),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.21: Synaptic homeostasis
func SAFApplyHomeostasis_CUDA(weights, activity *data.Slice,
	target_rate, tau_homeo, dt float32, N int) {

	C.k_applyHomeostasis_async(
		(*C.float)(unsafe.Pointer(weights.DevPtr(0))),
		(*C.float)(unsafe.Pointer(activity.DevPtr(0))),
		C.float(target_rate), C.float(tau_homeo), C.float(dt), C.int(N), nil)
}

// V2.22: Dendritic computation
func SAFComputeDendritic_CUDA(output, inputs, weights *data.Slice,
	threshold, V_reset float32, N int) {

	C.k_computeDendritic_async(
		(*C.float)(unsafe.Pointer(output.DevPtr(0))),
		(*C.float)(unsafe.Pointer(inputs.DevPtr(0))),
		(*C.float)(unsafe.Pointer(weights.DevPtr(0))),
		C.float(threshold), C.float(V_reset), C.int(N), nil)
}

// V2.23: Winner-Take-All
func SAFUpdateWTA_CUDA(activity, input *data.Slice,
	inhibition_strength, tau_wta, dt float32, N int) {

	C.k_updateWTA_async(
		(*C.float)(unsafe.Pointer(activity.DevPtr(0))),
		(*C.float)(unsafe.Pointer(input.DevPtr(0))),
		C.float(inhibition_strength), C.float(tau_wta), C.float(dt), C.int(N), nil)
}

// V2.24: Topological Hall effect
// Corrected signature: passes J_current first as vector field, then rho_n as scalar
func SAFCalculateTopologicalHall_CUDA(V_Hall, m, J_current *data.Slice,
	rho_n, dx, dy, thickness float32, Nx, Ny, Nz, N int) {

	C.k_calculateTopologicalHall_async(
		(*C.float)(unsafe.Pointer(V_Hall.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(unsafe.Pointer(J_current.DevPtr(0))),
		(*C.float)(unsafe.Pointer(J_current.DevPtr(1))),
		(*C.float)(unsafe.Pointer(J_current.DevPtr(2))), // Pass Jz if available or just ensure J is 3D
		C.float(rho_n),
		C.float(dx), C.float(dy), C.float(thickness),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// V2.25: Spin-charge pumping
func SAFCalculateSpinPumping_CUDA(I_pump, m, m_old *data.Slice,
	g_eff, lambda_N, dt float32, Nx, Ny, Nz, N int) {

	C.k_calculateSpinPumping_async(
		(*C.float)(unsafe.Pointer(I_pump.DevPtr(0))),
		(*C.float)(unsafe.Pointer(I_pump.DevPtr(1))),
		(*C.float)(unsafe.Pointer(I_pump.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m.DevPtr(2))),
		(*C.float)(unsafe.Pointer(m_old.DevPtr(0))),
		(*C.float)(unsafe.Pointer(m_old.DevPtr(1))),
		(*C.float)(unsafe.Pointer(m_old.DevPtr(2))),
		C.float(g_eff), C.float(lambda_N), C.float(dt),
		C.int(Nx), C.int(Ny), C.int(Nz), nil)
}

// SAFRecycle returns a buffer to the MuMax3 pool
func SAFRecycle(s *data.Slice) {
	Recycle(s)
}
