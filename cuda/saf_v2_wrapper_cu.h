// saf_v2_wrapper_cu.h
// C function declarations for V2 CUDA kernels
// This file is included by saf_wrapper.go for CGo bindings
//
// Copyright © 2025-2026 Dr. Santhosh Sivasubramani
//
// Affiliation:
// INTRINSIC Lab, Centre for Sensors Instrumentation and
// Cyber Physical System Engineering (SeNSE)
// Indian Institute of Technology Delhi, New Delhi, India
//
// Contact: ssivasub@iitd.ac.in, ragansanthosh@ieee.org
// MuMax3-SAF-NeuroSpin V2.1 Extension

#ifndef SAF_V2_WRAPPER_CU_H
#define SAF_V2_WRAPPER_CU_H

#include <stdint.h>

// Forward declaration for curandState (defined in curand_kernel.h which is C++ only)
typedef struct curandStateXORWOW curandState;

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// V2.1: MULTI-NEIGHBOR RKKY
// ============================================================================
void k_addMultiNeighborRKKY_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const uint8_t* regions,
    float J1, float J2, float J3, float J_lateral,
    int layer1, int layer2, int n_neighbors,
    float thickness, int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.2: INTERLAYER DMI
// ============================================================================
void k_addInterlayerDMI_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const uint8_t* regions,
    float D_inter, int layer1, int layer2,
    float dx, float dy, float dz,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.3: NON-COLLINEAR RKKY
// ============================================================================
void k_addNonCollinearRKKY_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const uint8_t* regions,
    float J_ncol, float alpha_ncol,
    int layer1, int layer2, float thickness,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.4: SPIN DIFFUSION - VALET-FERT SOLVER
// ============================================================================
void k_solveSpinDiffusion_async(
    float* mu_s_x, float* mu_s_y, float* mu_s_z,
    const float* Jc, const float* P, const float* lambda_sf,
    float dz, int Nx, int Ny, int Nz, int max_iter,
    void* stream);

void k_addSpinDiffusionField_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* mu_s_x, const float* mu_s_y, const float* mu_s_z,
    const float* Ms, float theta_SH, float thickness,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.5: STOCHASTIC STDP
// ============================================================================
void k_applyStochasticSTDP_async(
    float* weights,
    const float* pre_spikes, const float* post_spikes,
    curandState* states,
    float A_plus, float A_minus, float tau_plus, float tau_minus,
    float noise_sigma, float retention_tau, float dt, int N,
    void* stream);

// ============================================================================
// V2.6: RESERVOIR COMPUTING
// ============================================================================
void k_updateReservoirState_async(
    float* state, const float* input, const float* W_res,
    float leak_rate, float spectral_radius, int N, int N_inputs,
    void* stream);

// ============================================================================
// V2.7: METAPLASTICITY
// ============================================================================
void k_applyMetaplasticity_async(
    float* weights, float* thresh, const float* activity,
    float theta_target, float tau_meta, float dt, int N,
    void* stream);

// ============================================================================
// V2.8: HEAT DIFFUSION
// ============================================================================
void k_solveHeatDiffusion_async(
    float* T_new, const float* T_old,
    const float* Jc, const float* sigma, const float* rho_cp,
    float kappa, float dt, float dx, float dy, float dz,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.9: NONLINEAR VCMA
// ============================================================================
void k_addNonlinearVCMA_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const float* Ez,
    float xi1, float xi2, float xi3, float t_int,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.10: MAGNON-PHONON COUPLING
// ============================================================================
void k_addMagnonPhononCoupling_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const float* strain_xx, const float* strain_yy,
    float B1, float B2,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.11: QUANTUM TUNNELING
// ============================================================================
void k_calculateQuantumTMR_async(
    float* TMR,
    const float* mx1, const float* my1, const float* mz1,
    const float* mx2, const float* my2, const float* mz2,
    float t_barrier, float phi_barrier, float P1, float P2,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.13: TEMPERATURE-DEPENDENT RKKY
// ============================================================================
void k_addTemperatureDependentRKKY_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const uint8_t* regions,
    const float* Temperature,
    float J0_base, float T_curie, float temp_exponent,
    int layer1, int layer2, float thickness,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.14: SPIN WAVE FFT ANALYSIS
// ============================================================================
void k_computeSpinWaveFFT_async(
    float* spectrum,
    const float* mx, const float* my, const float* mz,
    float f_max, int n_bins, float dt,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.15: SHNO (SPIN HALL NANO-OSCILLATORS)
// ============================================================================
void k_updateSHNO_async(
    float* mx, float* my, float* mz,
    const float* Ms, const uint8_t* regions,
    float J_SHE, float theta_SH, float alpha_damping, float dt,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.16: EXCHANGE BIAS
// ============================================================================
void k_addExchangeBias_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const uint8_t* regions,
    float J_exbias, float Hx_exbias, float Hy_exbias, float Hz_exbias,
    int layer_FM, int layer_AFM, float thickness,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.17: VOLTAGE-CONTROLLED RKKY
// ============================================================================
void k_addVoltageRKKY_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const uint8_t* regions,
    const float* Ez,
    float J0, float dJ_dV,
    int layer1, int layer2, float thickness,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.18: ATOMISTIC-CONTINUUM COUPLING
// ============================================================================
void k_coupleAtomisticContinuum_async(
    float* m_cont_x, float* m_cont_y, float* m_cont_z,
    const float* S_atom_x, const float* S_atom_y, const float* S_atom_z,
    const float* Ms,
    float a_lattice, float J_exchange, float Aex_continuum, float dx_cont,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.19: LANDAU-LIFSHITZ-BLOCH EQUATIONS
// ============================================================================
void k_updateLLBloch_async(
    float* mx, float* my, float* mz,
    float* Ms_effective,
    const float* Ms, const float* Temperature,
    float Ms0, float T_curie, float lambda_long, float alpha, float dt,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.20: DIPOLAR SKYRMION INTERACTIONS
// ============================================================================
void k_addDipolarSkyrmion_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms,
    float r_cutoff, float dx, float dy,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.21: SYNAPTIC HOMEOSTASIS
// ============================================================================
void k_applyHomeostasis_async(
    float* weights, const float* activity,
    float target_rate, float tau_homeo, float dt, int N,
    void* stream);

// ============================================================================
// V2.22: DENDRITIC COMPUTATION
// ============================================================================
void k_computeDendritic_async(
    float* output, const float* inputs, const float* weights,
    float threshold, float V_reset, int N,
    void* stream);

// ============================================================================
// V2.23: WINNER-TAKE-ALL
// ============================================================================
void k_updateWTA_async(
    float* activity, const float* input,
    float inhibition_strength, float tau_wta, float dt, int N,
    void* stream);

// ============================================================================
// V2.24: TOPOLOGICAL HALL EFFECT
// ============================================================================
void k_calculateTopologicalHall_async(
    float* V_Hall,
    const float* mx, const float* my, const float* mz,
    const float* Jx, const float* Jy, const float* Jz,
    float rho_n, float dx, float dy, float thickness,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// V2.25: SPIN-CHARGE PUMPING
// ============================================================================
void k_calculateSpinPumping_async(
    float* I_pump_x, float* I_pump_y, float* I_pump_z,
    const float* mx, const float* my, const float* mz,
    const float* mx_old, const float* my_old, const float* mz_old,
    float g_eff, float lambda_N, float dt,
    int Nx, int Ny, int Nz,
    void* stream);

// ============================================================================
// CURAND STATE MANAGEMENT (for stochastic kernels)
// ============================================================================
curandState* allocate_curand_states(int N);
void free_curand_states(curandState* states);
void init_curand_states_async(curandState* states, unsigned long long seed, int N, void* stream);

#ifdef __cplusplus
}
#endif

#endif // SAF_V2_WRAPPER_CU_H
