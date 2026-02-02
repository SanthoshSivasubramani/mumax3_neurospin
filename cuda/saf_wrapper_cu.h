/*
 * MuMax3-SAF-NeuroSpin CUDA Wrapper
 * Provides C/CUDA interface bindings for SAF and neuromorphic kernels.
 *
 * Copyright © 2025-2026 Dr. Santhosh Sivasubramani
 *
 * Affiliations:
 * 1. INTRINSIC Lab, Centre for Sensors Instrumentation and
 *    Cyber Physical System Engineering (SeNSE)
 *    Indian Institute of Technology Delhi, New Delhi, India
 * 2. April AI Hub, Centre for Electronic Frontiers
 *    The University of Edinburgh, Edinburgh, United Kingdom
 *
 * Contact: ssivasub@iitd.ac.in, ssivasub@ed.ac.uk, ragansanthosh@ieee.org
 * Repository: https://github.com/SanthoshSivasubramani/mumax3-neurospin
 * License: GPLv3
 */

#ifndef SAF_WRAPPER_CU_H
#define SAF_WRAPPER_CU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer type - hides curandState from C compiler
// Forward declaration of CUDA RNG state (opaque to C)
struct curandStateXORWOW;
typedef struct curandStateXORWOW curandState;

// RKKY coupling
void k_addRKKYField_async(float *Bx, float *By, float *Bz, const float *mx,
                          const float *my, const float *mz, const float *Ms,
                          const uint8_t *regions, float J_rkky, int layer1,
                          int layer2, float thickness, int Nx, int Ny, int Nz,
                          int N, int oscillatory_enable, float wavelength,
                          float phase, float decay_length, void *stream);

// RKKY energy
void k_getRKKYEnergy_async(float *E_out, const float *mx, const float *my,
                           const float *mz, const uint8_t *regions,
                           float J_rkky, int layer1, int layer2, int Nx, int Ny,
                           int Nz, int N, float cellArea, float thickness,
                           int oscillatory_enable, float wavelength,
                           float phase, float decay_length, void *stream);

// Thermal random initialization
void k_initThermalRandom_async(curandState *states, unsigned long seed, int N,
                               void *stream);

// Thermal field
void k_addThermalField_async(float *Bx, float *By, float *Bz, const float *Ms,
                             const float *alpha, const uint8_t *regions,
                             curandState *states, float temperature, float dt,
                             float dx, float dy, float dz, int Nx, int Ny,
                             int Nz, int N, void *stream);

// Spin-orbit torque
void k_addSOTField_async(float *Bx, float *By, float *Bz, const float *mx,
                         const float *my, const float *mz, const float *Ms,
                         const float *Jc, const uint8_t *regions,
                         float theta_SH, float theta_FL, float thickness,
                         int Nx, int Ny, int Nz, int N, void *stream);

// Spin-transfer torque
void k_addSTTField_async(float *Bx, float *By, float *Bz, const float *mx,
                         const float *my, const float *mz, const float *Ms,
                         const float *Jc, const float *pol,
                         const uint8_t *regions, float xi_ad, float xi_nad,
                         float thickness, float dx, int Nx, int Ny, int Nz,
                         int N, void *stream);

// VCMA field
void k_addVCMAField_async(float *Bx, float *By, float *Bz, const float *mx,
                          const float *my, const float *mz, const float *Ms,
                          const float *Ez, const uint8_t *regions,
                          float xi_vcma, float t_interface, int Nx, int Ny,
                          int Nz, int N, void *stream);

// Oersted field
void k_addOerstedField_async(float *Bx, float *By, float *Bz, const float *Jc,
                             const uint8_t *regions, float wire_width,
                             float wire_thickness, float dx, float dy, float dz,
                             int Nx, int Ny, int Nz, int N, void *stream);

// Topological charge
void k_computeTopologicalCharge_async(float *Q_out, const float *mx,
                                      const float *my, const float *mz,
                                      const uint8_t *regions, float dx,
                                      float dy, int Nx, int Ny, int Nz, int N,
                                      void *stream);

// Analog weight programming
void k_programAnalogWeight_async(float *target_param,
                                 const float *current_param,
                                 const float *target_weights,
                                 const uint8_t *regions, float param_min,
                                 float param_max, float programming_rate,
                                 int region_id, int N, void *stream);

// STDP learning
void k_applySTDP_async(float *weight_updates, const float *pre_spike_times,
                       const float *post_spike_times,
                       const float *current_weights, float A_plus,
                       float A_minus, float tau_plus, float tau_minus, float dt,
                       int N, void *stream);

// Region-wise magnetization accumulation
void k_accumulateRegionSums_async(const float *mx, const float *my,
                                  const float *mz, const uint8_t *regions,
                                  int N, int target_region, float *mx_sum,
                                  float *my_sum, float *mz_sum, int *count,
                                  void *stream);

curandState *allocate_curand_states(int N);
void free_curand_states(curandState *states);

// Helper for float memset
void k_setValue_async(float *dst, float val, int N, void *stream);

#ifdef __cplusplus
}
#endif

#endif // SAF_WRAPPER_CU_H
