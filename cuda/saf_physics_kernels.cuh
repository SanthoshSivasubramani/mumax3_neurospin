/*
 * SAF Physics Kernels - Header
 * Micromagnetic simulation extensions for synthetic antiferromagnets (SAF)
 *
 * Modules:
 *  - RKKY interlayer coupling
 *  - Thermal stochastic fields
 *  - Spin-orbit torque
 *  - Spin-transfer torque
 *  - Voltage-controlled magnetic anisotropy
 *  - Oersted fields
 *  - Neuromorphic primitives (analog weight programming, STDP)
 *
 * Requirements: CUDA 11.8+, cuRAND library
 */

#ifndef SAF_PHYSICS_KERNELS_CUH
#define SAF_PHYSICS_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

#define MU_B        9.274009994e-24f
#define K_B         1.380649e-23f
#define HBAR        1.054571817e-34f
#define E_CHARGE    1.602176634e-19f
#define MU_0        (4.0e-7f * 3.14159265359f)
#define GAMMA_E     1.760859644e11f
#define PI          3.14159265359f

// -------------------------------
// C wrappers (extern "C")
// FIXED: Changed all cudaStream_t to void* for compatibility
// -------------------------------
extern "C" {

void k_addRKKYField_async(float* Bx, float* By, float* Bz,
                          const float* mx, const float* my, const float* mz,
                          const float* Ms, const uint8_t* regions,
                          float J_rkky, int layer1, int layer2, float thickness,
                          int Nx, int Ny, int Nz, int N, void* stream);

void k_getRKKYEnergy_async(float* E_out, const float* mx, const float* my, const float* mz,
                           const uint8_t* regions, float J_rkky, int layer1, int layer2,
                           int Nx, int Ny, int Nz, int N, float cellArea, void* stream);

void k_initThermalRandom_async(curandState* states, unsigned long seed, int N, void* stream);

void k_addThermalField_async(float* Bx, float* By, float* Bz,
                             const float* Ms, const float* alpha, const uint8_t* regions,
                             curandState* states, float temperature, float dt,
                             float dx, float dy, float dz,
                             int Nx, int Ny, int Nz, int N, void* stream);

void k_addSOTField_async(float* Bx, float* By, float* Bz,
                         const float* mx, const float* my, const float* mz,
                         const float* Ms, const float* Jc, const uint8_t* regions,
                         float theta_SH, float theta_FL, float thickness,
                         int Nx, int Ny, int Nz, int N, void* stream);

// STT - add dx parameter
void k_addSTTField_async(
    float* Bx, float* By, float* Bz,
    const float* mx, const float* my, const float* mz,
    const float* Ms, const float* Jc, const float* pol,
    const uint8_t* regions, float xi_ad, float xi_nad, float thickness, float dx,
    int Nx, int Ny, int Nz, int N, void* stream);

void k_addVCMAField_async(float* Bx, float* By, float* Bz,
                          const float* mx, const float* my, const float* mz,
                          const float* Ms, const float* Ez, const uint8_t* regions,
                          float xi_vcma, float t_interface,
                          int Nx, int Ny, int Nz, int N, void* stream);

// Oersted - add dx, dy, dz parameters
void k_addOerstedField_async(
    float* Bx, float* By, float* Bz,
    const float* Jc, const uint8_t* regions,
    float wire_width, float wire_thickness, float dx, float dy, float dz,
    int Nx, int Ny, int Nz, int N, void* stream);

void k_computeTopologicalCharge_async(float* Q_out,
                                      const float* mx, const float* my, const float* mz,
                                      const uint8_t* regions, float dx, float dy,
                                      int Nx, int Ny, int Nz, int N, void* stream);

void k_programAnalogWeight_async(float* target_param,
                                 const float* current_param,
                                 const float* target_weights,
                                 const uint8_t* regions,
                                 float param_min, float param_max, float programming_rate,
                                 int region_id, int N, void* stream);

void k_applySTDP_async(float* weight_updates,
                       const float* pre_spike_times,
                       const float* post_spike_times,
                       const float* current_weights,
                       float A_plus, float A_minus,
                       float tau_plus, float tau_minus, float dt,
                       int N, void* stream);

// Region-wise magnetization accumulation (for unbiased averaging)
void k_accumulateRegionSums_async(
    const float* mx, const float* my, const float* mz,
    const uint8_t* regions,
    int N, int target_region,
    float* mx_sum, float* my_sum, float* mz_sum, int* count,
    void* stream);

curandState* allocate_curand_states(int N);
void free_curand_states(curandState* states);

} // extern "C"

#endif // SAF_PHYSICS_KERNELS_CUH
