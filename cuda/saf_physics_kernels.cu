/*
 * MuMax3-SAF-NeuroSpin Extension
 * Synthetic Antiferromagnet (SAF) Physics & Neuromorphic Device Modeling
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
 *
 * This file extends the MuMax3 kernel set with:
 *  - RKKY interlayer coupling (oscillatory and decaying)
 *  - Thermal noise with cuRAND
 *  - Spin-orbit and spin-transfer torques
 *  - Voltage-controlled anisotropy (VCMA)
 *  - Neuromorphic primitives (STDP, analog weight programming)
 *  - Region-wise magnetization accumulation (SAF-accurate)
 *
 * Licensed under GPLv3
 */

#include <cstdint> // For uint8_t on Windows MSVC
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>

#define MU_B 9.274009994e-24f
#define K_B 1.380649e-23f
#define HBAR 1.054571817e-34f
#define E_CHARGE 1.602176634e-19f
#define MU_0 (4.0e-7f * 3.14159265359f)
#define GAMMA_E 1.760859644e11f
#define PI 3.14159265359f

__device__ inline int idx3D(int x, int y, int z, int Nx, int Ny) {
  return z * Nx * Ny + y * Nx + x;
}

__device__ inline bool validIdx(int x, int y, int z, int Nx, int Ny, int Nz) {
  return (x >= 0 && x < Nx && y >= 0 && y < Ny && z >= 0 && z < Nz);
}

__global__ void addRKKYField_kernel(
    float *__restrict__ Bx, float *__restrict__ By, float *__restrict__ Bz,
    const float *__restrict__ mx, const float *__restrict__ my,
    const float *__restrict__ mz, const float *__restrict__ Ms,
    const uint8_t *__restrict__ regions, float J_rkky, int layer1, int layer2,
    float thickness, int Nx, int Ny, int Nz, int oscillatory_enable,
    float wavelength, float phase, float decay_length) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = Nx * Ny * Nz;
  if (idx >= N)
    return;

  uint8_t region = regions[idx];
  float Ms_val = Ms[idx];
  if (Ms_val <= 0.0f)
    return;

  if (region != layer1 && region != layer2)
    return;

  int z = idx / (Nx * Ny);
  int xy = idx % (Nx * Ny);
  int y = xy / Nx;
  int x = xy % Nx;

  // Calculate local J value
  float J_local = J_rkky;

  if (oscillatory_enable) {
    float z_pos = (z + 0.5f) * thickness;
    float arg = 2.0f * PI * z_pos / wavelength + phase;
    J_local = J_rkky * cosf(arg);

    if (decay_length > 0.0f) {
      J_local *= expf(-z_pos / decay_length);
    }
  }

  float prefactor = J_local / (Ms_val * thickness);

  if (region == layer1) {
    if (z + 1 < Nz) {
      int idx_up = (z + 1) * Nx * Ny + y * Nx + x;
      if (regions[idx_up] == layer2) {
        atomicAdd(&Bx[idx], prefactor * mx[idx_up]);
        atomicAdd(&By[idx], prefactor * my[idx_up]);
        atomicAdd(&Bz[idx], prefactor * mz[idx_up]);
      }
    }
    if (z > 0) {
      int idx_down = (z - 1) * Nx * Ny + y * Nx + x;
      if (regions[idx_down] == layer2) {
        atomicAdd(&Bx[idx], prefactor * mx[idx_down]);
        atomicAdd(&By[idx], prefactor * my[idx_down]);
        atomicAdd(&Bz[idx], prefactor * mz[idx_down]);
      }
    }
  }

  if (region == layer2) {
    if (z + 1 < Nz) {
      int idx_up = (z + 1) * Nx * Ny + y * Nx + x;
      if (regions[idx_up] == layer1) {
        atomicAdd(&Bx[idx], prefactor * mx[idx_up]);
        atomicAdd(&By[idx], prefactor * my[idx_up]);
        atomicAdd(&Bz[idx], prefactor * mz[idx_up]);
      }
    }
    if (z > 0) {
      int idx_down = (z - 1) * Nx * Ny + y * Nx + x;
      if (regions[idx_down] == layer1) {
        atomicAdd(&Bx[idx], prefactor * mx[idx_down]);
        atomicAdd(&By[idx], prefactor * my[idx_down]);
        atomicAdd(&Bz[idx], prefactor * mz[idx_down]);
      }
    }
  }
}

__global__ void
getRKKYEnergy_kernel(float *__restrict__ E_out, const float *__restrict__ mx,
                     const float *__restrict__ my, const float *__restrict__ mz,
                     const uint8_t *__restrict__ regions, float J_rkky,
                     int layer1, int layer2, float cellArea, float thickness,
                     int Nx, int Ny, int Nz, int oscillatory_enable,
                     float wavelength, float phase, float decay_length) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = Nx * Ny * Nz;
  if (idx >= N)
    return;

  E_out[idx] = 0.0f;

  uint8_t region = regions[idx];
  if (region != layer1)
    return;

  int z = idx / (Nx * Ny);
  int xy = idx % (Nx * Ny);
  int y = xy / Nx;
  int x = xy % Nx;

  // Calculate local J value
  float J_local = J_rkky;

  if (oscillatory_enable) {
    float z_pos = (z + 0.5f) * thickness;
    float arg = 2.0f * PI * z_pos / wavelength + phase;
    J_local = J_rkky * cosf(arg);

    if (decay_length > 0.0f) {
      J_local *= expf(-z_pos / decay_length);
    }
  }

  // Check z+1 neighbor
  if (z + 1 < Nz) {
    int idx_up = (z + 1) * Nx * Ny + y * Nx + x;
    if (regions[idx_up] == layer2) {
      float dot =
          mx[idx] * mx[idx_up] + my[idx] * my[idx_up] + mz[idx] * mz[idx_up];
      E_out[idx] += -J_local * dot * cellArea;
    }
  }

  // Check z-1 neighbor
  if (z > 0) {
    int idx_down = (z - 1) * Nx * Ny + y * Nx + x;
    if (regions[idx_down] == layer2) {
      float dot = mx[idx] * mx[idx_down] + my[idx] * my[idx_down] +
                  mz[idx] * mz[idx_down];
      E_out[idx] += -J_local * dot * cellArea;
    }
  }
}

__global__ void addThermalField_kernel(
    float *__restrict__ Bx, float *__restrict__ By, float *__restrict__ Bz,
    const float *__restrict__ Ms, const float *__restrict__ alpha,
    const uint8_t *__restrict__ regions, curandState *__restrict__ states,
    float temperature, float dt, float cellVolume, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  // uint8_t region = regions[idx]; // Unused
  float Ms_val = Ms[idx];
  float alpha_val = alpha[idx];

  if (Ms_val <= 0.0f || temperature <= 0.0f)
    return;

  float prefactor = sqrtf(2.0f * alpha_val * K_B * temperature /
                          (GAMMA_E * Ms_val * cellVolume * dt));

  curandState localState = states[idx];
  float nx = curand_normal(&localState);
  float ny = curand_normal(&localState);
  float nz = curand_normal(&localState);
  states[idx] = localState;

  atomicAdd(&Bx[idx], prefactor * nx);
  atomicAdd(&By[idx], prefactor * ny);
  atomicAdd(&Bz[idx], prefactor * nz);
}

__global__ void initThermalRNG_kernel(curandState *__restrict__ states,
                                      unsigned long seed, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  curand_init(seed, idx, 0, &states[idx]);
}

// ============================================================================
// SPIN-ORBIT TORQUE (SOT) - Torque Convention
// ============================================================================
// Damping-like torque: τ_DL = -θ_SH (ℏJ)/(2eMs·t) [m × (m × ŷ)]
// Field-like torque:   τ_FL =  θ_FL (ℏJ)/(2eMs·t) [m × ŷ]
//
// Convention: Current flows in +x direction, spin accumulation in +y
// Units: J in A/m², t in m, result in equivalent field (T)
// ============================================================================

__global__ void
addSOTField_kernel(float *__restrict__ Bx, float *__restrict__ By,
                   float *__restrict__ Bz, const float *__restrict__ mx,
                   const float *__restrict__ my, const float *__restrict__ mz,
                   const float *__restrict__ Ms, const float *__restrict__ Jc,
                   const uint8_t *__restrict__ regions, float theta_SH,
                   float theta_FL, float thickness, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  // uint8_t region = regions[idx]; // Unused
  float Ms_val = Ms[idx];
  float Jc_val = Jc[idx];

  if (Ms_val <= 0.0f || Jc_val == 0.0f)
    return;

  float m_x = mx[idx];
  // float m_y = my[idx]; // Unused in current implementation
  float m_z = mz[idx];

  float prefactor = (HBAR * Jc_val) / (2.0f * E_CHARGE * Ms_val * thickness);

  float B_DL_x = -theta_SH * prefactor * m_z;
  float B_DL_y = 0.0f;
  float B_DL_z = theta_SH * prefactor * m_x;

  float B_FL_x = 0.0f;
  float B_FL_y = theta_FL * prefactor;
  float B_FL_z = 0.0f;

  atomicAdd(&Bx[idx], B_DL_x + B_FL_x);
  atomicAdd(&By[idx], B_DL_y + B_FL_y);
  atomicAdd(&Bz[idx], B_DL_z + B_FL_z);
}
// ============================================================================
// SPIN-TRANSFER TORQUE (STT) - Zhang-Li Convention
// ============================================================================
// Adiabatic torque:     τ_ad  = -ξ_ad β ∇_x·m
// Non-adiabatic torque: τ_nad =  ξ_nad β [m × (∇_x·m)]
//
// β = (ℏP)/(2e) · (J/Ms·t) where P is spin polarization
// Convention: Current in +x, spatial derivatives ∂/∂x
// Units: J in A/m², t in m, result in equivalent field (T)
// ============================================================================

__global__ void
addSTTField_kernel(float *__restrict__ Bx, float *__restrict__ By,
                   float *__restrict__ Bz, const float *__restrict__ mx,
                   const float *__restrict__ my, const float *__restrict__ mz,
                   const float *__restrict__ Ms, const float *__restrict__ Jc,
                   const float *__restrict__ pol,
                   const uint8_t *__restrict__ regions, float xi_ad,
                   float xi_nad, float thickness, float dx, // ADD dx parameter
                   int Nx, int Ny, int Nz) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = Nx * Ny * Nz;
  if (idx >= N)
    return;

  // uint8_t region = regions[idx]; // Unused
  int z = idx / (Nx * Ny);
  int xy = idx % (Nx * Ny);
  int y = xy / Nx;
  int x = xy % Nx;
  float Ms_val = Ms[idx];
  float Jc_val = Jc[idx];
  float P = pol[idx];

  if (Ms_val <= 0.0f || Jc_val == 0.0f)
    return;

  // Central difference in x-direction using correct spacing
  float dmx_dx = 0.0f, dmy_dx = 0.0f, dmz_dx = 0.0f;
  if (x > 0 && x < Nx - 1) {
    int idx_l = z * Nx * Ny + y * Nx + (x - 1);
    int idx_r = z * Nx * Ny + y * Nx + (x + 1);
    float delta_x = 2.0f * dx; // Use actual cell spacing, not thickness
    dmx_dx = (mx[idx_r] - mx[idx_l]) / delta_x;
    dmy_dx = (my[idx_r] - my[idx_l]) / delta_x;
    dmz_dx = (mz[idx_r] - mz[idx_l]) / delta_x;
  }

  float beta = (HBAR * P * Jc_val) / (2.0f * E_CHARGE * Ms_val * thickness);

  // Adiabatic STT
  float B_ad_x = -xi_ad * beta * dmx_dx;
  float B_ad_y = -xi_ad * beta * dmy_dx;
  float B_ad_z = -xi_ad * beta * dmz_dx;

  // Non-adiabatic STT
  float m_x = mx[idx];
  float m_y = my[idx];
  float m_z = mz[idx];

  float B_nad_x = xi_nad * beta * (m_y * dmz_dx - m_z * dmy_dx);
  float B_nad_y = xi_nad * beta * (m_z * dmx_dx - m_x * dmz_dx);
  float B_nad_z = xi_nad * beta * (m_x * dmy_dx - m_y * dmx_dx);

  atomicAdd(&Bx[idx], B_ad_x + B_nad_x);
  atomicAdd(&By[idx], B_ad_y + B_nad_y);
  atomicAdd(&Bz[idx], B_ad_z + B_nad_z);
}

__global__ void
addVCMAField_kernel(float *__restrict__ Bx, float *__restrict__ By,
                    float *__restrict__ Bz, const float *__restrict__ mx,
                    const float *__restrict__ my, const float *__restrict__ mz,
                    const float *__restrict__ Ms, const float *__restrict__ Ez,
                    const uint8_t *__restrict__ regions, float xi_vcma,
                    float t_interface, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  // uint8_t region = regions[idx]; // Unused
  float Ms_val = Ms[idx];
  float Ez_val = Ez[idx];

  if (Ms_val <= 0.0f || Ez_val == 0.0f)
    return;

  float m_z = mz[idx];
  float DeltaK = xi_vcma * Ez_val;
  float B_vcma_z = (2.0f * DeltaK * m_z) / (Ms_val * t_interface);

  atomicAdd(&Bz[idx], B_vcma_z);
}

__global__ void
addOerstedField_kernel(float *__restrict__ Bx, float *__restrict__ By,
                       float *__restrict__ Bz, const float *__restrict__ Jc,
                       const uint8_t *__restrict__ regions, float wire_width,
                       float wire_thickness, float dx, float dy, float dz,
                       int Nx, int Ny, int Nz) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = Nx * Ny * Nz;
  if (idx >= N)
    return;

  float Jc_val = Jc[idx];
  if (Jc_val == 0.0f)
    return; // Keep this

  // Extract 3D position in meters
  int z = idx / (Nx * Ny);
  int xy = idx % (Nx * Ny);
  int y = xy / Nx;

  float pos_y = (y + 0.5f) * dy;
  float pos_z = (z + 0.5f) * dz;

  // Assume wire centered at domain origin in x-y, extending in x-direction
  float wire_center_y = (Ny * dy) / 2.0f;
  float wire_center_z = (Nz * dz) / 2.0f;

  // Distance from wire center (perpendicular to current flow in x)
  float dy_wire = pos_y - wire_center_y;
  float dz_wire = pos_z - wire_center_z;
  float r = sqrtf(dy_wire * dy_wire + dz_wire * dz_wire);

  if (r < 1e-12f)
    return; // Avoid division by zero at wire center

  // Biot-Savart for infinite wire: B = (μ₀ I)/(2π r) in azimuthal direction
  float current = Jc_val * wire_width * wire_thickness;
  // Use explicit literals to avoid macro issues
  float mu0 = 1.25663706e-6f;
  float pi = 3.14159265f;
  float B_magnitude = (mu0 * current) / (2.0f * pi * r);

  // Azimuthal field (perpendicular to r and current direction)
  // Current flows in +x, so B circles around x-axis
  float B_y = -B_magnitude * dz_wire / r; // -∂z/r component
  float B_z = B_magnitude * dy_wire / r;  //  ∂y/r component

  // Ensure values are numbers
  bool failure = false;
  if (isnan(B_y) || isinf(B_y)) {
    B_y = 0.0f;
    failure = true;
  }
  if (isnan(B_z) || isinf(B_z)) {
    B_z = 0.0f;
    failure = true;
  }

  // DEBUG PRINT for Thread 0
  if (idx == 0) {
    printf("DEBUG KERNEL: Jc=%.2e, w=%.2e, t=%.2e, r=%.2e, mag=%.2e, By=%.2e, "
           "Bz=%.2e, Fail=%d\n",
           Jc_val, wire_width, wire_thickness, r, B_magnitude, B_y, B_z,
           failure);
  }

  atomicAdd(&Bx[idx], 0.0f); // Touch Bx to ensure consistency
  atomicAdd(&By[idx], B_y);
  atomicAdd(&Bz[idx], B_z);
}

__global__ void computeTopologicalCharge_kernel(
    float *__restrict__ Q_out, const float *__restrict__ mx,
    const float *__restrict__ my, const float *__restrict__ mz,
    const uint8_t *__restrict__ regions, float dx, float dy, int Nx, int Ny,
    int Nz) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = Nx * Ny * Nz;
  if (idx >= N)
    return;

  int z = idx / (Nx * Ny);
  int xy = idx % (Nx * Ny);
  int y = xy / Nx;
  int x = xy % Nx;
  Q_out[idx] = 0.0f;

  if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1)
    return;

  float m_x = mx[idx];
  float m_y = my[idx];
  float m_z = mz[idx];

  int idx_xp = idx3D(x + 1, y, z, Nx, Ny);
  int idx_xm = idx3D(x - 1, y, z, Nx, Ny);
  int idx_yp = idx3D(x, y + 1, z, Nx, Ny);
  int idx_ym = idx3D(x, y - 1, z, Nx, Ny);

  float dmx_dx = (mx[idx_xp] - mx[idx_xm]) / (2.0f * dx);
  float dmy_dx = (my[idx_xp] - my[idx_xm]) / (2.0f * dx);
  float dmz_dx = (mz[idx_xp] - mz[idx_xm]) / (2.0f * dx);

  float dmx_dy = (mx[idx_yp] - mx[idx_ym]) / (2.0f * dy);
  float dmy_dy = (my[idx_yp] - my[idx_ym]) / (2.0f * dy);
  float dmz_dy = (mz[idx_yp] - mz[idx_ym]) / (2.0f * dy);

  float cross_x = dmy_dx * dmz_dy - dmz_dx * dmy_dy;
  float cross_y = dmz_dx * dmx_dy - dmx_dx * dmz_dy;
  float cross_z = dmx_dx * dmy_dy - dmy_dx * dmx_dy;

  float dot = m_x * cross_x + m_y * cross_y + m_z * cross_z;

  Q_out[idx] = dot / (4.0f * PI);
}

__global__ void programAnalogWeight_kernel(
    float *__restrict__ target_param, const float *__restrict__ current_param,
    const float *__restrict__ target_weights,
    const uint8_t *__restrict__ regions, float param_min, float param_max,
    float programming_rate, int region_id, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  if (regions[idx] != region_id)
    return; // Proper region check

  float current_val = current_param[idx];
  float target_val = target_weights[idx];

  float desired_param = param_min + target_val * (param_max - param_min);
  float error = desired_param - current_val;
  float update = programming_rate * error;

  float new_val = current_val + update;
  new_val = fmaxf(param_min, fminf(param_max, new_val));

  target_param[idx] = new_val;
}

__global__ void applySTDP_kernel(float *__restrict__ weight_updates,
                                 const float *__restrict__ pre_spike_times,
                                 const float *__restrict__ post_spike_times,
                                 const float *__restrict__ current_weights,
                                 float A_plus, float A_minus, float tau_plus,
                                 float tau_minus, float dt, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  float t_pre = pre_spike_times[idx];
  float t_post = post_spike_times[idx];

  float delta_t = t_post - t_pre;
  float dw = 0.0f;

  if (delta_t > 0.0f) {
    dw = A_plus * expf(-delta_t / tau_plus);
  } else if (delta_t < 0.0f) {
    dw = -A_minus * expf(delta_t / tau_minus);
  }

  weight_updates[idx] = dw / dt;
}
// ============================================================================
// REGION-WISE MAGNETIZATION SUM KERNEL (for correct averaging)
// ============================================================================
// This kernel replaces the biased per-block average approach. It sums all
// magnetization components per region, then the host divides once by total
// count.
// ============================================================================

__global__ void accumulateRegionSums_kernel(
    const float *__restrict__ mx, const float *__restrict__ my,
    const float *__restrict__ mz, const uint8_t *__restrict__ regions, int N,
    int target_region, float *__restrict__ mx_sum, float *__restrict__ my_sum,
    float *__restrict__ mz_sum, int *__restrict__ count) {
  extern __shared__ unsigned char s[];
  float *sdata_x = (float *)s;
  float *sdata_y = sdata_x + blockDim.x;
  float *sdata_z = sdata_y + blockDim.x;
  int *scount = (int *)(sdata_z + blockDim.x);

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float lx = 0.f, ly = 0.f, lz = 0.f;
  int lc = 0;

  if (i < N && regions[i] == target_region) {
    lx = mx[i];
    ly = my[i];
    lz = mz[i];
    lc = 1;
  }

  sdata_x[tid] = lx;
  sdata_y[tid] = ly;
  sdata_z[tid] = lz;
  scount[tid] = lc;
  __syncthreads();

  // Parallel reduction inside block
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sdata_x[tid] += sdata_x[tid + offset];
      sdata_y[tid] += sdata_y[tid + offset];
      sdata_z[tid] += sdata_z[tid + offset];
      scount[tid] += scount[tid + offset];
    }
    __syncthreads();
  }

  // Correct patch: atomicAdd total sums, not averages
  if (tid == 0) {
    atomicAdd(&mx_sum[target_region], sdata_x[0]);
    atomicAdd(&my_sum[target_region], sdata_y[0]);
    atomicAdd(&mz_sum[target_region], sdata_z[0]);
    atomicAdd(&count[target_region], scount[0]);
  }
}

extern "C" {

void k_addRKKYField_async(float *Bx, float *By, float *Bz, const float *mx,
                          const float *my, const float *mz, const float *Ms,
                          const uint8_t *regions, float J_rkky, int layer1,
                          int layer2, float thickness, int Nx, int Ny, int Nz,
                          int N, int oscillatory_enable, float wavelength,
                          float phase, float decay_length, void *stream) {

  dim3 threads(256);
  dim3 blocks((N + threads.x - 1) / threads.x);
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  addRKKYField_kernel<<<blocks, threads, 0, cuda_stream>>>(
      Bx, By, Bz, mx, my, mz, Ms, regions, J_rkky, layer1, layer2, thickness,
      Nx, Ny, Nz, oscillatory_enable, wavelength, phase, decay_length);
}

void k_getRKKYEnergy_async(float *E_out, const float *mx, const float *my,
                           const float *mz, const uint8_t *regions,
                           float J_rkky, int layer1, int layer2, int Nx, int Ny,
                           int Nz, int N, float cellArea, float thickness,
                           int oscillatory_enable, float wavelength,
                           float phase, float decay_length, void *stream) {

  dim3 threads(256);
  dim3 blocks((N + threads.x - 1) / threads.x);
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  getRKKYEnergy_kernel<<<blocks, threads, 0, cuda_stream>>>(
      E_out, mx, my, mz, regions, J_rkky, layer1, layer2, cellArea, thickness,
      Nx, Ny, Nz, oscillatory_enable, wavelength, phase, decay_length);
}

void k_initThermalRandom_async(curandState *states, unsigned long seed, int N,
                               void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  initThermalRNG_kernel<<<blocks, threads, 0, cuda_stream>>>(states, seed, N);
}

void k_addThermalField_async(float *Bx, float *By, float *Bz, const float *Ms,
                             const float *alpha, const uint8_t *regions,
                             curandState *states, float temperature, float dt,
                             float dx, float dy, float dz, int Nx, int Ny,
                             int Nz, int N, void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  float cellVolume = dx * dy * dz;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  addThermalField_kernel<<<blocks, threads, 0, cuda_stream>>>(
      Bx, By, Bz, Ms, alpha, regions, states, temperature, dt, cellVolume, N);
}

void k_addSOTField_async(float *Bx, float *By, float *Bz, const float *mx,
                         const float *my, const float *mz, const float *Ms,
                         const float *Jc, const uint8_t *regions,
                         float theta_SH, float theta_FL, float thickness,
                         int Nx, int Ny, int Nz, int N, void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  addSOTField_kernel<<<blocks, threads, 0, cuda_stream>>>(
      Bx, By, Bz, mx, my, mz, Ms, Jc, regions, theta_SH, theta_FL, thickness,
      N);
}

void k_addSTTField_async(float *Bx, float *By, float *Bz, const float *mx,
                         const float *my, const float *mz, const float *Ms,
                         const float *Jc, const float *pol,
                         const uint8_t *regions, float xi_ad, float xi_nad,
                         float thickness, float dx, int Nx, int Ny, int Nz,
                         int N, void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  addSTTField_kernel<<<blocks, threads, 0, cuda_stream>>>(
      Bx, By, Bz, mx, my, mz, Ms, Jc, pol, regions, xi_ad, xi_nad, thickness,
      dx, Nx, Ny, Nz);
}

void k_addVCMAField_async(float *Bx, float *By, float *Bz, const float *mx,
                          const float *my, const float *mz, const float *Ms,
                          const float *Ez, const uint8_t *regions,
                          float xi_vcma, float t_interface, int Nx, int Ny,
                          int Nz, int N, void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  addVCMAField_kernel<<<blocks, threads, 0, cuda_stream>>>(
      Bx, By, Bz, mx, my, mz, Ms, Ez, regions, xi_vcma, t_interface, N);
}

void k_addOerstedField_async(float *Bx, float *By, float *Bz, const float *Jc,
                             const uint8_t *regions, float wire_width,
                             float wire_thickness, float dx, float dy, float dz,
                             int Nx, int Ny, int Nz, int N, void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  addOerstedField_kernel<<<blocks, threads, 0, cuda_stream>>>(
      Bx, By, Bz, Jc, regions, wire_width, wire_thickness, dx, dy, dz, Nx, Ny,
      Nz);
}

void k_computeTopologicalCharge_async(float *Q_out, const float *mx,
                                      const float *my, const float *mz,
                                      const uint8_t *regions, float dx,
                                      float dy, int Nx, int Ny, int Nz, int N,
                                      void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  computeTopologicalCharge_kernel<<<blocks, threads, 0, cuda_stream>>>(
      Q_out, mx, my, mz, regions, dx, dy, Nx, Ny, Nz);
}

void k_programAnalogWeight_async(float *target_param,
                                 const float *current_param,
                                 const float *target_weights,
                                 const uint8_t *regions, float param_min,
                                 float param_max, float programming_rate,
                                 int region_id, int N, void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  programAnalogWeight_kernel<<<blocks, threads, 0, cuda_stream>>>(
      target_param, current_param, target_weights, regions, param_min,
      param_max, programming_rate, region_id, N);
}

void k_applySTDP_async(float *weight_updates, const float *pre_spike_times,
                       const float *post_spike_times,
                       const float *current_weights, float A_plus,
                       float A_minus, float tau_plus, float tau_minus, float dt,
                       int N, void *stream) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  applySTDP_kernel<<<blocks, threads, 0, cuda_stream>>>(
      weight_updates, pre_spike_times, post_spike_times, current_weights,
      A_plus, A_minus, tau_plus, tau_minus, dt, N);
}
// ============================================================================
// Launcher for accumulateRegionSums_kernel
// ============================================================================

void k_accumulateRegionSums_async(const float *mx, const float *my,
                                  const float *mz, const uint8_t *regions,
                                  int N, int target_region, float *mx_sum,
                                  float *my_sum, float *mz_sum, int *count,
                                  void *stream) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  size_t shmem = threads * (3 * sizeof(float) + sizeof(int));
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  accumulateRegionSums_kernel<<<blocks, threads, shmem, cuda_stream>>>(
      mx, my, mz, regions, N, target_region, mx_sum, my_sum, mz_sum, count);
}

curandState *allocate_curand_states(int N) {
  curandState *states;
  cudaMalloc(&states, N * sizeof(curandState));
  return states;
}

void free_curand_states(curandState *states) { cudaFree(states); }

void init_curand_states_async(curandState *states, unsigned long long seed,
                              int N, void *stream) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  initThermalRNG_kernel<<<blocks, threads, 0, cuda_stream>>>(states, seed, N);
}

// Helper to set a float array to a value (since cudaMemset is byte-wise)
__global__ void setValue_kernel(float *dst, float val, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    dst[idx] = val;
  }
}

void k_setValue_async(float *dst, float val, int N, void *stream) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  setValue_kernel<<<blocks, threads, 0, cuda_stream>>>(dst, val, N);
}
}
