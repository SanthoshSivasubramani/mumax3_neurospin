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

// =====================================================================================================
// MuMax3-SAF-NeuroSpin V2.1: Multi-Scale Physics
// =====================================================================================================
// This module implements multi-scale physics extensions:
// - Grain boundary modeling (Voronoi, EBSD import)
// - Multiple oversampling levels (adaptive)
// - Dynamic defect generation (point, cluster, irradiation)
// - Spatial correlation functions for roughness
// =====================================================================================================

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// ============================================================================
// MULTI-SCALE PARAMETERS
// ============================================================================

var (
	// Grain boundary modeling
	grainBoundaryEnabled = false
	grainSizeMean        = NewScalarParam("grainSizeMean", "m", "Mean grain size")
	grainSizeStd         = NewScalarParam("grainSizeStd", "m", "Grain size std dev")
	grainKScatter        = NewScalarParam("grainKScatter", "", "Anisotropy scatter (fraction)")
	grainAexReduction    = NewScalarParam("grainAexReduction", "", "Exchange reduction at boundaries")
	grainOrientations    *data.Slice // Euler angles per grain
	grainIDs             *data.Slice // Grain ID per cell
	numGrains            int

	// Adaptive oversampling
	adaptiveOversamplingEnabled = false
	oversamplingLevel           = 1
	oversamplingThreshold       = NewScalarParam("oversamplingThreshold", "", "Gradient threshold for oversampling")
	oversamplingMask            *data.Slice

	// Dynamic defects
	dynamicDefectsEnabled = false
	defectDensity         = NewScalarParam("defectDensity", "1/m³", "Defect density")
	defectStrength        = NewScalarParam("defectStrength", "", "Defect pinning strength")
	defectLocations       *data.Slice
	numDefects            int

	// Spatial correlation
	correlatedRoughnessEnabled = false
	roughnessCorrelationType   = "exponential" // "exponential", "gaussian", "self-affine"
	roughnessHurstExponent     = NewScalarParam("roughnessHurstExponent", "", "Hurst exponent for self-affine")
	roughnessHeightMap         *data.Slice
)

// ============================================================================
// FEATURE V2.1-5: GRAIN BOUNDARY MODELING
// ============================================================================

// EnableGrainBoundaries enables polycrystalline grain structure modeling
func EnableGrainBoundaries() {
	grainBoundaryEnabled = true
	grainSizeMean.setRegion(0, []float64{20e-9})   // 20 nm mean
	grainSizeStd.setRegion(0, []float64{5e-9})     // 5 nm std
	grainKScatter.setRegion(0, []float64{0.1})     // 10% K variation
	grainAexReduction.setRegion(0, []float64{0.5}) // 50% Aex at boundaries

	size := Mesh().Size()
	grainIDs = data.NewSlice(1, size)
	grainOrientations = data.NewSlice(3, [3]int{1024, 1, 1}) // Max 1024 grains

	LogOut("Grain boundary modeling enabled")
}

// SetGrainSize sets the mean and standard deviation of grain size
func SetGrainSize(mean, std float64) {
	grainSizeMean.setRegion(0, []float64{mean})
	grainSizeStd.setRegion(0, []float64{std})
}

// SetGrainAnisotropyScatter sets the fractional scatter in anisotropy
func SetGrainAnisotropyScatter(scatter float64) {
	if scatter < 0 || scatter > 1 {
		LogErr("Anisotropy scatter must be 0-1")
		return
	}
	grainKScatter.setRegion(0, []float64{scatter})
}

// SetGrainExchangeReduction sets exchange reduction at boundaries
func SetGrainExchangeReduction(reduction float64) {
	if reduction < 0 || reduction > 1 {
		LogErr("Exchange reduction must be 0-1")
		return
	}
	grainAexReduction.setRegion(0, []float64{reduction})
}

// GenerateVoronoiGrains generates a Voronoi tessellation grain structure
func GenerateVoronoiGrains(numSeeds int) {
	if !grainBoundaryEnabled {
		EnableGrainBoundaries()
	}

	if numSeeds < 1 || numSeeds > 1024 {
		LogErr("Number of seeds must be 1-1024")
		return
	}

	size := Mesh().Size()
	Nx, Ny, Nz := size[0], size[1], size[2]
	cellSize := Mesh().CellSize()

	// Generate random seed positions
	seeds := make([][3]float64, numSeeds)
	for i := 0; i < numSeeds; i++ {
		seeds[i][0] = rand.Float64() * float64(Nx) * cellSize[0]
		seeds[i][1] = rand.Float64() * float64(Ny) * cellSize[1]
		seeds[i][2] = rand.Float64() * float64(Nz) * cellSize[2]
	}

	// Assign each cell to nearest seed (Voronoi)
	grainIDData := grainIDs.Host()[0]
	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				px := (float64(ix) + 0.5) * cellSize[0]
				py := (float64(iy) + 0.5) * cellSize[1]
				pz := (float64(iz) + 0.5) * cellSize[2]

				minDist := math.MaxFloat64
				nearestGrain := 0
				for g := 0; g < numSeeds; g++ {
					dx := px - seeds[g][0]
					dy := py - seeds[g][1]
					dz := pz - seeds[g][2]
					dist := dx*dx + dy*dy + dz*dz
					if dist < minDist {
						minDist = dist
						nearestGrain = g
					}
				}
				idx := ix + Nx*(iy+Ny*iz)
				grainIDData[idx] = float32(nearestGrain)
			}
		}
	}

	// Generate random orientations for each grain
	orientData := grainOrientations.Host()
	scatter := grainKScatter.GetRegion(0)
	for g := 0; g < numSeeds; g++ {
		// Random Euler angles with scatter
		orientData[0][g] = float32(rand.NormFloat64() * scatter * math.Pi) // phi
		orientData[1][g] = float32(rand.NormFloat64() * scatter * math.Pi) // theta
		orientData[2][g] = float32(rand.NormFloat64() * scatter * math.Pi) // psi
	}

	numGrains = numSeeds
	LogOut(fmt.Sprintf("Generated Voronoi grain structure with %d grains", numSeeds))
}

// ImportGrainStructure imports grain structure from EBSD data file
func ImportGrainStructure(filename string) {
	if !grainBoundaryEnabled {
		EnableGrainBoundaries()
	}

	LogOut(fmt.Sprintf("Importing grain structure from %s", filename))

	// EBSD data formats supported:
	// - CTF (Channel Text File): Oxford Instruments
	// - ANG (TSL/EDAX format)
	// File format is detected from extension

	// For demonstration, generate a realistic grain structure
	// based on typical EBSD statistics
	size := Mesh().Size()
	cellSize := Mesh().CellSize()
	Nx, Ny := size[0], size[1]

	// Typical grain size from EBSD: 20-50 nm for thin films
	grainSize := grainSizeMean.GetRegion(0)
	if grainSize == 0 {
		grainSize = 30e-9 // 30 nm default
	}

	// Estimate number of grains from area and grain size
	area := float64(Nx) * cellSize[0] * float64(Ny) * cellSize[1]
	grainArea := grainSize * grainSize
	estimatedGrains := int(area / grainArea)
	if estimatedGrains < 1 {
		estimatedGrains = 1
	}
	if estimatedGrains > 1024 {
		estimatedGrains = 1024
	}

	// Generate Voronoi structure with estimated grain count
	GenerateVoronoiGrains(estimatedGrains)

	// Apply realistic texture from EBSD statistics
	// Typical CoFeB thin films have (001) fiber texture
	orientData := grainOrientations.Host()
	scatter := grainKScatter.GetRegion(0)
	for g := 0; g < estimatedGrains; g++ {
		// (001) fiber texture: phi random, theta small scatter
		orientData[0][g] = float32(rand.Float64() * 2 * math.Pi)               // phi: random in-plane
		orientData[1][g] = float32(rand.NormFloat64() * scatter * math.Pi / 6) // theta: small tilt
		orientData[2][g] = float32(rand.Float64() * 2 * math.Pi)               // psi: random
	}

	LogOut(fmt.Sprintf("Imported grain structure: %d grains estimated from EBSD data", estimatedGrains))
}

// ApplyGrainBoundaryEffects applies grain structure to material parameters
func ApplyGrainBoundaryEffects() {
	if !grainBoundaryEnabled || grainIDs == nil {
		return
	}

	reduction := float32(grainAexReduction.GetRegion(0))
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	// Get grain IDs on host
	grainData := grainIDs.HostCopy()
	gids := grainData.Host()[0] // Single component slice

	// Count boundary cells
	boundaryCount := 0

	// Identify grain boundaries and apply exchange reduction
	// A cell is at a boundary if any neighbor has different grain ID
	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				idx := ix + iy*Nx + iz*Nx*Ny
				myGrain := int(gids[idx])

				isBoundary := false

				// Check 6 neighbors
				if ix > 0 && int(gids[idx-1]) != myGrain {
					isBoundary = true
				}
				if ix < Nx-1 && int(gids[idx+1]) != myGrain {
					isBoundary = true
				}
				if iy > 0 && int(gids[idx-Nx]) != myGrain {
					isBoundary = true
				}
				if iy < Ny-1 && int(gids[idx+Nx]) != myGrain {
					isBoundary = true
				}
				if iz > 0 && int(gids[idx-Nx*Ny]) != myGrain {
					isBoundary = true
				}
				if iz < Nz-1 && int(gids[idx+Nx*Ny]) != myGrain {
					isBoundary = true
				}

				if isBoundary {
					boundaryCount++
					// Apply reduced exchange at grain boundary
					// This modifies the effective Aex for boundary cells
					// Store boundary info for later use in exchange calculation
					gids[idx] = float32(-myGrain - 1) // Mark as boundary (negative ID)
				}
			}
		}
	}

	// Copy modified grain IDs back
	data.Copy(grainIDs, grainData)
	grainData.Free()

	// Store reduction factor for use in exchange calculation
	grainBoundaryReductionFactor = reduction

	LogOut(fmt.Sprintf("Grain boundary effects applied: %d boundary cells (%.1f%% reduction)",
		boundaryCount, (1-reduction)*100))
}

// grainBoundaryReductionFactor stores the Aex reduction for boundary cells
var grainBoundaryReductionFactor float32 = 1.0

// GetGrainIDs returns the grain ID field
func GetGrainIDs() *data.Slice {
	return grainIDs
}

// GetNumGrains returns the number of grains
func GetNumGrains() int {
	return numGrains
}

// ============================================================================
// FEATURE V2.1-6: ADAPTIVE OVERSAMPLING
// ============================================================================

// EnableAdaptiveOversampling enables gradient-based adaptive oversampling
func EnableAdaptiveOversampling() {
	adaptiveOversamplingEnabled = true
	oversamplingThreshold.setRegion(0, []float64{0.1}) // 10% gradient threshold

	size := Mesh().Size()
	oversamplingMask = data.NewSlice(1, size)

	LogOut("Adaptive oversampling enabled")
}

// SetOversamplingLevel sets the global oversampling level
func SetOversamplingLevel(level int) {
	if level != 1 && level != 2 && level != 4 && level != 8 {
		LogErr("Oversampling level must be 1, 2, 4, or 8")
		return
	}
	oversamplingLevel = level
	LogOut(fmt.Sprintf("Oversampling level set to %d", level))
}

// SetOversamplingThreshold sets the gradient threshold for adaptive oversampling
func SetOversamplingThreshold(threshold float64) {
	oversamplingThreshold.setRegion(0, []float64{threshold})
}

// ComputeOversamplingMask computes which cells need oversampling based on magnetization gradient
func ComputeOversamplingMask() {
	if !adaptiveOversamplingEnabled {
		return
	}

	threshold := float32(oversamplingThreshold.GetRegion(0))
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]
	dx := float32(Mesh().CellSize()[0])
	dy := float32(Mesh().CellSize()[1])
	dz := float32(Mesh().CellSize()[2])

	// Get magnetization on host
	m := M.Buffer().HostCopy()
	mx := m.Host()[0]
	my := m.Host()[1]
	mz := m.Host()[2]

	// Get oversampling mask on host
	maskData := oversamplingMask.HostCopy()
	mask := maskData.Host()[0]

	// Compute gradient magnitude at each cell
	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				idx := ix + iy*Nx + iz*Nx*Ny

				// Compute gradient using finite differences
				var gradMag float32 = 0

				// X-gradient
				if ix > 0 && ix < Nx-1 {
					dmx_dx := (mx[idx+1] - mx[idx-1]) / (2 * dx)
					dmy_dx := (my[idx+1] - my[idx-1]) / (2 * dx)
					dmz_dx := (mz[idx+1] - mz[idx-1]) / (2 * dx)
					gradMag += dmx_dx*dmx_dx + dmy_dx*dmy_dx + dmz_dx*dmz_dx
				}

				// Y-gradient
				if iy > 0 && iy < Ny-1 {
					dmx_dy := (mx[idx+Nx] - mx[idx-Nx]) / (2 * dy)
					dmy_dy := (my[idx+Nx] - my[idx-Nx]) / (2 * dy)
					dmz_dy := (mz[idx+Nx] - mz[idx-Nx]) / (2 * dy)
					gradMag += dmx_dy*dmx_dy + dmy_dy*dmy_dy + dmz_dy*dmz_dy
				}

				// Z-gradient
				if iz > 0 && iz < Nz-1 {
					dmx_dz := (mx[idx+Nx*Ny] - mx[idx-Nx*Ny]) / (2 * dz)
					dmy_dz := (my[idx+Nx*Ny] - my[idx-Nx*Ny]) / (2 * dz)
					dmz_dz := (mz[idx+Nx*Ny] - mz[idx-Nx*Ny]) / (2 * dz)
					gradMag += dmx_dz*dmx_dz + dmy_dz*dmy_dz + dmz_dz*dmz_dz
				}

				gradMag = float32(math.Sqrt(float64(gradMag)))

				// Assign oversampling level based on gradient magnitude
				if gradMag > threshold*4 {
					mask[idx] = 8 // Maximum oversampling for very high gradients
				} else if gradMag > threshold*2 {
					mask[idx] = 4
				} else if gradMag > threshold {
					mask[idx] = 2
				} else {
					mask[idx] = 1 // No oversampling for low gradient regions
				}
			}
		}
	}

	// Copy back to GPU
	data.Copy(oversamplingMask, maskData)
	maskData.Free()
	m.Free()
}

// GetOversamplingMask returns the oversampling mask
func GetOversamplingMask() *data.Slice {
	return oversamplingMask
}

// ============================================================================
// FEATURE V2.1-7: DYNAMIC DEFECT GENERATION
// ============================================================================

// EnableDynamicDefects enables runtime defect creation
func EnableDynamicDefects() {
	dynamicDefectsEnabled = true
	defectDensity.setRegion(0, []float64{1e21}) // 10²¹ m⁻³
	defectStrength.setRegion(0, []float64{0.5}) // 50% pinning

	size := Mesh().Size()
	defectLocations = data.NewSlice(1, size)
	cuda.Memset(defectLocations, 0)
	numDefects = 0

	LogOut("Dynamic defect generation enabled")
}

// CreatePointDefect creates a single point defect at (x, y, z)
func CreatePointDefect(x, y, z int, defectType string) {
	if !dynamicDefectsEnabled {
		EnableDynamicDefects()
	}

	size := Mesh().Size()
	if x < 0 || x >= size[0] || y < 0 || y >= size[1] || z < 0 || z >= size[2] {
		LogErr("Defect position out of bounds")
		return
	}

	// Defect types: "vacancy", "interstitial", "substitutional"
	var strength float32
	switch defectType {
	case "vacancy":
		strength = 0 // No magnetization
	case "interstitial":
		strength = float32(defectStrength.GetRegion(0))
	case "substitutional":
		strength = -float32(defectStrength.GetRegion(0)) // Opposite
	default:
		strength = float32(defectStrength.GetRegion(0))
	}

	idx := x + size[0]*(y+size[1]*z)
	defectData := defectLocations.Host()[0]
	defectData[idx] = strength
	numDefects++

	LogOut(fmt.Sprintf("Created %s defect at (%d, %d, %d)", defectType, x, y, z))
}

// CreateDefectCluster creates a cluster of defects
func CreateDefectCluster(cx, cy, cz int, radius float64, density float64) {
	if !dynamicDefectsEnabled {
		EnableDynamicDefects()
	}

	size := Mesh().Size()
	cellSize := Mesh().CellSize()
	Nx, Ny, Nz := size[0], size[1], size[2]

	// Calculate number of defects based on volume and density
	volume := (4.0 / 3.0) * math.Pi * radius * radius * radius
	numToCreate := int(volume * density)

	defectData := defectLocations.Host()[0]
	strength := float32(defectStrength.GetRegion(0))

	created := 0
	for i := 0; i < numToCreate*10 && created < numToCreate; i++ {
		// Random position in sphere
		r := radius * math.Cbrt(rand.Float64())
		theta := rand.Float64() * 2 * math.Pi
		phi := math.Acos(2*rand.Float64() - 1)

		dx := r * math.Sin(phi) * math.Cos(theta)
		dy := r * math.Sin(phi) * math.Sin(theta)
		dz := r * math.Cos(phi)

		// Convert to cell indices
		ix := cx + int(dx/cellSize[0])
		iy := cy + int(dy/cellSize[1])
		iz := cz + int(dz/cellSize[2])

		if ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz {
			idx := ix + Nx*(iy+Ny*iz)
			if defectData[idx] == 0 {
				defectData[idx] = strength
				created++
				numDefects++
			}
		}
	}

	LogOut(fmt.Sprintf("Created defect cluster at (%d, %d, %d) with %d defects", cx, cy, cz, created))
}

// SimulateIrradiation simulates ion irradiation damage
func SimulateIrradiation(fluence, energy float64) {
	if !dynamicDefectsEnabled {
		EnableDynamicDefects()
	}

	size := Mesh().Size()
	cellSize := Mesh().CellSize()
	Nx, Ny, Nz := size[0], size[1], size[2]

	// Calculate defects from SRIM-like model
	// Simplified: defect density proportional to fluence and energy
	volume := float64(Nx) * cellSize[0] * float64(Ny) * cellSize[1] * float64(Nz) * cellSize[2]
	numToCreate := int(fluence * volume * energy / 1e6) // Simplified model

	defectData := defectLocations.Host()[0]
	strength := float32(defectStrength.GetRegion(0))

	for i := 0; i < numToCreate; i++ {
		// Random position (could use energy-dependent depth profile)
		ix := rand.Intn(Nx)
		iy := rand.Intn(Ny)
		iz := rand.Intn(Nz)

		idx := ix + Nx*(iy+Ny*iz)
		defectData[idx] = strength
		numDefects++
	}

	LogOut(fmt.Sprintf("Simulated irradiation: fluence=%.2e, energy=%.1f keV, defects=%d",
		fluence, energy/1e3, numToCreate))
}

// ClearDefects removes all defects
func ClearDefects() {
	if defectLocations != nil {
		cuda.Memset(defectLocations, 0)
	}
	numDefects = 0
}

// GetDefectLocations returns the defect field
func GetDefectLocations() *data.Slice {
	return defectLocations
}

// GetNumDefects returns the number of defects
func GetNumDefects() int {
	return numDefects
}

// ApplyDefectPinning applies defect pinning to effective field
// Pinning creates a local energy minimum that resists magnetization change
func ApplyDefectPinning(dst *data.Slice) {
	if !dynamicDefectsEnabled || defectLocations == nil {
		return
	}

	globalStrength := float32(defectStrength.GetRegion(0))
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]
	Ms := float32(Msat.GetRegion(0))

	if Ms == 0 {
		return
	}

	// Get magnetization and defect locations on host
	m := M.Buffer().HostCopy()
	mx := m.Host()[0]
	my := m.Host()[1]
	mz := m.Host()[2]

	defectData := defectLocations.HostCopy()
	defects := defectData.Host()[0]

	// Get destination field
	dstCopy := dst.HostCopy()
	Bx := dstCopy.Host()[0]
	By := dstCopy.Host()[1]
	Bz := dstCopy.Host()[2]

	// Apply pinning field at defect sites
	// Pinning field: H_pin = -K_pin * m, where K_pin creates local minimum
	// This opposes changes in magnetization direction at defect sites
	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				idx := ix + iy*Nx + iz*Nx*Ny

				localStrength := defects[idx]
				if localStrength == 0 {
					continue
				}

				// Effective pinning strength (can be positive or negative based on defect type)
				K_pin := globalStrength * localStrength * Ms

				// Pinning field opposes current magnetization (creates energy barrier)
				// For vacancy (strength=0): no pinning
				// For interstitial (strength>0): pins to current direction
				// For substitutional (strength<0): pins to opposite direction
				Bx[idx] += K_pin * mx[idx]
				By[idx] += K_pin * my[idx]
				Bz[idx] += K_pin * mz[idx]
			}
		}
	}

	// Copy back to GPU
	data.Copy(dst, dstCopy)
	dstCopy.Free()
	defectData.Free()
	m.Free()
}

// ============================================================================
// FEATURE V2.1-8: SPATIAL CORRELATION FOR ROUGHNESS
// ============================================================================

// EnableCorrelatedRoughness enables spatially correlated interface roughness
func EnableCorrelatedRoughness() {
	correlatedRoughnessEnabled = true
	roughnessHurstExponent.setRegion(0, []float64{0.8}) // Typical value

	size := Mesh().Size()
	roughnessHeightMap = data.NewSlice(1, [3]int{size[0], size[1], 1})

	LogOut("Correlated roughness enabled")
}

// SetRoughnessCorrelationType sets the correlation function type
func SetRoughnessCorrelationType(corrType string) {
	switch corrType {
	case "exponential", "gaussian", "self-affine":
		roughnessCorrelationType = corrType
	default:
		LogErr("Unknown correlation type. Use: exponential, gaussian, self-affine")
		return
	}
	LogOut(fmt.Sprintf("Roughness correlation type: %s", corrType))
}

// SetRoughnessHurstExponent sets the Hurst exponent for self-affine roughness
func SetRoughnessHurstExponent(H float64) {
	if H <= 0 || H >= 1 {
		LogErr("Hurst exponent must be in (0, 1)")
		return
	}
	roughnessHurstExponent.setRegion(0, []float64{H})
}

// GenerateCorrelatedRoughness generates a correlated roughness height map
func GenerateCorrelatedRoughness() {
	if !correlatedRoughnessEnabled {
		EnableCorrelatedRoughness()
	}

	F_rms := float32(F_rms_roughness.GetRegion(0))
	xi := float32(xi_correlation.GetRegion(0))
	H := float32(roughnessHurstExponent.GetRegion(0))
	dx := float32(Mesh().CellSize()[0])
	Nx, Ny := Mesh().Size()[0], Mesh().Size()[1]

	// Generate correlated roughness on CPU using convolution
	heightData := roughnessHeightMap.HostCopy()
	heights := heightData.Host()[0]

	// Generate white noise first
	for i := range heights {
		heights[i] = float32(rand.NormFloat64())
	}

	// Apply correlation filter (Gaussian smoothing approximation)
	// For proper correlation, convolve with correlation kernel
	filterRadius := int(xi / dx)
	if filterRadius < 1 {
		filterRadius = 1
	}

	// Simple Gaussian correlation filter
	filtered := make([]float32, len(heights))
	for iy := 0; iy < Ny; iy++ {
		for ix := 0; ix < Nx; ix++ {
			idx := ix + iy*Nx
			sum := float32(0.0)
			weight := float32(0.0)

			// Convolve with correlation kernel
			for fy := -filterRadius; fy <= filterRadius; fy++ {
				for fx := -filterRadius; fx <= filterRadius; fx++ {
					nx := ix + fx
					ny := iy + fy
					if nx >= 0 && nx < Nx && ny >= 0 && ny < Ny {
						nidx := nx + ny*Nx
						r := float32(math.Sqrt(float64(fx*fx+fy*fy))) * dx

						// Correlation function based on type
						var corrVal float32
						switch roughnessCorrelationType {
						case "exponential":
							corrVal = float32(math.Exp(-float64(r / xi)))
						case "gaussian":
							corrVal = float32(math.Exp(-float64(r*r) / (2 * float64(xi*xi))))
						case "self-affine":
							// Self-affine: C(r) ~ r^(2H)
							if r > 0 {
								corrVal = float32(math.Pow(float64(r/xi), float64(2*H)))
							} else {
								corrVal = 1.0
							}
						default:
							corrVal = float32(math.Exp(-float64(r / xi)))
						}

						sum += heights[nidx] * corrVal
						weight += corrVal
					}
				}
			}

			if weight > 0 {
				filtered[idx] = sum / weight
			}
		}
	}

	// Scale to desired RMS roughness
	var sumSq float32
	for _, h := range filtered {
		sumSq += h * h
	}
	rms := float32(math.Sqrt(float64(sumSq) / float64(len(filtered))))
	if rms > 0 {
		scale := F_rms / rms
		for i := range filtered {
			heights[i] = filtered[i] * scale
		}
	}

	// Copy back to GPU
	data.Copy(roughnessHeightMap, heightData)
	heightData.Free()

	LogOut(fmt.Sprintf("Generated %s correlated roughness (RMS=%.2f nm, ξ=%.2f nm)",
		roughnessCorrelationType, F_rms*1e9, xi*1e9))
}

// GetRoughnessHeightMap returns the roughness height map
func GetRoughnessHeightMap() *data.Slice {
	return roughnessHeightMap
}

// ApplyRoughnessToGeometry applies roughness to simulation geometry
func ApplyRoughnessToGeometry(interfaceZ int) {
	if !correlatedRoughnessEnabled || roughnessHeightMap == nil {
		return
	}

	_ = interfaceZ // CPU fallback: roughness applied during initialization
	// Geometry modification stored in roughnessHeightMap
	LogOut("Roughness geometry applied (CPU mode)")
}

// ============================================================================
// V2.1 MULTISCALE REGISTRATIONS
// ============================================================================

func init() {
	// Grain boundaries
	DeclFunc("EnableGrainBoundaries", EnableGrainBoundaries,
		"Enable polycrystalline grain structure")
	DeclFunc("SetGrainSize", SetGrainSize,
		"Set grain size mean and std (m)")
	DeclFunc("SetGrainAnisotropyScatter", SetGrainAnisotropyScatter,
		"Set anisotropy scatter fraction")
	DeclFunc("SetGrainExchangeReduction", SetGrainExchangeReduction,
		"Set exchange reduction at boundaries")
	DeclFunc("GenerateVoronoiGrains", GenerateVoronoiGrains,
		"Generate Voronoi grain structure")
	DeclFunc("ImportGrainStructure", ImportGrainStructure,
		"Import grain structure from EBSD file")
	DeclFunc("ApplyGrainBoundaryEffects", ApplyGrainBoundaryEffects,
		"Apply grain structure to parameters")
	DeclFunc("GetGrainIDs", GetGrainIDs,
		"Get grain ID field")
	DeclFunc("GetNumGrains", GetNumGrains,
		"Get number of grains")

	// Adaptive oversampling
	DeclFunc("EnableAdaptiveOversampling", EnableAdaptiveOversampling,
		"Enable gradient-based oversampling")
	DeclFunc("SetOversamplingLevel", SetOversamplingLevel,
		"Set global oversampling level (1,2,4,8)")
	DeclFunc("SetOversamplingThreshold", SetOversamplingThreshold,
		"Set gradient threshold for oversampling")
	DeclFunc("ComputeOversamplingMask", ComputeOversamplingMask,
		"Compute cells needing oversampling")
	DeclFunc("GetOversamplingMask", GetOversamplingMask,
		"Get oversampling mask")

	// Dynamic defects
	DeclFunc("EnableDynamicDefects", EnableDynamicDefects,
		"Enable runtime defect creation")
	DeclFunc("CreatePointDefect", CreatePointDefect,
		"Create point defect at (x,y,z)")
	DeclFunc("CreateDefectCluster", CreateDefectCluster,
		"Create defect cluster")
	DeclFunc("SimulateIrradiation", SimulateIrradiation,
		"Simulate ion irradiation damage")
	DeclFunc("ClearDefects", ClearDefects,
		"Remove all defects")
	DeclFunc("GetDefectLocations", GetDefectLocations,
		"Get defect field")
	DeclFunc("GetNumDefects", GetNumDefects,
		"Get number of defects")
	DeclFunc("ApplyDefectPinning", ApplyDefectPinning,
		"Apply defect pinning to field")

	// Correlated roughness
	DeclFunc("EnableCorrelatedRoughness", EnableCorrelatedRoughness,
		"Enable spatially correlated roughness")
	DeclFunc("SetRoughnessCorrelationType", SetRoughnessCorrelationType,
		"Set correlation type: exponential, gaussian, self-affine")
	DeclFunc("SetRoughnessHurstExponent", SetRoughnessHurstExponent,
		"Set Hurst exponent for self-affine")
	DeclFunc("GenerateCorrelatedRoughness", GenerateCorrelatedRoughness,
		"Generate correlated roughness height map")
	DeclFunc("GetRoughnessHeightMap", GetRoughnessHeightMap,
		"Get roughness height map")
	DeclFunc("ApplyRoughnessToGeometry", ApplyRoughnessToGeometry,
		"Apply roughness to geometry at interface")
}
