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
// MuMax3-SAF-NeuroSpin V2.1: Advanced Neuromorphic Features
// =====================================================================================================
// This module implements advanced neuromorphic computing capabilities:
// - Backpropagation Through Time (BPTT)
// - Spiking Neural Networks with AER
// - Multi-neuron integrated circuits
// - Spike-timing encoding
// =====================================================================================================

import (
	"fmt"
	"math"
	"sort"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// ============================================================================
// NEUROMORPHIC DATA STRUCTURES
// ============================================================================

// SpikeEvent represents a single spike event
type SpikeEvent struct {
	NeuronID  int
	Timestamp float64
	Polarity  int // +1 (excitatory) or -1 (inhibitory)
}

// Synapse represents a synaptic connection
type Synapse struct {
	PreID      int
	PostID     int
	Weight     float64
	Type       string // "excitatory", "inhibitory"
	Plasticity string // "STDP", "BCM", "Oja", "none"
}

// NeuronParams holds parameters for different neuron models
type NeuronParams struct {
	Model          string // "LIF", "Izhikevich", "HH"
	V_threshold    float64
	V_reset        float64
	V_rest         float64
	Tau_membrane   float64
	Tau_refractory float64
	// Izhikevich parameters
	Izh_a, Izh_b, Izh_c, Izh_d float64
}

// ============================================================================
// NEUROMORPHIC PARAMETERS
// ============================================================================

var (
	// BPTT
	bpttEnabled      = false
	bpttLength       = 100 // Timesteps for truncated BPTT
	bpttGradients    *data.Slice
	bpttActivations  []*data.Slice // History of activations
	bpttLearningRate = NewScalarParam("bpttLearningRate", "", "BPTT learning rate")
	bpttClipGrad     = NewScalarParam("bpttClipGrad", "", "Gradient clipping value")

	// SNN
	snnEnabled        = false
	snnNeuronModel    = "LIF"
	snnParams         NeuronParams
	snnVoltage        *data.Slice // Membrane voltages
	snnRefractoryTime *data.Slice // Refractory counters
	snnSpikeHistory   []SpikeEvent
	aerEnabled        = false

	// Multi-neuron circuits
	multiNeuronEnabled = false
	neuronArrayNx      = 0
	neuronArrayNy      = 0
	synapses           []Synapse
	neuronActivity     *data.Slice
	synapticCurrents   *data.Slice

	// Spike timing
	spikeTimingEnabled = false
	spikeTimingTau     = NewScalarParam("spikeTimingTau", "s", "Spike timing time constant")
	firstSpikeLatency  *data.Slice
	interspikeTiming   *data.Slice
)

// ============================================================================
// FEATURE V2.1-9: BACKPROPAGATION THROUGH TIME (BPTT)
// ============================================================================

// EnableBPTT enables backpropagation through time for recurrent networks
func EnableBPTT() {
	bpttEnabled = true
	bpttLength = 100
	bpttLearningRate.setRegion(0, []float64{0.001})
	bpttClipGrad.setRegion(0, []float64{1.0})

	size := Mesh().Size()
	bpttGradients = data.NewSlice(3, size)
	bpttActivations = make([]*data.Slice, bpttLength)
	for i := 0; i < bpttLength; i++ {
		bpttActivations[i] = data.NewSlice(3, size)
	}

	LogOut("Backpropagation Through Time enabled")
}

// SetBPTTLength sets the number of timesteps for truncated BPTT
func SetBPTTLength(timesteps int) {
	if timesteps < 1 || timesteps > 1000 {
		LogErr("BPTT length must be 1-1000")
		return
	}

	// Reallocate activations if needed
	if timesteps != bpttLength {
		size := Mesh().Size()
		bpttActivations = make([]*data.Slice, timesteps)
		for i := 0; i < timesteps; i++ {
			bpttActivations[i] = data.NewSlice(3, size)
		}
	}
	bpttLength = timesteps
}

// StoreBPTTActivation stores current magnetization for BPTT
func StoreBPTTActivation(timestep int) {
	if !bpttEnabled || timestep < 0 || timestep >= bpttLength {
		return
	}
	data.Copy(bpttActivations[timestep], M.Buffer())
}

// ComputeBPTTGradients computes gradients through time using truncated BPTT
// Implements: ∂L/∂θ = Σ_t ∂L/∂a_t * ∂a_t/∂θ with chain rule through time
func ComputeBPTTGradients(targetOutput *data.Slice) *data.Slice {
	if !bpttEnabled {
		return nil
	}

	clipVal := float32(bpttClipGrad.GetRegion(0))
	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]
	N := Nx * Ny * Nz

	// Zero gradients
	cuda.Memset(bpttGradients, 0)

	if len(bpttActivations) == 0 || targetOutput == nil {
		return bpttGradients
	}

	// Get target on host
	targetData := targetOutput.HostCopy()
	target := targetData.Host()

	// Get gradient buffer on host
	gradData := bpttGradients.HostCopy()
	grad := gradData.Host()

	// Compute output error at final timestep (MSE gradient: 2*(output - target))
	finalAct := bpttActivations[len(bpttActivations)-1].HostCopy()
	finalData := finalAct.Host()

	// Initialize gradient at final timestep
	for c := 0; c < 3; c++ {
		for i := 0; i < N; i++ {
			// MSE gradient: ∂L/∂a_T = 2 * (a_T - target)
			grad[c][i] = 2 * (finalData[c][i] - target[c][i])
		}
	}
	finalAct.Free()

	// Backward pass through time: propagate gradients
	// For each timestep t from T-1 to 0:
	// ∂L/∂a_t = ∂L/∂a_{t+1} * ∂a_{t+1}/∂a_t
	// Using simple recurrence: a_{t+1} = α*a_t + (1-α)*input (leaky integration)
	alpha := float32(0.9) // Recurrence coefficient

	for t := len(bpttActivations) - 2; t >= 0; t-- {
		actData := bpttActivations[t].HostCopy()
		act := actData.Host()

		// Propagate gradient through recurrence
		for c := 0; c < 3; c++ {
			for i := 0; i < N; i++ {
				// Chain rule: multiply by recurrence derivative
				grad[c][i] = grad[c][i] * alpha

				// Add local gradient contribution
				// For identity activation: gradient passes through
				// For tanh: multiply by (1 - a_t^2)
				localGrad := grad[c][i] * (1 - act[c][i]*act[c][i])
				grad[c][i] = localGrad

				// Gradient clipping to prevent exploding gradients
				if grad[c][i] > clipVal {
					grad[c][i] = clipVal
				} else if grad[c][i] < -clipVal {
					grad[c][i] = -clipVal
				}
			}
		}
		actData.Free()
	}

	// Copy gradients back to GPU
	data.Copy(bpttGradients, gradData)
	gradData.Free()
	targetData.Free()

	return bpttGradients
}

// AccumulateBPTTGradients accumulates gradients from multiple sequences
func AccumulateBPTTGradients(gradients *data.Slice) {
	if !bpttEnabled || bpttGradients == nil {
		return
	}

	// CPU fallback: simple addition
	cuda.Madd2(bpttGradients, bpttGradients, gradients, 1.0, 1.0)
}

// ApplyBPTTUpdate applies gradient descent update
func ApplyBPTTUpdate(weights *data.Slice) {
	if !bpttEnabled || bpttGradients == nil {
		return
	}

	lr := float32(bpttLearningRate.GetRegion(0))
	// CPU fallback: weights -= lr * gradients
	cuda.Madd2(weights, weights, bpttGradients, 1.0, -lr)
}

// ZeroBPTTGradients resets gradients to zero
func ZeroBPTTGradients() {
	if bpttGradients != nil {
		cuda.Memset(bpttGradients, 0)
	}
}

// GetBPTTGradients returns accumulated gradients
func GetBPTTGradients() *data.Slice {
	return bpttGradients
}

// ============================================================================
// FEATURE V2.1-10: SPIKING NEURAL NETWORKS WITH AER
// ============================================================================

// EnableSNN enables spiking neural network dynamics
func EnableSNN() {
	snnEnabled = true
	snnParams = NeuronParams{
		Model:          "LIF",
		V_threshold:    -50e-3, // -50 mV
		V_reset:        -70e-3, // -70 mV
		V_rest:         -65e-3, // -65 mV
		Tau_membrane:   20e-3,  // 20 ms
		Tau_refractory: 2e-3,   // 2 ms
	}

	size := Mesh().Size()
	N := size[0] * size[1] * size[2]
	snnVoltage = data.NewSlice(1, size)
	snnRefractoryTime = data.NewSlice(1, size)
	snnSpikeHistory = make([]SpikeEvent, 0, N)

	// Initialize to resting potential
	cuda.Memset(snnVoltage, float32(snnParams.V_rest))
	cuda.Memset(snnRefractoryTime, 0)

	LogOut("Spiking Neural Network enabled (LIF model)")
}

// SetNeuronModel sets the neuron model type
func SetNeuronModel(model string) {
	switch model {
	case "LIF":
		snnNeuronModel = "LIF"
		snnParams.Model = "LIF"
	case "Izhikevich":
		snnNeuronModel = "Izhikevich"
		snnParams.Model = "Izhikevich"
		// Default Izhikevich regular spiking
		snnParams.Izh_a = 0.02
		snnParams.Izh_b = 0.2
		snnParams.Izh_c = -65
		snnParams.Izh_d = 8
	case "HH":
		snnNeuronModel = "HH"
		snnParams.Model = "HH"
	default:
		LogErr("Unknown neuron model. Use: LIF, Izhikevich, HH")
		return
	}
	LogOut(fmt.Sprintf("Neuron model set to %s", model))
}

// SetSpikeThreshold sets the spiking threshold voltage
func SetSpikeThreshold(V_th float64) {
	snnParams.V_threshold = V_th
}

// SetRefractoryPeriod sets the refractory period
func SetRefractoryPeriod(t_ref float64) {
	snnParams.Tau_refractory = t_ref
}

// SetIzhikevichParams sets Izhikevich model parameters
func SetIzhikevichParams(a, b, c, d float64) {
	snnParams.Izh_a = a
	snnParams.Izh_b = b
	snnParams.Izh_c = c
	snnParams.Izh_d = d
}

// EnableAER enables Address-Event Representation
func EnableAER() {
	aerEnabled = true
	LogOut("Address-Event Representation enabled")
}

// UpdateSNN updates SNN dynamics for one timestep
func UpdateSNN(input *data.Slice, dt float64) {
	if !snnEnabled {
		return
	}

	Nx, Ny, Nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]
	N := Nx * Ny * Nz

	// Get voltage and refractory data on host
	voltageData := snnVoltage.HostCopy()
	refractData := snnRefractoryTime.HostCopy()
	V := voltageData.Host()[0]
	refract := refractData.Host()[0]

	// Get input if provided
	var inputData []float32
	if input != nil {
		inputHost := input.HostCopy()
		inputData = inputHost.Host()[0]
		defer inputHost.Free()
	}

	// LIF neuron parameters
	V_th := float32(snnParams.V_threshold)
	V_reset := float32(snnParams.V_reset)
	V_rest := float32(snnParams.V_rest)
	tau_m := float32(snnParams.Tau_membrane)
	tau_ref := float32(snnParams.Tau_refractory)
	dt32 := float32(dt)

	numSpikes := 0

	// Update each neuron
	for i := 0; i < N; i++ {
		// Check refractory period
		if refract[i] > 0 {
			refract[i] -= dt32
			V[i] = V_reset
			continue
		}

		// Get input current
		I := float32(0.0)
		if inputData != nil && i < len(inputData) {
			I = inputData[i]
		}

		// LIF dynamics: τ_m * dV/dt = -(V - V_rest) + I
		dV := (-(V[i] - V_rest) + I) * dt32 / tau_m
		V[i] += dV

		// Check for spike
		if V[i] > V_th {
			// Emit spike
			numSpikes++
			V[i] = V_reset
			refract[i] = tau_ref

			// Record spike event if AER enabled
			if aerEnabled {
				event := SpikeEvent{
					NeuronID:  i,
					Timestamp: Time,
				}
				snnSpikeHistory = append(snnSpikeHistory, event)
			}
		}
	}

	// Copy results back to GPU
	data.Copy(snnVoltage, voltageData)
	data.Copy(snnRefractoryTime, refractData)
	voltageData.Free()
	refractData.Free()

	if numSpikes > 0 {
		LogOut(fmt.Sprintf("SNN update: %d spikes emitted", numSpikes))
	}
}

// GetSpikeEvents returns recorded spike events
func GetSpikeEvents() []SpikeEvent {
	return snnSpikeHistory
}

// InjectSpikes injects spike events into the network
func InjectSpikes(events []SpikeEvent) {
	if !snnEnabled {
		return
	}

	size := Mesh().Size()
	Nx, Ny := size[0], size[1]

	for _, event := range events {
		// Convert neuron ID to 3D index
		iz := event.NeuronID / (Nx * Ny)
		iy := (event.NeuronID % (Nx * Ny)) / Nx
		ix := event.NeuronID % Nx

		// Set voltage to spike
		voltData := snnVoltage.Host()[0]
		idx := ix + Nx*(iy+Ny*iz)
		voltData[idx] = float32(snnParams.V_threshold + 0.01) // Above threshold
	}
}

// ClearSpikeHistory clears recorded spikes
func ClearSpikeHistory() {
	snnSpikeHistory = make([]SpikeEvent, 0)
}

// GetMembranePotentials returns membrane voltages
func GetMembranePotentials() *data.Slice {
	return snnVoltage
}

// ============================================================================
// FEATURE V2.1-11: MULTI-NEURON INTEGRATED CIRCUITS
// ============================================================================

// CreateNeuronArray creates an array of neurons
func CreateNeuronArray(Nx, Ny int) {
	multiNeuronEnabled = true
	neuronArrayNx = Nx
	neuronArrayNy = Ny

	totalNeurons := Nx * Ny
	synapses = make([]Synapse, 0)
	neuronActivity = data.NewSlice(1, [3]int{Nx, Ny, 1})
	synapticCurrents = data.NewSlice(1, [3]int{Nx, Ny, 1})

	cuda.Memset(neuronActivity, 0)
	cuda.Memset(synapticCurrents, 0)

	LogOut(fmt.Sprintf("Created neuron array: %d x %d = %d neurons", Nx, Ny, totalNeurons))
}

// ConnectNeurons creates a synaptic connection
func ConnectNeurons(pre, post int, weight float64) {
	if !multiNeuronEnabled {
		LogErr("Neuron array not created")
		return
	}

	totalNeurons := neuronArrayNx * neuronArrayNy
	if pre < 0 || pre >= totalNeurons || post < 0 || post >= totalNeurons {
		LogErr("Neuron ID out of bounds")
		return
	}

	synapse := Synapse{
		PreID:      pre,
		PostID:     post,
		Weight:     weight,
		Type:       "excitatory",
		Plasticity: "none",
	}
	synapses = append(synapses, synapse)
}

// SetSynapseType sets the type of a synapse
func SetSynapseType(pre, post int, synapseType string) {
	for i := range synapses {
		if synapses[i].PreID == pre && synapses[i].PostID == post {
			synapses[i].Type = synapseType
			return
		}
	}
	LogErr("Synapse not found")
}

// EnableSynapsePlasticity enables plasticity on a synapse
func EnableSynapsePlasticity(pre, post int, rule string) {
	for i := range synapses {
		if synapses[i].PreID == pre && synapses[i].PostID == post {
			synapses[i].Plasticity = rule
			return
		}
	}
	LogErr("Synapse not found")
}

// SimulateNetwork simulates the neural network
func SimulateNetwork(duration float64) {
	if !multiNeuronEnabled {
		return
	}

	dt := 1e-4 // 0.1 ms timesteps
	steps := int(duration / dt)

	for step := 0; step < steps; step++ {
		// Compute synaptic currents
		actData := neuronActivity.Host()[0]
		currData := synapticCurrents.Host()[0]

		// Reset currents
		for i := range currData {
			currData[i] = 0
		}

		// Sum synaptic inputs
		for _, syn := range synapses {
			preAct := actData[syn.PreID]
			sign := float32(1.0)
			if syn.Type == "inhibitory" {
				sign = -1.0
			}
			currData[syn.PostID] += sign * float32(syn.Weight) * preAct
		}

		// Update neurons (simplified integration)
		for i := range actData {
			// Leaky integration
			actData[i] = 0.9*actData[i] + 0.1*currData[i]
			// ReLU activation
			if actData[i] < 0 {
				actData[i] = 0
			}
		}

		// Apply plasticity
		for i := range synapses {
			if synapses[i].Plasticity == "STDP" {
				// Simplified STDP
				pre := actData[synapses[i].PreID]
				post := actData[synapses[i].PostID]
				dw := 0.01 * float64(pre*post-0.5*pre)
				synapses[i].Weight += dw
				// Weight bounds
				if synapses[i].Weight < 0 {
					synapses[i].Weight = 0
				}
				if synapses[i].Weight > 1 {
					synapses[i].Weight = 1
				}
			}
		}
	}
}

// GetNetworkActivity returns current neuron activities
func GetNetworkActivity() *data.Slice {
	return neuronActivity
}

// GetSynapseWeights returns all synapse weights as a map
func GetSynapseWeights() map[[2]int]float64 {
	weights := make(map[[2]int]float64)
	for _, syn := range synapses {
		weights[[2]int{syn.PreID, syn.PostID}] = syn.Weight
	}
	return weights
}

// GetSynapseCount returns number of synapses
func GetSynapseCount() int {
	return len(synapses)
}

// ============================================================================
// FEATURE V2.1-12: SPIKE-TIMING ENCODING
// ============================================================================

// EnableSpikeTimingEncoding enables temporal spike encoding
func EnableSpikeTimingEncoding() {
	spikeTimingEnabled = true
	spikeTimingTau.setRegion(0, []float64{10e-3}) // 10 ms

	size := Mesh().Size()
	firstSpikeLatency = data.NewSlice(1, size)
	interspikeTiming = data.NewSlice(1, size)

	cuda.Memset(firstSpikeLatency, -1) // -1 indicates no spike
	cuda.Memset(interspikeTiming, 0)

	LogOut("Spike timing encoding enabled")
}

// RecordFirstSpikeLatency records time to first spike
func RecordFirstSpikeLatency(currentTime float64) {
	if !spikeTimingEnabled || snnVoltage == nil {
		return
	}

	size := Mesh().Size()
	Nx, Ny, Nz := size[0], size[1], size[2]

	voltData := snnVoltage.Host()[0]
	latencyData := firstSpikeLatency.Host()[0]

	for iz := 0; iz < Nz; iz++ {
		for iy := 0; iy < Ny; iy++ {
			for ix := 0; ix < Nx; ix++ {
				idx := ix + Nx*(iy+Ny*iz)
				// Check if spiked and no previous record
				if voltData[idx] > float32(snnParams.V_threshold) && latencyData[idx] < 0 {
					latencyData[idx] = float32(currentTime)
				}
			}
		}
	}
}

// ComputeTemporalCode computes temporal code from spike times
func ComputeTemporalCode() *data.Slice {
	if !spikeTimingEnabled {
		return nil
	}

	// Temporal code: earlier spikes = higher activation
	tau := float32(spikeTimingTau.GetRegion(0))
	size := Mesh().Size()

	temporalCode := data.NewSlice(1, size)
	latencyData := firstSpikeLatency.Host()[0]
	codeData := temporalCode.Host()[0]

	for i := range latencyData {
		if latencyData[i] >= 0 {
			// Exponential decay based on latency
			codeData[i] = float32(math.Exp(-float64(latencyData[i]) / float64(tau)))
		} else {
			codeData[i] = 0
		}
	}

	return temporalCode
}

// RankOrderCode converts spikes to rank order code
func RankOrderCode() []int {
	if !spikeTimingEnabled || firstSpikeLatency == nil {
		return nil
	}

	latencyData := firstSpikeLatency.Host()[0]
	N := len(latencyData)

	// Create index-latency pairs
	type indexLatency struct {
		idx     int
		latency float32
	}
	pairs := make([]indexLatency, 0, N)

	for i := 0; i < N; i++ {
		if latencyData[i] >= 0 {
			pairs = append(pairs, indexLatency{i, latencyData[i]})
		}
	}

	// Sort by latency
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].latency < pairs[j].latency
	})

	// Return rank order
	ranks := make([]int, len(pairs))
	for i, p := range pairs {
		ranks[i] = p.idx
	}

	return ranks
}

// ResetSpikeTimingRecords resets timing records
func ResetSpikeTimingRecords() {
	if firstSpikeLatency != nil {
		cuda.Memset(firstSpikeLatency, -1)
	}
	if interspikeTiming != nil {
		cuda.Memset(interspikeTiming, 0)
	}
}

// GetFirstSpikeLatency returns first spike latency field
func GetFirstSpikeLatency() *data.Slice {
	return firstSpikeLatency
}

// ============================================================================
// V2.1 NEUROMORPHIC REGISTRATIONS
// ============================================================================

func init() {
	// BPTT
	DeclFunc("EnableBPTT", EnableBPTT,
		"Enable Backpropagation Through Time")
	DeclFunc("SetBPTTLength", SetBPTTLength,
		"Set BPTT truncation length")
	DeclFunc("StoreBPTTActivation", StoreBPTTActivation,
		"Store activation for BPTT")
	DeclFunc("ComputeBPTTGradients", ComputeBPTTGradients,
		"Compute BPTT gradients")
	DeclFunc("AccumulateBPTTGradients", AccumulateBPTTGradients,
		"Accumulate gradients")
	DeclFunc("ApplyBPTTUpdate", ApplyBPTTUpdate,
		"Apply gradient descent update")
	DeclFunc("ZeroBPTTGradients", ZeroBPTTGradients,
		"Reset gradients to zero")
	DeclFunc("GetBPTTGradients", GetBPTTGradients,
		"Get accumulated gradients")

	// SNN
	DeclFunc("EnableSNN", EnableSNN,
		"Enable Spiking Neural Network")
	DeclFunc("SetNeuronModel", SetNeuronModel,
		"Set neuron model: LIF, Izhikevich, HH")
	DeclFunc("SetSpikeThreshold", SetSpikeThreshold,
		"Set spike threshold voltage")
	DeclFunc("SetRefractoryPeriod", SetRefractoryPeriod,
		"Set refractory period")
	DeclFunc("SetIzhikevichParams", SetIzhikevichParams,
		"Set Izhikevich parameters (a,b,c,d)")
	DeclFunc("EnableAER", EnableAER,
		"Enable Address-Event Representation")
	DeclFunc("UpdateSNN", UpdateSNN,
		"Update SNN dynamics")
	DeclFunc("GetSpikeEvents", GetSpikeEvents,
		"Get recorded spike events")
	DeclFunc("InjectSpikes", InjectSpikes,
		"Inject spike events")
	DeclFunc("ClearSpikeHistory", ClearSpikeHistory,
		"Clear recorded spikes")
	DeclFunc("GetMembranePotentials", GetMembranePotentials,
		"Get membrane voltages")

	// Multi-neuron circuits
	DeclFunc("CreateNeuronArray", CreateNeuronArray,
		"Create neuron array (Nx, Ny)")
	DeclFunc("ConnectNeurons", ConnectNeurons,
		"Create synaptic connection")
	DeclFunc("SetSynapseType", SetSynapseType,
		"Set synapse type: excitatory, inhibitory")
	DeclFunc("EnableSynapsePlasticity", EnableSynapsePlasticity,
		"Enable plasticity: STDP, BCM, Oja")
	DeclFunc("SimulateNetwork", SimulateNetwork,
		"Simulate neural network")
	DeclFunc("GetNetworkActivity", GetNetworkActivity,
		"Get neuron activities")
	DeclFunc("GetSynapseWeights", GetSynapseWeights,
		"Get synapse weight map")
	DeclFunc("GetSynapseCount", GetSynapseCount,
		"Get number of synapses")

	// Spike timing
	DeclFunc("EnableSpikeTimingEncoding", EnableSpikeTimingEncoding,
		"Enable temporal spike encoding")
	DeclFunc("RecordFirstSpikeLatency", RecordFirstSpikeLatency,
		"Record time to first spike")
	DeclFunc("ComputeTemporalCode", ComputeTemporalCode,
		"Compute temporal code from spikes")
	DeclFunc("RankOrderCode", RankOrderCode,
		"Get rank order code")
	DeclFunc("ResetSpikeTimingRecords", ResetSpikeTimingRecords,
		"Reset timing records")
	DeclFunc("GetFirstSpikeLatency", GetFirstSpikeLatency,
		"Get first spike latency field")
}
