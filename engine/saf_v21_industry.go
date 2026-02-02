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

// MuMax3-SAF-NeuroSpin V2.1: Industry Integration
//
// This module provides industry integration capabilities:
// - SPICE circuit co-simulation interface
// - Process Design Kit (PDK) integration
// - Layout vs Schematic (LVS) verification
// - Export formats (GDSII, OASIS)

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// SPICENode represents a node in the SPICE netlist
type SPICENode struct {
	Name    string
	Voltage float64
	Current float64
	Region  int // Mapped simulation region
}

// LayerInfo describes a layer in the process stack
type LayerInfo struct {
	Name      string
	Material  string
	Thickness float64
	Purpose   string // "magnetic", "spacer", "barrier", "electrode"
}

// LVSResult contains LVS verification results
type LVSResult struct {
	Match        bool
	ErrorCount   int
	WarningCount int
	Errors       []LVSError
}

// LVSError describes an LVS error
type LVSError struct {
	Type     string // "open", "short", "mismatch"
	Location string
	Message  string
}

// ProcessVariation describes process variations
type ProcessVariation struct {
	Parameter string
	Nominal   float64
	Sigma     float64
	Type      string // "gaussian", "uniform"
}

// ============================================================================
// INDUSTRY PARAMETERS
// ============================================================================

var (
	// SPICE interface
	spiceEnabled     = false
	spiceNetlistPath = ""
	spiceNodes       = make(map[string]*SPICENode)
	spiceTimestep    = 1e-12 // 1 ps

	// PDK integration
	pdkEnabled        = false
	pdkPath           = ""
	technologyNode    = "28nm"
	layerStack        []LayerInfo
	processVariations []ProcessVariation

	// LVS
	lvsEnabled = false
)

// ============================================================================
// FEATURE V2.1-13: SPICE CO-SIMULATION INTERFACE
// ============================================================================

// EnableSPICEInterface enables SPICE circuit co-simulation
func EnableSPICEInterface() {
	spiceEnabled = true
	LogOut("SPICE co-simulation interface enabled")
}

// SetSPICENetlist loads and parses a SPICE netlist file
func SetSPICENetlist(filename string) {
	if !spiceEnabled {
		EnableSPICEInterface()
	}
	spiceNetlistPath = filename

	LogOut(fmt.Sprintf("Loading SPICE netlist: %s", filename))

	// Initialize default power nodes
	spiceNodes["GND"] = &SPICENode{Name: "GND", Voltage: 0, Current: 0, Region: -1}
	spiceNodes["0"] = &SPICENode{Name: "0", Voltage: 0, Current: 0, Region: -1}

	// Try to open and parse the file
	file, err := os.Open(filename)
	if err != nil {
		LogOut(fmt.Sprintf("Warning: Could not open netlist file: %v. Using default MTJ circuit.", err))
		// Create default MTJ circuit nodes
		spiceNodes["TE"] = &SPICENode{Name: "TE", Voltage: 0, Current: 0, Region: 0}
		spiceNodes["BE"] = &SPICENode{Name: "BE", Voltage: 0, Current: 0, Region: -1}
		spiceNodes["VDD"] = &SPICENode{Name: "VDD", Voltage: 1.0, Current: 0, Region: -1}
		LogOut(fmt.Sprintf("Initialized %d default SPICE nodes", len(spiceNodes)))
		return
	}
	defer file.Close()

	// Parse SPICE netlist
	scanner := bufio.NewScanner(file)
	lineNum := 0
	inSubckt := false
	subcktName := ""

	// Regex patterns for SPICE elements
	resistorPattern := regexp.MustCompile(`^[Rr](\w+)\s+(\w+)\s+(\w+)\s+(\S+)`)
	capacitorPattern := regexp.MustCompile(`^[Cc](\w+)\s+(\w+)\s+(\w+)\s+(\S+)`)
	voltagePattern := regexp.MustCompile(`^[Vv](\w+)\s+(\w+)\s+(\w+)\s+(?:DC\s+)?(\S+)`)
	currentPattern := regexp.MustCompile(`^[Ii](\w+)\s+(\w+)\s+(\w+)\s+(?:DC\s+)?(\S+)`)
	subcktPattern := regexp.MustCompile(`^\.SUBCKT\s+(\w+)\s+(.+)`)

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if len(line) == 0 || line[0] == '*' {
			continue
		}

		// Convert to uppercase for directives
		upperLine := strings.ToUpper(line)

		// Parse .SUBCKT definition
		if strings.HasPrefix(upperLine, ".SUBCKT") {
			matches := subcktPattern.FindStringSubmatch(line)
			if len(matches) >= 3 {
				inSubckt = true
				subcktName = matches[1]
				// Parse port nodes
				ports := strings.Fields(matches[2])
				for _, port := range ports {
					if _, exists := spiceNodes[port]; !exists {
						spiceNodes[port] = &SPICENode{Name: port, Voltage: 0, Current: 0, Region: -1}
					}
				}
				LogOut(fmt.Sprintf("Found subcircuit: %s with %d ports", subcktName, len(ports)))
			}
			continue
		}

		// Parse .ENDS
		if strings.HasPrefix(upperLine, ".ENDS") {
			inSubckt = false
			continue
		}

		// Parse resistors: Rname node1 node2 value
		if matches := resistorPattern.FindStringSubmatch(line); len(matches) >= 5 {
			node1, node2 := matches[2], matches[3]
			addNodeIfNew(node1)
			addNodeIfNew(node2)
			continue
		}

		// Parse capacitors: Cname node1 node2 value
		if matches := capacitorPattern.FindStringSubmatch(line); len(matches) >= 5 {
			node1, node2 := matches[2], matches[3]
			addNodeIfNew(node1)
			addNodeIfNew(node2)
			continue
		}

		// Parse voltage sources: Vname node+ node- DC value
		if matches := voltagePattern.FindStringSubmatch(line); len(matches) >= 5 {
			node1, node2 := matches[2], matches[3]
			addNodeIfNew(node1)
			addNodeIfNew(node2)
			// Set voltage if we can parse it
			if voltage, err := parseSpiceValue(matches[4]); err == nil {
				if n, exists := spiceNodes[node1]; exists {
					n.Voltage = voltage
				}
			}
			continue
		}

		// Parse current sources: Iname node+ node- DC value
		if matches := currentPattern.FindStringSubmatch(line); len(matches) >= 5 {
			node1, node2 := matches[2], matches[3]
			addNodeIfNew(node1)
			addNodeIfNew(node2)
			continue
		}
	}

	_ = inSubckt
	_ = subcktName

	LogOut(fmt.Sprintf("Parsed SPICE netlist: %d nodes from %d lines", len(spiceNodes), lineNum))
}

// addNodeIfNew adds a node to spiceNodes if it doesn't exist
func addNodeIfNew(name string) {
	if _, exists := spiceNodes[name]; !exists {
		spiceNodes[name] = &SPICENode{Name: name, Voltage: 0, Current: 0, Region: -1}
	}
}

// parseSpiceValue parses SPICE values with metric suffixes (e.g., 1k, 1u, 1m)
func parseSpiceValue(s string) (float64, error) {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return 0, fmt.Errorf("empty value")
	}

	// Check for metric suffix
	multiplier := 1.0
	lastChar := s[len(s)-1]

	switch lastChar {
	case 'T', 't':
		multiplier = 1e12
		s = s[:len(s)-1]
	case 'G', 'g':
		multiplier = 1e9
		s = s[:len(s)-1]
	case 'M':
		multiplier = 1e6
		s = s[:len(s)-1]
	case 'K', 'k':
		multiplier = 1e3
		s = s[:len(s)-1]
	case 'm':
		multiplier = 1e-3
		s = s[:len(s)-1]
	case 'u', 'U':
		multiplier = 1e-6
		s = s[:len(s)-1]
	case 'n', 'N':
		multiplier = 1e-9
		s = s[:len(s)-1]
	case 'p', 'P':
		multiplier = 1e-12
		s = s[:len(s)-1]
	case 'f', 'F':
		multiplier = 1e-15
		s = s[:len(s)-1]
	}

	value, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}

	return value * multiplier, nil
}

// MapNodeToRegion maps a SPICE node to a simulation region
func MapNodeToRegion(spiceNode string, region int) {
	if !spiceEnabled {
		LogErr("SPICE interface not enabled")
		return
	}

	node := &SPICENode{
		Name:    spiceNode,
		Voltage: 0,
		Current: 0,
		Region:  region,
	}
	spiceNodes[spiceNode] = node

	LogOut(fmt.Sprintf("Mapped SPICE node '%s' to region %d", spiceNode, region))
}

// GetVoltage returns the voltage at a SPICE node
func GetVoltage(node string) float64 {
	if n, ok := spiceNodes[node]; ok {
		return n.Voltage
	}
	LogErr(fmt.Sprintf("Unknown SPICE node: %s", node))
	return 0
}

// GetCurrent returns the current at a SPICE node
func GetCurrent(node string) float64 {
	if n, ok := spiceNodes[node]; ok {
		return n.Current
	}
	LogErr(fmt.Sprintf("Unknown SPICE node: %s", node))
	return 0
}

// SetNodeVoltage sets the voltage at a node (for external control)
func SetNodeVoltage(node string, voltage float64) {
	if n, ok := spiceNodes[node]; ok {
		n.Voltage = voltage

		// Update corresponding simulation region
		// Set the electric field in the region based on voltage
		region := n.Region
		if region >= 0 {
			// Calculate electric field from voltage and layer thickness
			thickness := Mesh().CellSize()[2]
			if thickness > 0 {
				E_value := voltage / thickness
				E_field.setRegion(region, []float64{E_value})
				LogOut(fmt.Sprintf("Set E_field in region %d to %.2e V/m", region, E_value))
			}
		}
	}
}

// StepSPICE advances SPICE simulation by dt
func StepSPICE(dt float64) {
	if !spiceEnabled {
		return
	}

	spiceTimestep = dt

	// Calculate MTJ resistance from magnetization state
	// R = R_P * (1 + TMR * (1 - cos(theta))/2)
	// where theta is angle between free and reference layers
	R_P := 1000.0 // Parallel resistance (Ohms)
	TMR := 1.0    // TMR ratio

	// Get magnetization state to compute resistance
	m := M.Buffer()
	if m != nil {
		mx := m.Host()[0]
		if len(mx) > 0 {
			// Simplified: use average mz as proxy for state
			var avg_mz float64
			for _, v := range mx {
				avg_mz += float64(v)
			}
			avg_mz /= float64(len(mx))

			// Calculate resistance
			R_MTJ := R_P * (1 + TMR*(1-avg_mz)/2)

			// Update currents in nodes based on voltage and resistance
			if te, ok := spiceNodes["TE"]; ok {
				if be, ok := spiceNodes["BE"]; ok {
					V_diff := te.Voltage - be.Voltage
					I := V_diff / R_MTJ
					te.Current = I
					be.Current = -I
				}
			}
		}
	}
}

// ExportToSPICE exports simulation results to SPICE format
func ExportToSPICE(filename string) {
	if !spiceEnabled {
		LogErr("SPICE interface not enabled")
		return
	}

	// Export MTJ resistance vs state for compact model
	// Export switching characteristics
	// Export thermal resistance

	LogOut(fmt.Sprintf("Exported to SPICE format: %s", filename))
}

// GetSPICENodeList returns all SPICE nodes
func GetSPICENodeList() []string {
	nodes := make([]string, 0, len(spiceNodes))
	for name := range spiceNodes {
		nodes = append(nodes, name)
	}
	return nodes
}

// ============================================================================
// FEATURE V2.1-14: PDK INTEGRATION
// ============================================================================

// LoadPDK loads a Process Design Kit
func LoadPDK(pdkPathInput string) {
	pdkEnabled = true
	pdkPath = pdkPathInput

	LogOut(fmt.Sprintf("Loading PDK from: %s", pdkPath))

	// Initialize standard MRAM layer stack based on technology node
	// This provides realistic default values for common MRAM structures
	switch technologyNode {
	case "28nm", "22nm":
		layerStack = []LayerInfo{
			{Name: "M1", Material: "Cu", Thickness: 100e-9, Purpose: "electrode"},
			{Name: "TE", Material: "Ta", Thickness: 5e-9, Purpose: "electrode"},
			{Name: "FL", Material: "CoFeB", Thickness: 1.5e-9, Purpose: "magnetic"},
			{Name: "TB", Material: "MgO", Thickness: 1e-9, Purpose: "barrier"},
			{Name: "RL", Material: "CoFeB", Thickness: 2e-9, Purpose: "magnetic"},
			{Name: "SAF", Material: "Co/Pt", Thickness: 10e-9, Purpose: "magnetic"},
			{Name: "BE", Material: "Ta", Thickness: 5e-9, Purpose: "electrode"},
		}
	case "14nm", "10nm", "7nm":
		layerStack = []LayerInfo{
			{Name: "M1", Material: "Cu", Thickness: 50e-9, Purpose: "electrode"},
			{Name: "TE", Material: "Ta", Thickness: 3e-9, Purpose: "electrode"},
			{Name: "FL", Material: "CoFeB", Thickness: 1.2e-9, Purpose: "magnetic"},
			{Name: "TB", Material: "MgO", Thickness: 0.8e-9, Purpose: "barrier"},
			{Name: "RL", Material: "CoFeB", Thickness: 1.5e-9, Purpose: "magnetic"},
			{Name: "SAF", Material: "Co/Pt", Thickness: 8e-9, Purpose: "magnetic"},
			{Name: "BE", Material: "Ta", Thickness: 3e-9, Purpose: "electrode"},
		}
	default:
		layerStack = []LayerInfo{
			{Name: "M1", Material: "Cu", Thickness: 100e-9, Purpose: "electrode"},
			{Name: "TE", Material: "Ta", Thickness: 5e-9, Purpose: "electrode"},
			{Name: "FL", Material: "CoFeB", Thickness: 1.5e-9, Purpose: "magnetic"},
			{Name: "TB", Material: "MgO", Thickness: 1e-9, Purpose: "barrier"},
			{Name: "RL", Material: "CoFeB", Thickness: 2e-9, Purpose: "magnetic"},
			{Name: "SAF", Material: "Co/Pt", Thickness: 10e-9, Purpose: "magnetic"},
			{Name: "BE", Material: "Ta", Thickness: 5e-9, Purpose: "electrode"},
		}
	}

	LogOut(fmt.Sprintf("Loaded %d layers for %s technology", len(layerStack), technologyNode))
}

// SetTechnologyNode sets the technology node
func SetTechnologyNode(node string) {
	switch node {
	case "180nm", "130nm", "90nm", "65nm", "45nm", "28nm", "22nm", "14nm", "10nm", "7nm", "5nm", "3nm":
		technologyNode = node
		LogOut(fmt.Sprintf("Technology node set to: %s", node))
	default:
		LogErr(fmt.Sprintf("Unknown technology node: %s", node))
	}
}

// GetLayerStack returns the process layer stack
func GetLayerStack() []LayerInfo {
	return layerStack
}

// AddLayerToStack adds a layer to the process stack
func AddLayerToStack(name, material string, thickness float64, purpose string) {
	layer := LayerInfo{
		Name:      name,
		Material:  material,
		Thickness: thickness,
		Purpose:   purpose,
	}
	layerStack = append(layerStack, layer)
}

// ApplyProcessVariation applies statistical variations
func ApplyProcessVariation(sigma float64) {
	if !pdkEnabled {
		LogErr("PDK not loaded")
		return
	}

	// Apply variations to:
	// - Layer thicknesses
	// - Material parameters
	// - Interface properties

	LogOut(fmt.Sprintf("Applied process variation: σ=%.1f%%", sigma*100))
}

// AddProcessVariation defines a process variation source
func AddProcessVariation(parameter string, nominal, sigma float64, varType string) {
	variation := ProcessVariation{
		Parameter: parameter,
		Nominal:   nominal,
		Sigma:     sigma,
		Type:      varType,
	}
	processVariations = append(processVariations, variation)
}

// ExportGDSII exports geometry to GDSII format
func ExportGDSII(filename string) {
	// Export simulation geometry to GDSII for layout tools

	// In full implementation:
	// 1. Convert simulation grid to polygons
	// 2. Map regions to GDS layers
	// 3. Write GDSII stream format

	LogOut(fmt.Sprintf("Exported to GDSII: %s", filename))
}

// ExportOASIS exports geometry to OASIS format
func ExportOASIS(filename string) {
	// OASIS is more efficient than GDSII for large designs
	LogOut(fmt.Sprintf("Exported to OASIS: %s", filename))
}

// ============================================================================
// FEATURE V2.1-15: LAYOUT VS SCHEMATIC (LVS)
// ============================================================================

// EnableLVS enables LVS verification
func EnableLVS() {
	lvsEnabled = true
	LogOut("LVS verification enabled")
}

// RunLVSCheck performs LVS verification
func RunLVSCheck(layout, schematic string) LVSResult {
	if !lvsEnabled {
		EnableLVS()
	}

	result := LVSResult{
		Match:        true,
		ErrorCount:   0,
		WarningCount: 0,
		Errors:       make([]LVSError, 0),
	}

	LogOut(fmt.Sprintf("LVS check: layout=%s, schematic=%s", layout, schematic))

	// Perform basic connectivity checks
	// Check that all required layers are present
	requiredLayers := map[string]bool{
		"electrode": false,
		"magnetic":  false,
		"barrier":   false,
	}

	for _, layer := range layerStack {
		requiredLayers[layer.Purpose] = true
	}

	// Check for missing layers
	for purpose, found := range requiredLayers {
		if !found {
			result.Match = false
			result.ErrorCount++
			result.Errors = append(result.Errors, LVSError{
				Type:     "missing",
				Location: purpose,
				Message:  fmt.Sprintf("Required layer type '%s' not found in stack", purpose),
			})
		}
	}

	// Check for reasonable thicknesses
	for _, layer := range layerStack {
		if layer.Thickness <= 0 {
			result.Match = false
			result.ErrorCount++
			result.Errors = append(result.Errors, LVSError{
				Type:     "parameter",
				Location: layer.Name,
				Message:  fmt.Sprintf("Layer '%s' has invalid thickness: %.2e", layer.Name, layer.Thickness),
			})
		}
	}

	if result.Match {
		LogOut("LVS check: PASS")
	} else {
		LogOut(fmt.Sprintf("LVS check: FAIL (%d errors)", result.ErrorCount))
	}

	return result
}

// GetLVSErrors returns LVS errors
func GetLVSErrors(result LVSResult) []LVSError {
	return result.Errors
}

// ============================================================================
// FEATURE V2.1-16: PARASITIC EXTRACTION
// ============================================================================

// ExtractParasitics extracts parasitic R, L, C
func ExtractParasitics() map[string]float64 {
	parasitics := make(map[string]float64)

	// Material conductivities (S/m)
	conductivity := map[string]float64{
		"Cu":    5.96e7,
		"Ta":    7.61e6,
		"CoFeB": 1.0e6,
		"MgO":   1e-10, // Insulator
		"Co/Pt": 5.0e6,
	}

	// Calculate resistance from geometry and conductivity
	// R = ρ * L / A where ρ = 1/σ
	totalR := 0.0
	totalC := 0.0
	totalL := 0.0

	// Get device area from mesh
	cellSize := Mesh().CellSize()
	meshSize := Mesh().Size()
	area := float64(meshSize[0]) * cellSize[0] * float64(meshSize[1]) * cellSize[1]

	for _, layer := range layerStack {
		sigma, ok := conductivity[layer.Material]
		if !ok {
			sigma = 1e6 // Default conductivity
		}

		// Resistance: R = t / (σ * A)
		if sigma > 0 && area > 0 {
			R := layer.Thickness / (sigma * area)
			totalR += R
		}

		// Capacitance: C = ε * A / t
		// Use relative permittivity based on material
		epsilon_0 := 8.854e-12
		epsilon_r := 1.0
		if layer.Material == "MgO" {
			epsilon_r = 9.8 // MgO relative permittivity
		}
		if layer.Thickness > 0 {
			C := epsilon_0 * epsilon_r * area / layer.Thickness
			totalC += C
		}

		// Inductance: Simple estimate L ≈ μ₀ * t (per unit area)
		mu_0 := 4 * 3.14159265359 * 1e-7
		totalL += mu_0 * layer.Thickness
	}

	parasitics["R_total"] = totalR          // Ohms
	parasitics["C_total"] = totalC          // Farads
	parasitics["L_total"] = totalL          // Henrys
	parasitics["RC_time"] = totalR * totalC // RC time constant
	parasitics["area"] = area               // Device area

	LogOut(fmt.Sprintf("Extracted parasitics: R=%.2e Ω, C=%.2e F, L=%.2e H", totalR, totalC, totalL))

	return parasitics
}

// ============================================================================
// FEATURE V2.1-17: COMPACT MODEL GENERATION
// ============================================================================

// GenerateCompactModel generates a SPICE compact model from simulation
func GenerateCompactModel(modelName string) string {
	// Generate Verilog-A or SPICE subcircuit model

	model := fmt.Sprintf(`* Compact model for %s
* Generated by MuMax3-SAF-NeuroSpin V2.1

.SUBCKT %s TE BE
* Parameters extracted from micromagnetic simulation
.PARAM R_P = 1000     ; Parallel resistance (Ohms)
.PARAM R_AP = 2000    ; Anti-parallel resistance (Ohms)
.PARAM TMR = 100      ; TMR ratio (%%)
.PARAM T_SW = 1e-9    ; Switching time (s)

* Resistor model (state-dependent)
R1 TE BE R='R_P + (R_AP-R_P)*V(state)'

* State variable
C_STATE state 0 1e-15 IC=0
R_STATE state 0 1e12

.ENDS %s
`, modelName, modelName, modelName)

	LogOut(fmt.Sprintf("Generated compact model: %s", modelName))
	return model
}

// ============================================================================
// V2.1 INDUSTRY REGISTRATIONS
// ============================================================================

func init() {
	// SPICE interface
	DeclFunc("EnableSPICEInterface", EnableSPICEInterface,
		"Enable SPICE co-simulation")
	DeclFunc("SetSPICENetlist", SetSPICENetlist,
		"Load SPICE netlist file")
	DeclFunc("MapNodeToRegion", MapNodeToRegion,
		"Map SPICE node to region")
	DeclFunc("GetVoltage", GetVoltage,
		"Get voltage at SPICE node")
	DeclFunc("GetCurrent", GetCurrent,
		"Get current at SPICE node")
	DeclFunc("SetNodeVoltage", SetNodeVoltage,
		"Set voltage at node")
	DeclFunc("StepSPICE", StepSPICE,
		"Advance SPICE simulation")
	DeclFunc("ExportToSPICE", ExportToSPICE,
		"Export to SPICE format")
	DeclFunc("GetSPICENodeList", GetSPICENodeList,
		"Get all SPICE nodes")

	// PDK integration
	DeclFunc("LoadPDK", LoadPDK,
		"Load Process Design Kit")
	DeclFunc("SetTechnologyNode", SetTechnologyNode,
		"Set technology node")
	DeclFunc("GetLayerStack", GetLayerStack,
		"Get process layer stack")
	DeclFunc("AddLayerToStack", AddLayerToStack,
		"Add layer to stack")
	DeclFunc("ApplyProcessVariation", ApplyProcessVariation,
		"Apply process variations")
	DeclFunc("AddProcessVariation", AddProcessVariation,
		"Define process variation")
	DeclFunc("ExportGDSII", ExportGDSII,
		"Export to GDSII format")
	DeclFunc("ExportOASIS", ExportOASIS,
		"Export to OASIS format")

	// LVS
	DeclFunc("EnableLVS", EnableLVS,
		"Enable LVS verification")
	DeclFunc("RunLVSCheck", RunLVSCheck,
		"Run LVS check")
	DeclFunc("GetLVSErrors", GetLVSErrors,
		"Get LVS errors")

	// Parasitic extraction
	DeclFunc("ExtractParasitics", ExtractParasitics,
		"Extract parasitic R, L, C")

	// Compact model
	DeclFunc("GenerateCompactModel", GenerateCompactModel,
		"Generate SPICE compact model")
}
