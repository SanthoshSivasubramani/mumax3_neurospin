// cuda/version_saf_neurospin.go
package cuda

// SAF-NeuroSpin extension metadata for MuMax3
// ------------------------------------------------------------
// This variable embeds authorship and version information
// directly into the compiled binary without changing CLI output.
// It is discoverable via `strings` but not printed on startup.
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
// ------------------------------------------------------------

var SAFNeuroSpinVersion = "MuMax3-SAF-NeuroSpin v2.1 (2025-2026)\n" +
	"Copyright © 2025-2026 Prof. Santhosh Sivasubramani\n" +
	"INTRINSIC Lab, IIT Delhi | April AI Hub, University of Edinburgh\n" +
	"Email: ssivasub@iitd.ac.in, ssivasub@ed.ac.uk, ragansanthosh@ieee.org\n" +
	"Repository: https://github.com/SanthoshSivasubramani/Mumax3_SAF_Neurospin"

// init() keeps a live reference so the linker retains this variable.
func init() {
	_ = SAFNeuroSpinVersion
}

// SAFNeuroSpinTag returns the version string
func SAFNeuroSpinTag() string {
	return SAFNeuroSpinVersion
}

// PrintSAFAttribution prints the full attribution banner
func PrintSAFAttribution() {
	println("╔═══════════════════════════════════════════════════════════════════╗")
	println("║  MuMax3-SAF-NeuroSpin v2.1 - Neuromorphic Computing Extension    ║")
	println("║  Copyright © 2025-2026 Prof. Santhosh Sivasubramani              ║")
	println("║  INTRINSIC Lab, IIT Delhi | April AI Hub, UoE                    ║")
	println("║  Email: ssivasub@iitd.ac.in, ssivasub@ed.ac.uk                   ║")
	println("║  GitHub: github.com/SanthoshSivasubramani/Mumax3_SAF_Neurospin  ║")
	println("╚═══════════════════════════════════════════════════════════════════╝")
}
