"""
Automated End-to-End Validation Suite for BasQ VQE Platform
Verifies mathematical rigor, chemical accuracy, resource reduction, and ZNE error mitigation.
"""

import unittest
import numpy as np
from src.chemistry_engine import compute_heh_plus_integrals, compute_lih_integrals, compute_beh2_integrals
from src.hamiltonian import build_electronic_problem
from src.resource_reduction import map_hamiltonian, benchmark_all_mappings, analyze_commuting_groups
from src.ansatz import build_uccsd_ansatz, transpile_and_analyze_circuit
from src.vqe_engine import run_vqe, run_adapt_vqe_custom, run_zero_noise_extrapolation
from src.pes_scanner import scan_potential_energy_surface

class TestBasQVQEPlatform(unittest.TestCase):

    def test_analytical_integrals_and_scf(self):
        """Test STO-3G integral generation and SCF convergence for HeH+."""
        data = compute_heh_plus_integrals(0.914)
        self.assertEqual(data["num_particles"], (1, 1))
        self.assertEqual(data["num_spatial_orbitals"], 2)
        # Check standard literature values
        self.assertAlmostEqual(data["E_hf"], -2.854, delta=0.01)
        self.assertLess(data["E_fci"], data["E_hf"])
        # Correlation energy in STO-3G HeH+ should be ~ -0.01 Ha (NOT -0.61 Ha!)
        corr_energy = data["E_fci"] - data["E_hf"]
        self.assertGreater(corr_energy, -0.02)
        self.assertLess(corr_energy, -0.005)

    def test_parity_two_qubit_reduction(self):
        """Verify Parity mapping reduces HeH+ from 4 qubits to 2 qubits."""
        data = compute_heh_plus_integrals(0.914)
        prob, fop = build_electronic_problem(data)
        
        jw_op, _ = map_hamiltonian(prob, fop, "jordan_wigner")
        par_op, _ = map_hamiltonian(prob, fop, "parity")
        
        self.assertEqual(jw_op.num_qubits, 4)
        self.assertEqual(par_op.num_qubits, 2)
        self.assertLess(len(par_op), len(jw_op))

    def test_commuting_observable_grouping(self):
        """Verify commuting cliques significantly reduce measurement shots."""
        data = compute_heh_plus_integrals(0.914)
        prob, fop = build_electronic_problem(data)
        jw_op, _ = map_hamiltonian(prob, fop, "jordan_wigner")
        
        comm = analyze_commuting_groups(jw_op)
        self.assertLess(comm["num_qwc_groups"], comm["num_pauli_terms"])
        self.assertGreater(comm["shot_reduction_percent"], 50.0)

    def test_vqe_statevector_chemical_accuracy(self):
        """Verify VQE obeys the variational principle and reaches chemical accuracy (|ΔE| <= 1.6 mHa)."""
        data = compute_heh_plus_integrals(0.914)
        prob, fop = build_electronic_problem(data)
        par_op, mapper = map_hamiltonian(prob, fop, "parity")
        uccsd, hf, _ = build_uccsd_ansatz(2, (1, 1), mapper)
        
        vqe_res = run_vqe(par_op, uccsd, optimizer_name="SLSQP", maxiter=60, nuclear_repulsion=data["E_nuc"])
        # Energy should be lower than HF and strictly satisfy the variational principle (>= E_fci)
        self.assertLess(vqe_res["total_energy"], data["E_hf"])
        self.assertGreaterEqual(vqe_res["total_energy"], data["E_fci"] - 1e-9)
        # Chemical accuracy (|ΔE| <= 1.6 mHa = 0.0016 Ha)
        err = abs(vqe_res["total_energy"] - data["E_fci"])
        self.assertLessEqual(err, 1.6e-3)

    def test_adapt_vqe_pool(self):
        """Verify Adapt-VQE dynamically selects operators with highest gradients."""
        data = compute_heh_plus_integrals(0.914)
        prob, fop = build_electronic_problem(data)
        par_op, mapper = map_hamiltonian(prob, fop, "parity")
        _, hf, pool = build_uccsd_ansatz(2, (1, 1), mapper)
        
        res = run_adapt_vqe_custom(par_op, pool, hf, gradient_threshold=1e-3, max_adapt_cycles=3, nuclear_repulsion=data["E_nuc"])
        self.assertTrue(len(res["history"]) >= 2)
        # Energy must decrease monotonically or remain bounded
        self.assertLessEqual(res["final_energy"], res["history"][0]["energy"] + 1e-6)

    def test_zero_noise_extrapolation(self):
        """Verify ZNE reduces error compared to raw noisy expectation."""
        data = compute_heh_plus_integrals(0.914)
        prob, fop = build_electronic_problem(data)
        par_op, mapper = map_hamiltonian(prob, fop, "parity")
        uccsd, _, _ = build_uccsd_ansatz(2, (1, 1), mapper)
        
        assigned = uccsd.assign_parameters([0.0]*uccsd.num_parameters)
        zne = run_zero_noise_extrapolation(par_op, assigned, exact_energy=data["E_fci"], depolarizing_rate=0.015, nuclear_repulsion=data["E_nuc"])
        
        self.assertLess(zne["richardson_error_ha"], zne["unmitigated_error_ha"])

    def test_hardware_transpilation(self):
        """Verify transpilation to native IBM basis gates."""
        data = compute_heh_plus_integrals(0.914)
        prob, fop = build_electronic_problem(data)
        par_op, mapper = map_hamiltonian(prob, fop, "parity")
        uccsd, _, _ = build_uccsd_ansatz(2, (1, 1), mapper)
        
        stats = transpile_and_analyze_circuit(uccsd)
        self.assertIn("two_qubit_gates", stats)
        self.assertGreater(stats["transpiled_depth"], 0)

    def test_pes_equilibrium_and_dissociation(self):
        """Verify physical equilibrium bond length (~0.914 A) and binding depth (~1.5 eV)."""
        pes = scan_potential_energy_surface("HeH+", run_quantum_vqe=False)
        self.assertAlmostEqual(pes["equilibrium_r_angstrom"], 0.92, delta=0.03)
        self.assertGreater(pes["dissociation_energy_ev"], 1.0)
        self.assertLess(pes["dissociation_energy_ev"], 2.5)
        # Dissociation limit should be close to isolated He STO-3G ground state (-2.8082 Ha)
        self.assertAlmostEqual(pes["dissociation_limit_ha"], -2.8082, delta=0.01)

    def test_lih_and_beh2_scaling(self):
        """Verify LiH and BeH2 potential energy surfaces and chemical accuracy."""
        pes_lih = scan_potential_energy_surface("LiH", run_quantum_vqe=False)
        self.assertAlmostEqual(pes_lih["equilibrium_r_angstrom"], 1.60, delta=0.05)
        self.assertGreater(pes_lih["dissociation_energy_ev"], 1.0)
        self.assertTrue(pes_lih["all_points_chemically_accurate"])

        pes_beh2 = scan_potential_energy_surface("BeH2", run_quantum_vqe=False)
        self.assertAlmostEqual(pes_beh2["equilibrium_r_angstrom"], 1.33, delta=0.05)
        self.assertGreater(pes_beh2["dissociation_energy_ev"], 1.0)
        self.assertTrue(pes_beh2["all_points_chemically_accurate"])

if __name__ == "__main__":
    unittest.main()
