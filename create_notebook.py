"""
Generates the documented Jupyter Notebook: molecular_vqe_demo.ipynb
"""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Basque Quantum (BasQ) • Qiskit Fall Fest 2026\n",
            "## Molecular Ground-State Energy Estimation with VQE: Beyond Equilibrium\n",
            "**Author / Provider**: Benjamin Tirado (Basque Quantum / BasQ)\n",
            "**Implementation**: End-to-End Quantum Chemistry Simulation Pipeline with Qiskit Nature & StatevectorEstimator\n",
            "\n",
            "---\n",
            "### Notebook Overview\n",
            "1. **Track 1 (Beginner)**: Complete Potential Energy Surface (PES) of $\\text{HeH}^+$ across $R \\in [0.4, 2.5]\\text{ \\AA}$, equilibrium bond distance $R_e$, dissociation limit $E_{\\text{dissoc}}$, and UCCSD multireference analysis.\n",
            "2. **Track 2 (Intermediate)**: Quantum resource reduction: Jordan-Wigner vs. Parity (2-qubit reduction) vs. Bravyi-Kitaev, $Z_2$ symmetry tapering, core orbital freezing, and commuting observable grouping.\n",
            "3. **Track 3 (Advanced)**: Adapt-VQE dynamic ansatz growth and strongly correlated bond-breaking.\n",
            "4. **Track 4 (Advanced)**: Zero-Noise Extrapolation (ZNE) error mitigation on noisy quantum backends.\n",
            "5. **Track 5**: Hardware-aware transpilation and IBM native basis scaling."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Core Imports\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from src.chemistry_engine import compute_heh_plus_integrals, compute_lih_integrals\n",
            "from src.hamiltonian import build_electronic_problem\n",
            "from src.resource_reduction import map_hamiltonian, benchmark_all_mappings, analyze_commuting_groups\n",
            "from src.ansatz import build_uccsd_ansatz, transpile_and_analyze_circuit\n",
            "from src.vqe_engine import run_vqe, run_adapt_vqe_custom, run_zero_noise_extrapolation\n",
            "from src.pes_scanner import scan_potential_energy_surface\n",
            "\n",
            "print('All quantum chemistry modules loaded successfully!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Track 1: HeH⁺ Potential Energy Surface Across Internuclear Distances\n",
            "We sweep the internuclear separation $R \\in [0.4, 2.5]\\text{ \\AA}$ using analytical STO-3G Gaussian integrals and Boys functions $F_0(t)$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Scan PES across 22 points\n",
            "r_grid = np.linspace(0.4, 2.5, 22)\n",
            "pes = scan_potential_energy_surface('HeH+', r_grid, run_quantum_vqe=False)\n",
            "\n",
            "print(f\"Equilibrium Bond Distance R_e: {pes['equilibrium_r_angstrom']:.3f} Å\")\n",
            "print(f\"Equilibrium Energy E(R_e): {pes['equilibrium_energy_ha']:.5f} Ha\")\n",
            "print(f\"Dissociation Limit: {pes['dissociation_limit_ha']:.5f} Ha\")\n",
            "print(f\"Dissociation Energy D_e: {pes['dissociation_energy_ev']:.2f} eV ({pes['dissociation_energy_ha']:.4f} Ha)\")\n",
            "print(f\"Chemical Accuracy Met (|ΔE| <= 1.6 mHa): {pes['all_points_chemically_accurate']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Track 2: Quantum Resource Reduction Techniques\n",
            "Comparing **Jordan-Wigner**, **Parity (with 2-qubit reduction)**, and **Bravyi-Kitaev** mappings, plus $Z_2$ symmetry reduction and commuting observable grouping."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "mol_eq = compute_heh_plus_integrals(0.914)\n",
            "prob, fop = build_electronic_problem(mol_eq)\n",
            "bench = benchmark_all_mappings(prob, fop)\n",
            "\n",
            "for m_key, m_info in bench.items():\n",
            "    print(f\"=== {m_info['display_name']} ===\")\n",
            "    print(f\"  • Qubits: {m_info['num_qubits']} (Tapered: {m_info['tapered_qubits']})\")\n",
            "    print(f\"  • Pauli Terms: {m_info['num_pauli_terms']}\")\n",
            "    print(f\"  • Commuting Measurement Groups (QWC): {m_info['num_qwc_groups']}\")\n",
            "    print(f\"  • Measurement Shot Savings: {m_info['shot_reduction_pct']:.1f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Track 3: Adapt-VQE Dynamic Operator Growth\n",
            "In strongly correlated bond-breaking regimes, fixed single-reference UCCSD circuits face depth and parameter bottlenecks.\n",
            "**Adapt-VQE** dynamically appends operators with maximal commutator gradients $\\langle [H, A_k] \\rangle$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "qop_par, mapper_par = map_hamiltonian(prob, fop, 'parity')\n",
            "_, hf_state, pool = build_uccsd_ansatz(2, (1, 1), mapper_par)\n",
            "\n",
            "adapt_res = run_adapt_vqe_custom(\n",
            "    qubit_op=qop_par,\n",
            "    pool=pool,\n",
            "    initial_state=hf_state,\n",
            "    gradient_threshold=1e-3,\n",
            "    max_adapt_cycles=4,\n",
            "    nuclear_repulsion=mol_eq['E_nuc']\n",
            ")\n",
            "\n",
            "print(f\"Adapt-VQE Final Energy: {adapt_res['final_energy']:.6f} Ha\")\n",
            "print(f\"Exact Reference Energy: {mol_eq['E_fci']:.6f} Ha\")\n",
            "print(f\"Error: {abs(adapt_res['final_energy'] - mol_eq['E_fci']):.2e} Ha\")\n",
            "print(\"Cycle History:\")\n",
            "for step in adapt_res['history']:\n",
            "    print(f\"  Cycle {step['cycle']}: {step['operator']} | Energy = {step['energy']:.6f} Ha | Params = {step['num_params']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Track 4: Zero-Noise Extrapolation (ZNE) Error Mitigation\n",
            "Simulating expectation values under depolarizing noise with unitary folding ($c \\in \\{1, 3, 5\\}$) and Richardson quadratic extrapolation to $c \\to 0$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "uccsd_par, _, _ = build_uccsd_ansatz(2, (1, 1), mapper_par)\n",
            "assigned = uccsd_par.assign_parameters([0.0]*uccsd_par.num_parameters)\n",
            "\n",
            "zne = run_zero_noise_extrapolation(\n",
            "    qubit_op=qop_par,\n",
            "    circuit=assigned,\n",
            "    exact_energy=mol_eq['E_fci'],\n",
            "    depolarizing_rate=0.015,\n",
            "    nuclear_repulsion=mol_eq['E_nuc']\n",
            ")\n",
            "\n",
            "print(f\"Exact Reference: {zne['exact_energy']:.5f} Ha\")\n",
            "print(f\"Unmitigated Noisy (c=1): {zne['unmitigated_energy']:.5f} Ha (Error: {zne['unmitigated_error_ha']*1000:.2f} mHa)\")\n",
            "print(f\"Linear Extrapolated (c ∈ {{1, 3}}): {zne['linear_extrapolated_energy']:.5f} Ha (Error: {zne['linear_error_ha']*1000:.2f} mHa)\")\n",
            "print(f\"Richardson Extrapolated (c ∈ {{1, 3, 5}}): {zne['richardson_extrapolated_energy']:.5f} Ha (Error: {zne['richardson_error_ha']*1000:.2f} mHa)\")\n",
            "print(f\"Chemical Accuracy Met via ZNE: {zne['richardson_chem_acc']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Track 5: Hardware-Aware Transpilation\n",
            "Compiling the ansatz to native IBM Quantum basis gates (`['cz', 'sx', 'x', 'rz']`) to extract physical qubit footprint, CNOT depth, and 2-qubit entangler counts."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "stats = transpile_and_analyze_circuit(uccsd_par)\n",
            "print(f\"Active Physical Qubits: {stats['num_qubits']}\")\n",
            "print(f\"Variational Parameters: {stats['num_parameters']}\")\n",
            "print(f\"Transpiled 2-Qubit Gates (CZ/CX): {stats['two_qubit_gates']}\")\n",
            "print(f\"Transpiled Circuit Depth: {stats['transpiled_depth']}\")\n",
            "print(f\"Transpiled Gate Operations: {stats['transpiled_ops']}\")"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.3"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("molecular_vqe_demo.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Jupyter Notebook 'molecular_vqe_demo.ipynb' created successfully!")
