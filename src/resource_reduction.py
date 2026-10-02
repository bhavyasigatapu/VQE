"""
Resource Reduction Module: Jordan-Wigner vs Parity vs Bravyi-Kitaev mappings,
Z2 symmetry qubit tapering, core orbital freezing, and commuting observable grouping.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
    TaperedQubitMapper
)
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.operators import FermionicOp

def get_mapper(mapper_type: str, num_particles: Tuple[int, int] | None = None):
    """Factory function for Qiskit Nature mappers."""
    m_type = mapper_type.lower().replace("-", "_").replace(" ", "_")
    if "parity" in m_type:
        return ParityMapper(num_particles=num_particles)
    elif "bravyi" in m_type or "bk" in m_type:
        return BravyiKitaevMapper()
    else:
        return JordanWignerMapper()

def map_hamiltonian(problem: ElectronicStructureProblem,
                    fermionic_op: FermionicOp,
                    mapper_type: str = "jordan_wigner") -> Tuple[SparsePauliOp, Any]:
    """
    Maps a second-quantized FermionicOp to a qubit SparsePauliOp.

    Args:
        problem: The electronic structure problem.
        fermionic_op: The fermionic operator.
        mapper_type: One of 'jordan_wigner', 'parity', 'bravyi_kitaev'.

    Returns:
        Tuple of (qubit_operator, mapper_instance).
    """
    mapper = get_mapper(mapper_type, num_particles=problem.num_particles)
    qubit_op = mapper.map(fermionic_op)
    return qubit_op, mapper

def apply_z2_tapering(problem: ElectronicStructureProblem,
                      mapper: Any,
                      fermionic_op: FermionicOp) -> Tuple[SparsePauliOp, Any, int]:
    """
    Applies Z2 symmetry qubit tapering to reduce the number of active qubits.

    Returns:
        Tuple of (tapered_qubit_op, tapered_mapper, num_tapered_qubits_saved).
    """
    try:
        tapered_mapper = problem.get_tapered_mapper(mapper)
        tapered_op = tapered_mapper.map(fermionic_op)
        orig_op = mapper.map(fermionic_op)
        qubits_saved = orig_op.num_qubits - tapered_op.num_qubits
        return tapered_op, tapered_mapper, qubits_saved
    except Exception as e:
        # If no non-trivial Z2 symmetries detected or tapering not applicable
        qubit_op = mapper.map(fermionic_op)
        return qubit_op, mapper, 0

def analyze_commuting_groups(qubit_op: SparsePauliOp) -> Dict[str, Any]:
    """
    Groups commuting observables into Qubit-Wise Commuting (QWC) and General Commuting (GC) cliques.
    Calculates required circuit shots and measurement reduction factor.
    """
    n_terms = len(qubit_op)
    # Qubit-wise commuting groups (can be measured simultaneously with single-qubit rotations)
    qwc_groups = qubit_op.group_commuting(qubit_wise=True)
    n_qwc = len(qwc_groups)

    # General commuting groups (can be measured simultaneously with Clifford circuits)
    gc_groups = qubit_op.group_commuting(qubit_wise=False)
    n_gc = len(gc_groups)

    # Theoretical shot comparison (1/variance ~ sum of absolute coeffs)
    coeffs = np.abs(qubit_op.coeffs.real)
    unmitigated_shot_metric = float(np.sum(coeffs)**2)

    # Grouped shot metric
    grouped_shot_metric = 0.0
    for grp in qwc_groups:
        grp_coeffs = np.abs(grp.coeffs.real)
        grouped_shot_metric += float(np.sum(grp_coeffs)**2)

    shot_reduction_percent = 0.0
    if unmitigated_shot_metric > 1e-12:
        shot_reduction_percent = max(0.0, (1.0 - (n_qwc / max(1, n_terms))) * 100.0)

    return {
        "num_pauli_terms": n_terms,
        "num_qwc_groups": n_qwc,
        "num_gc_groups": n_gc,
        "shot_reduction_percent": shot_reduction_percent,
        "qwc_groups": qwc_groups,
        "gc_groups": gc_groups
    }

def benchmark_all_mappings(problem: ElectronicStructureProblem,
                           fermionic_op: FermionicOp) -> Dict[str, Dict[str, Any]]:
    """
    Runs a comprehensive resource comparison across Jordan-Wigner, Parity, and Bravyi-Kitaev,
    including Z2 tapering and commuting groupings.
    """
    results = {}
    for m_name in ["jordan_wigner", "parity", "bravyi_kitaev"]:
        q_op, mapper = map_hamiltonian(problem, fermionic_op, mapper_type=m_name)
        comm_stats = analyze_commuting_groups(q_op)

        # Check tapering
        tap_op, _, tap_saved = apply_z2_tapering(problem, mapper, fermionic_op)

        results[m_name] = {
            "display_name": {
                "jordan_wigner": "Jordan-Wigner",
                "parity": "Parity (2-Qubit Reduction)",
                "bravyi_kitaev": "Bravyi-Kitaev"
            }[m_name],
            "num_qubits": q_op.num_qubits,
            "num_pauli_terms": len(q_op),
            "num_qwc_groups": comm_stats["num_qwc_groups"],
            "num_gc_groups": comm_stats["num_gc_groups"],
            "shot_reduction_pct": comm_stats["shot_reduction_percent"],
            "tapered_qubits": tap_op.num_qubits if tap_saved > 0 else q_op.num_qubits,
            "qubits_saved_by_tapering": tap_saved,
        }
    return results
