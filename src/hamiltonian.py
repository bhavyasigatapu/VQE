"""
Hamiltonian Module: Converts electronic integrals into Qiskit Nature ElectronicStructureProblem
and generates the second-quantized FermionicOp.
"""

from typing import Dict, Any, Tuple
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.operators import FermionicOp

def build_electronic_problem(mol_data: Dict[str, Any]) -> Tuple[ElectronicStructureProblem, FermionicOp]:
    """
    Constructs the Qiskit Nature ElectronicStructureProblem and FermionicOp from molecular integrals.

    Args:
        mol_data: Dictionary containing h1_mo, h2_mo, E_nuc, num_particles, num_spatial_orbitals.

    Returns:
        Tuple of (ElectronicStructureProblem, FermionicOp).
    """
    h1 = mol_data["h1_mo"]
    h2 = mol_data["h2_mo"]
    e_nuc = mol_data["E_nuc"]
    num_particles = mol_data["num_particles"]
    num_spatial = mol_data["num_spatial_orbitals"]

    # Construct ElectronicEnergy from raw MO integrals
    ee = ElectronicEnergy.from_raw_integrals(
        h1_a=h1,
        h2_aa=h2,
        auto_index_order=True
    )
    ee.nuclear_repulsion_energy = e_nuc

    problem = ElectronicStructureProblem(ee)
    problem.num_particles = num_particles
    problem.num_spatial_orbitals = num_spatial

    fermionic_op = problem.hamiltonian.second_q_op()
    return problem, fermionic_op
