"""
Ansatz Module: Chemically-inspired UCCSD, Hardware-Efficient (RealAmplitudes),
and Adapt-VQE excitation operator pools with hardware-aware transpilation analysis.
"""

from typing import Tuple, List, Dict, Any
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import real_amplitudes
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit.quantum_info import SparsePauliOp

def build_hf_state(num_spatial_orbitals: int,
                   num_particles: Tuple[int, int],
                   mapper: Any) -> QuantumCircuit:
    """Constructs the Hartree-Fock reference state circuit."""
    return HartreeFock(num_spatial_orbitals, num_particles, mapper)

def build_uccsd_ansatz(num_spatial_orbitals: int,
                       num_particles: Tuple[int, int],
                       mapper: Any) -> Tuple[QuantumCircuit, QuantumCircuit, List[SparsePauliOp]]:
    """
    Constructs the Unitary Coupled Cluster with Singles and Doubles (UCCSD) ansatz.

    Returns:
        Tuple of (full_uccsd_circuit, hf_reference_circuit, operator_pool).
    """
    hf = build_hf_state(num_spatial_orbitals, num_particles, mapper)
    uccsd = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=hf
    )
    # Extract pool of excitation operators
    pool = list(uccsd.operators) if hasattr(uccsd, "operators") and uccsd.operators is not None else []
    return uccsd, hf, pool

def build_hardware_efficient_ansatz(num_qubits: int,
                                    reps: int = 2,
                                    entanglement: str = "linear") -> QuantumCircuit:
    """
    Constructs a hardware-efficient parameterized trial wavefunction (RealAmplitudes)
    for NISQ devices.
    """
    return real_amplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)

def transpile_and_analyze_circuit(circuit: QuantumCircuit,
                                  basis_gates: List[str] = None,
                                  optimization_level: int = 3) -> Dict[str, Any]:
    """
    Performs hardware-aware transpilation and extracts key quantitative metrics:
    active physical qubits, parameter count, 2-qubit CNOT depth, and total gate count.
    """
    if basis_gates is None:
        # Standard IBM Quantum basis
        basis_gates = ['cz', 'sx', 'x', 'rz']

    # Transpile to hardware native gates
    transpiled = transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=optimization_level
    )

    ops_count = transpiled.count_ops()
    cnot_cz_count = ops_count.get("cz", 0) + ops_count.get("cx", 0)

    return {
        "num_qubits": circuit.num_qubits,
        "num_parameters": circuit.num_parameters,
        "circuit_depth": circuit.depth(),
        "transpiled_depth": transpiled.depth(),
        "two_qubit_gates": cnot_cz_count,
        "total_transpiled_gates": sum(ops_count.values()),
        "transpiled_ops": dict(ops_count),
        "transpiled_circuit": transpiled
    }
