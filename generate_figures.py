"""
Script to generate publication-grade figures for README and documentation.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0

from src.pes_scanner import scan_potential_energy_surface
from src.chemistry_engine import compute_heh_plus_integrals
from src.hamiltonian import build_electronic_problem
from src.resource_reduction import benchmark_all_mappings, map_hamiltonian
from src.ansatz import build_uccsd_ansatz
from src.vqe_engine import run_zero_noise_extrapolation, run_adapt_vqe_custom

print("Generating Figure 1: PES...")
pes = scan_potential_energy_surface('HeH+', np.linspace(0.4, 2.5, 50), run_quantum_vqe=False)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(pes['r_points'], pes['hf_energies'], 'k--', label='Hartree-Fock (HF)', alpha=0.7)
ax1.plot(pes['r_points'], pes['fci_energies'], color='#0284c7', lw=2.5, label='Exact Full CI (FCI)')
ax1.plot(pes['r_points'], pes['vqe_energies'], 'o-', color='#2563eb', ms=4, label='VQE (UCCSD, StatevectorEstimator)')
r_eq = pes['equilibrium_r_angstrom']
ax1.axvline(r_eq, color='#dc2626', ls=':', label=f'R_e = {r_eq:.3f} Å')
ax1.set_ylabel('Energy (Hartree)', fontsize=11, fontweight='bold')
ax1.set_title('HeH⁺ Potential Energy Surface (PES) Beyond Equilibrium', fontsize=13, fontweight='bold', pad=10)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(True, alpha=0.3)

errors_mha = [e * 1000 for e in pes['errors_ha']]
ax2.plot(pes['r_points'], errors_mha, 's-', color='#059669', ms=3, label='|E_VQE - E_exact|')
ax2.axhline(1.6, color='#dc2626', ls='--', label='Chemical Accuracy (1.6 mHa)')
ax2.set_xlabel('Internuclear Distance R (Å)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Error (mHa)', fontsize=11, fontweight='bold')
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig1_heh_plus_pes.png', dpi=200)
plt.close()

print("Generating Figure 2: Quantum Resource Reduction...")
mol = compute_heh_plus_integrals(0.914)
prob, fop = build_electronic_problem(mol)
bench = benchmark_all_mappings(prob, fop)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(3)
width = 0.25
labels = ['Jordan-Wigner', 'Parity (2Q Red.)', 'Bravyi-Kitaev']
qubits = [bench[k]['num_qubits'] for k in ['jordan_wigner', 'parity', 'bravyi_kitaev']]
paulis = [bench[k]['num_pauli_terms'] for k in ['jordan_wigner', 'parity', 'bravyi_kitaev']]
qwc = [bench[k]['num_qwc_groups'] for k in ['jordan_wigner', 'parity', 'bravyi_kitaev']]

ax.bar(x - width, qubits, width, label='Physical Qubits', color='#2563eb')
ax.bar(x, paulis, width, label='Pauli Terms', color='#94a3b8')
ax.bar(x + width, qwc, width, label='Commuting Groups (QWC)', color='#059669')

ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('Quantum Resource Reduction Across Mapping Techniques', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/fig2_resource_reduction.png', dpi=200)
plt.close()

print("Generating Figure 3: Zero-Noise Extrapolation...")
qop, mapper = map_hamiltonian(prob, fop, 'parity')
uccsd, _, _ = build_uccsd_ansatz(2, (1, 1), mapper)
assigned = uccsd.assign_parameters([0.0]*uccsd.num_parameters)
zne = run_zero_noise_extrapolation(qop, assigned, mol['E_fci'], 0.015, mol['E_nuc'])

fig, ax = plt.subplots(figsize=(8, 4.5))
c_fine = np.linspace(0.0, 5.2, 50)
p_quad = np.polyfit(zne['scale_factors'], zne['noisy_energies'], deg=2)
p_lin = np.polyfit(zne['scale_factors'][:2], zne['noisy_energies'][:2], deg=1)

ax.scatter(zne['scale_factors'], zne['noisy_energies'], color='#dc2626', s=70, zorder=5, label='Noisy Measurements (c ∈ {1, 3, 5})')
ax.plot(c_fine, np.polyval(p_lin, c_fine), '--', color='#2563eb', label='Linear Extrapolation (c ∈ {1, 3})')
ax.plot(c_fine, np.polyval(p_quad, c_fine), '-', color='#059669', lw=2, label='Richardson Extrapolation (c ∈ {1, 3, 5})')
ax.scatter([0.0], [zne['richardson_extrapolated_energy']], marker='*', s=150, color='#059669', zorder=6, label='ZNE Mitigated Limit')
ax.axhline(zne['exact_energy'], color='black', ls=':', label=f"Exact FCI ({zne['exact_energy']:.4f} Ha)")

ax.set_xlabel('Noise Amplification Factor (c)', fontsize=11, fontweight='bold')
ax.set_ylabel('Ground-State Energy (Hartree)', fontsize=11, fontweight='bold')
ax.set_title('Zero-Noise Extrapolation (ZNE) Quantum Error Mitigation', fontsize=13, fontweight='bold', pad=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig3_zero_noise_extrapolation.png', dpi=200)
plt.close()

print("All figures successfully created in figures/ folder!")
