"""
Direct measurement of the bare-mesh area-expansion modulus K_mesh under
uniform isotropic areal strain, with all non-bonded terms disabled. Fits
the stored energy U(eps_A) to 0.5 K_mesh A0 eps_A^2 and writes
mesh_modulus_calibration.csv.
"""

import numpy as np
import scipy.spatial as spatial
import math
import pandas as pd

# Coefficients of the linear constitutive mapping (manuscript Eq. 1); these
# must mirror GENE_COEFFS in kev_mechanics.py / kev_seeded_vs_spontaneous_hetro.py.
GENE_COEFFS = {'K_base': 240.0, 'alpha_SMT': 120.0, 'beta_FAD': 60.0}

# Genetic-state gene levels (must mirror MISSION_PROFILES /
# GENETIC_PROFILES in the simulation scripts).
GENETIC_PROFILES = {
    'STATE_A_RIPENING': {'SMT': 0.5, 'FAD': 3.0},
    'STATE_B_DEFENSE':  {'SMT': 4.0, 'FAD': 0.5},
}

# Operational micro-stiffness scaling used by the simulation scripts
# (k_bond = BOND_SCALE * K_local per bonded pair).
BOND_SCALE = 0.5


def composite_spring_map(genes):
    """Phase-composite stiffness map mirroring KEVContinuumBuilder.__init__.

    K_Lo  = K_base + alpha*G_SMT          (sterol-rich L_o patches)
    K_Ld  = max(10, K_base - beta*G_FAD)  (fluid L_d matrix)
    K_int = 2*K_Lo*K_Ld/(K_Lo+K_Ld)       (harmonic-mean interface bonds)
    """
    k_lo = GENE_COEFFS['K_base'] + GENE_COEFFS['alpha_SMT'] * genes['SMT']
    k_ld = max(10.0, GENE_COEFFS['K_base'] - GENE_COEFFS['beta_FAD'] * genes['FAD'])
    k_int = 2.0 * k_lo * k_ld / (k_lo + k_ld)
    return k_lo, k_ld, k_int


class MeshModulusCalibrator:
    """Bare Fibonacci-mesh calibrator for the area-expansion modulus K_mesh.

    Builds the same geometry and neighbour graph as the simulation scripts
    (Fibonacci sphere, density rho, bond cutoff r_cut) and applies uniform
    isotropic areal strain to the harmonic network only.
    """

    def __init__(self, radius=20.0, density=0.5, r_cut=2.5):
        self.radius = radius
        self.density = density
        self.r_cut = r_cut
        # Number of mesoscopic patches: N = 4*pi*R^2 * rho (as in the
        # simulation scripts).
        self.n_particles = int(4.0 * math.pi * (radius**2) * density)
        self.A0 = 4.0 * math.pi * (radius**2)

        # 1. Fibonacci-sphere coordinates (identical construction to
        #    KEVContinuumBuilder.build()).
        phi_golden = math.pi * (3.0 - math.sqrt(5.0))
        coords = []
        for i in range(self.n_particles):
            y = 1.0 - (i / float(self.n_particles - 1)) * 2.0
            rad_at_y = math.sqrt(max(0.0, 1.0 - y * y))
            theta = phi_golden * i
            pos = np.array([math.cos(theta) * rad_at_y, y, math.sin(theta) * rad_at_y]) * self.radius
            coords.append(pos)
        self.coords = np.array(coords)

        # 2. Neighbour graph (identical rule to the bond network: all pairs
        #    within r_cut). <z> is the mean coordination number.
        tree = spatial.cKDTree(self.coords)
        self.pairs = list(tree.query_pairs(r=self.r_cut))
        self.z_mean = 2.0 * len(self.pairs) / self.n_particles
        self.r0_bonds = np.array([np.linalg.norm(self.coords[i] - self.coords[j]) for i, j in self.pairs])

    def measure_kmesh(self, k_bond_val, max_areal_strain=0.04, n_points=20):
        """Measure K_mesh for a uniform bond stiffness k_bond_val.

        Applies incremental uniform isotropic areal strain
        eps_A = Delta A / A0 (radial scale s = sqrt(1+eps_A), so every bond
        stretches affinely to s*r0) and computes the stored harmonic energy

            U(eps_A) = 1/2 * k_bond * sum_b (s*r0_b - r0_b)^2.

        A linear fit of U against eps_A^2 then yields the continuum relation
        U = 1/2 * K_mesh * A0 * eps_A^2.
        """
        strains_A = np.linspace(0.0, max_areal_strain, n_points)
        energies = []

        for eps_A in strains_A:
            # Isotropic radial stretch: R = R0 * sqrt(1 + eps_A).
            scale = math.sqrt(1.0 + eps_A)
            r_stretched = self.r0_bonds * scale
            # Pure harmonic energy on the bare lattice.
            u_elastic = 0.5 * k_bond_val * np.sum((r_stretched - self.r0_bonds)**2)
            energies.append(u_elastic)

        # Fit U = 0.5 * K_mesh * A0 * eps_A^2  =>  U = P * eps_A^2 with
        # P = 0.5 * K_mesh * A0.
        poly = np.polyfit(strains_A**2, energies, 1)
        measured_kmesh = (2.0 * poly[0]) / self.A0
        prefactor = measured_kmesh / k_bond_val

        return measured_kmesh, prefactor


def run_calibration_suite():
    radii = [20.0, 40.0]

    # Legacy homogenised Eq. 1 targets (K_eff) for reference, plus the
    # phase-composite stiffnesses actually assigned by the revised scripts
    # (k_bond = 0.5 * K_local per pair).
    test_cases = [
        ('Ripening Matrix (legacy Eq.1)', 120.0, 0.5 * 120.0),
        ('POPC Baseline (legacy Eq.1)',   240.0, 0.5 * 240.0),
        ('Defence Raft (legacy Eq.1)',    690.0, 0.5 * 690.0),
    ]
    for state, genes in GENETIC_PROFILES.items():
        k_lo, k_ld, k_int = composite_spring_map(genes)
        label = 'Ripening (State A)' if state == 'STATE_A_RIPENING' else 'Defence (State B)'
        test_cases += [
            (f'{label} composite K_Lo',  k_lo,  BOND_SCALE * k_lo),
            (f'{label} composite K_Ld',  k_ld,  BOND_SCALE * k_ld),
            (f'{label} composite K_int', k_int, BOND_SCALE * k_int),
        ]

    records = []
    print("=" * 80)
    print("EMPIRICAL MESH AREA-EXPANSION MODULUS (K_mesh) CALIBRATION")
    print("=" * 80)

    for R in radii:
        calib = MeshModulusCalibrator(radius=R, density=0.5, r_cut=2.5)
        print(f"\n[Geometry] Radius R = {R} nm | N = {calib.n_particles} patches | "
              f"Coordination <z> = {calib.z_mean:.2f} | <r0> = {calib.r0_bonds.mean():.3f} nm")

        for label, k_target, k_bond_assigned in test_cases:
            k_mesh, prefactor = calib.measure_kmesh(k_bond_assigned)
            records.append({
                'Radius_nm': R,
                'Coordination_z': round(calib.z_mean, 2),
                'State_Description': label,
                'Target_K_mNm': k_target,
                'Assigned_k_bond_mNm': k_bond_assigned,
                'Empirical_K_mesh_mNm': round(k_mesh, 2),
                'Geometric_Prefactor_c': round(prefactor, 3),
                'Ratio_Kmesh_to_Target': round(k_mesh / k_target, 3)
            })

    df = pd.DataFrame(records)
    print("\n" + df.to_string(index=False))
    df.to_csv("mesh_modulus_calibration.csv", index=False)
    print("\nResults exported to 'mesh_modulus_calibration.csv'.")
    print("NOTE: K_mesh ~ 1.5 * k_bond ~ 0.77 * K_local (NOT 3.9x K_local);")
    print("      the prefactor is state-independent, so relative comparisons")
    print("      (ratios, % changes, effect sizes) are unaffected.")


if __name__ == "__main__":
    run_calibration_suite()
