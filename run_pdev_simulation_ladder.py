"""
Loading-rate ladder engine for the convergence study of the critical
tension (manuscript Appendix C.5). One script reproduces every rung of the
geometric ladder:

    rate r  ->  Delta_f = 0.5/sqrt(r) kJ mol^-1 nm^-1, dwell = 500*sqrt(r) steps
    r = 1    production ramp  (0.5 @ 500 steps)
    r = 4                     (0.25 @ 1000)
    r = 16                    (0.125 @ 2000)
    r = 64                    (0.0625 @ 4000)

Campaign axes (missions, modes, pH levels, replicates, output name) are
selected on the command line; see --help. The physics is identical to
run_pdev_simulation_r20.py, from which this engine is derived.
"""

import numpy as np
import scipy.spatial as spatial
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.integrate as integrate
import pandas as pd
import multiprocessing as mp
from openmm import *
from openmm.app import *
from openmm.unit import *
import math
import time
import traceback

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

NUM_GPUS = 1
# Worker processes per GPU. 4 is safe for the largest shells (R = 40 nm,
# N ~ 10,050); 8-10 can be used for R = 20 nm (N ~ 2,513) runs if VRAM allows.
WORKERS_PER_GPU = 10
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

# Full factorial sweep: 2 modes x 2 states x 3 pH levels x REPLICATES.
# Trim MODES or PH_LEVELS to reproduce the legacy sub-campaigns (e.g. the
# topology control is MODES x states at pH 6.8 only; the acid-shock sweep is
# SEEDED x states x pH).
import argparse
_P = argparse.ArgumentParser(
    description="Unified loading-rate ladder engine (supersedes _slowrate/_16x/_64x/_64x_B scripts). "
                "Rate factor r slows the tension ramp geometrically vs production: "
                "Delta_f = 0.5/sqrt(r), dwell = 500*sqrt(r) steps (r=1 -> 0.5@500, 4 -> 0.25@1000, "
                "16 -> 0.125@2000, 64 -> 0.0625@4000).")
_P.add_argument("--rate", type=int, default=1, choices=[1, 4, 16, 64],
                help="loading-rate slow-down factor vs the production ramp (default 1 = production)")
_P.add_argument("--missions", default="STATE_A_RIPENING,STATE_B_DEFENSE",
                help="comma-separated missions to run (default both)")
_P.add_argument("--modes", default="SEEDED,SPONTANEOUS",
                help="comma-separated initialisation modes (default both)")
_P.add_argument("--ph", default="6.8",
                help="comma-separated pH levels (default 6.8; production campaign used 2.5,5.0,6.8)")
_P.add_argument("--replicates", type=int, default=50,
                help="replicates per (mission, mode, pH) cell (default 50)")
_P.add_argument("--out", default=None,
                help="output csv name; default pdev_unified_R<radius>_<rate>x.csv")
ARGS = _P.parse_args()

MODES = [m.strip() for m in ARGS.modes.split(',')]
MISSIONS = [m.strip() for m in ARGS.missions.split(',')]
PH_LEVELS = [float(p) for p in ARGS.ph.split(',')]
REPLICATES = ARGS.replicates

# FIX Point 8: configurable vesicle radius for the R = 20 / 40 nm size sweep.
VESICLE_RADIUS = 20.0   # nm  (set to 40.0 for the large-radius campaign)
PARTICLE_DENSITY = 0.5  # mesoscopic patches / nm^2

# Physics parameters (manuscript Appendix B): T = 300 K, Langevin friction
# gamma = 1.0 ps^-1, dt = 4 fs; Debye length lambda_D = 1.0 nm (~150 mM).
TEMP = 300 * kelvin
FRICTION = 1.0 / picosecond
TIMESTEP = 0.004 * picosecond
DEBYE_LENGTH = 1.0 * nanometer
# Standardised equilibration (matches the legacy scripts).
EQUILIBRATION_STEPS = 20000

# Loading protocol: per-particle force increment at the R = 20 nm reference
# and the yield criterion (mean radius > 1.15 r0). Delta_f is scaled by
# (20 nm / R) per replicate to equalise the tension-space loading rate
# gamma_dot ~ rho R f_dot across radii (see module docstring).
PRESSURE_INCREMENT = 0.5 / (ARGS.rate ** 0.5)  # rate-normalised ladder rung
LOAD_STEPS_PER_INCREMENT = int(500 * ARGS.rate ** 0.5)  # ramp rate = 1/ARGS.rate of production
MAX_LOAD_STEPS = 300       # 2x the legacy 150, covering the halved Delta_f at R = 40
RUPTURE_RADIAL_STRAIN = 1.15

# Optional RNG seed for reproducibility. None (default)
# leaves the global NumPy RNG unseeded, as in the original release; an integer
# seeds each replicate deterministically as SEED + task_id.
SEED = None

GENE_COEFFS = {
    'K_base': 240.0, 'alpha_SMT': 120.0, 'beta_FAD': 60.0
}
# G_PLD / gamma_PLD are deliberately absent : the lipid
# fractions below are direct phenotypic inputs and no equation is wired for
# G_PLD (see module docstring, "Composition inputs").

MISSION_PROFILES = {
    'STATE_A_RIPENING': {
        # State A (High-Fluidity / Homeostatic): wild-type maturation profile.
        # High G_FAD upregulates polyunsaturated fatty acids (S_unsat ~ 80%).
        # 15% rigid (dispersed rafts), 5% anionic.
        'genes': {'SMT': 0.5, 'FAD': 3.0},
        'fractions': {'S_sat': 0.1, 'S_unsat': 0.8, 'Sterol': 0.05, 'Anionic': 0.05}
    },
    'STATE_B_DEFENSE':  {
        # State B (High-Barrier / Stress-Adapted): G_SMT at the upper bound
        # (4.0-fold) drives sterol accumulation and L_o domain formation.
        # 50% rigid (armoured, discontinuous plates), 40% anionic.
        'genes': {'SMT': 4.0, 'FAD': 0.5},
        'fractions': {'S_sat': 0.25, 'S_unsat': 0.1, 'Sterol': 0.25, 'Anionic': 0.4}
    }
}


class UnitConverter:
    """Unit-conversion helper.

    OMM_TO_SI scales OpenMM internal units into the reported mN m^-1 tension
    units: 1 kJ mol^-1 nm^-1 = 1.660539 pN, and 1 kJ mol^-1 nm^-2 =
    1.660539 mN m^-1 (equivalently 1 mN m^-1 = 0.6022 kJ mol^-1 nm^-2).
    It is a fixed unit-conversion factor, not a tunable physical parameter.
    """
    OMM_TO_SI = 1.660539

# =============================================================================
# 2. PHYSICS ENGINE & COMPOSITE CONTINUUM BUILDER
# =============================================================================

class KEVContinuumBuilder:
    """Builds the SCG-MD vesicle system (geometry, topology, force field).

    The vesicle is a 2D elastic shell represented as a spherical mesh of
    mesoscopic lipid patches. Bonded nearest-neighbours form a quenched
    harmonic network whose stiffness is phase-dependent (the composite map of
    Equation 1); non-bonded LJ + screened-Coulomb terms provide weak
    residual cohesion and electrostatics respectively.
    """

    def __init__(self, fractions, genes, radius=20.0, density=0.5):
        self.fractions = fractions
        self.genes = genes
        self.radius = radius
        self.density = density
        self.system = System()
        self.positions = []
        self.particle_info = []
        # Number of mesoscopic patches: N = 4*pi*R^2 * rho (manuscript, "SCG-MD").
        self.n_lipids = int(4 * math.pi * (radius**2) * density)

        # Phase-composite constitutive mapping :
        #   K_Lo  = K_base + alpha*G_SMT          (sterol-rich L_o patches)
        #   K_Ld  = max(10, K_base - beta*G_FAD)  (fluid L_d matrix; floored
        #                                          to stay physically positive)
        #   K_int = 2*K_Lo*K_Ld/(K_Lo+K_Ld)       (harmonic-mean interface,
        #                                          springs in series)
        # All in mN m^-1; converted to OpenMM kJ mol^-1 nm^-2 below.
        k_base = GENE_COEFFS['K_base']
        k_lo_raw = k_base + (GENE_COEFFS['alpha_SMT'] * genes['SMT'])
        k_ld_raw = max(10.0, k_base - (GENE_COEFFS['beta_FAD'] * genes['FAD']))
        k_int_raw = (2.0 * k_lo_raw * k_ld_raw) / (k_lo_raw + k_ld_raw)

        # Raw values (mN m^-1) retained for reporting in the output CSV.
        self.k_lo_raw = k_lo_raw
        self.k_ld_raw = k_ld_raw
        self.k_int_raw = k_int_raw
        self.k_lo_omm = k_lo_raw / UnitConverter.OMM_TO_SI
        self.k_ld_omm = k_ld_raw / UnitConverter.OMM_TO_SI
        self.k_int_omm = k_int_raw / UnitConverter.OMM_TO_SI

    def _assign_types(self, coords, mode='SEEDED'):
        """Assign L_o / L_d identities for the requested initialisation mode.

        SEEDED      : cluster-growth / BFS pre-nucleation of L_o domains
                      (manuscript Appendix A.1; nucleation sites expand over
                      the 2.0 nm neighbour ball until the target rigid
                      fraction S_sat + Sterol is reached). An initialisation
                      protocol for a quenched configuration, not a claim that
                      active sorting is required.
        SPONTANEOUS : uniform random assignment of the same number of L_o
                      identities, the randomised control.

        Both modes assign exactly the same L_o fraction at identical
        composition, so any mechanical difference is attributable to
        topology alone.
        """
        n_total = len(coords)
        target_lo = self.fractions.get('S_sat', 0) + self.fractions.get('Sterol', 0)
        n_lo_target = int(n_total * target_lo)
        types = ['Ld'] * n_total

        if mode == 'SPONTANEOUS':
            indices = np.random.choice(n_total, n_lo_target, replace=False)
            for i in indices:
                types[i] = 'Lo'

        elif mode == 'SEEDED':
            # Seed density is constant (n_seeds = N/25), so the characteristic
            # domain size is radius-independent while the normalised largest
            # component S_max drifts down with N (finite-size effect;
            # see the size-independence appendix).
            n_seeds = max(3, int(n_total / 25))
            seeds = np.random.choice(n_total, n_seeds, replace=False)
            active_front = list(seeds)
            assigned_count = 0
            for s in seeds:
                types[s] = 'Lo'
                assigned_count += 1

            tree = spatial.cKDTree(coords)
            while assigned_count < n_lo_target and active_front:
                idx = np.random.randint(len(active_front))
                current = active_front[idx]
                neighbors = tree.query_ball_point(coords[current], 2.0)
                growth = False
                for n in neighbors:
                    if types[n] == 'Ld' and assigned_count < n_lo_target:
                        types[n] = 'Lo'
                        active_front.append(n)
                        assigned_count += 1
                        growth = True
                if not growth:
                    active_front.pop(idx)

            if assigned_count < n_lo_target:
                rem = [i for i, t in enumerate(types) if t == 'Ld']
                fill = np.random.choice(rem, n_lo_target - assigned_count, replace=False)
                for f in fill:
                    types[f] = 'Lo'

        return types

    def build(self, mode='SEEDED'):
        """Generate the Fibonacci-sphere mesh and assign particle identities."""
        # Fibonacci spiral on the unit sphere, scaled to the vesicle radius R.
        phi_golden = math.pi * (3.0 - math.sqrt(5.0))
        temp_coords = []
        for i in range(self.n_lipids):
            y = 1.0 - (i / float(self.n_lipids - 1)) * 2.0
            rad_at_y = math.sqrt(max(0.0, 1.0 - y * y))
            theta = phi_golden * i
            pos = np.array([math.cos(theta) * rad_at_y, y, math.sin(theta) * rad_at_y]) * self.radius
            temp_coords.append(pos)

        types = self._assign_types(temp_coords, mode=mode)

        # Anionic assignment from the profile fractions (fixes the legacy
        # hard-coded 0.4 draw over L_d, which had inverted the intended
        # charge phenotypes: State A ~34% / State B ~20% instead of 5% / 40%).
        # The anionic fraction lives inside the L_d matrix:
        # P(anionic | L_d) = Anionic / (1 - S_sat - Sterol).
        target_lo = self.fractions.get('S_sat', 0) + self.fractions.get('Sterol', 0)
        target_anionic = self.fractions.get('Anionic', 0.0)
        p_anionic_in_ld = min(1.0, max(0.0, target_anionic / max(1e-5, (1.0 - target_lo))))

        for i, pos in enumerate(temp_coords):
            # Each particle is a mesoscopic lipid-continuum patch (the mass is
            # a dummy inertial weight, not a physical lipid mass).
            h_idx = self.system.addParticle(100.0 * amu)
            self.positions.append(pos * nanometer)
            p_type = types[i]
            # Class label used only to set non-bonded parameters below.
            l_class = 'Sterol' if p_type == 'Lo' else 'S_unsat'
            if p_type == 'Ld' and np.random.rand() < p_anionic_in_ld:
                l_class = 'Anionic'
            self.particle_info.append({'idx': h_idx, 'class': l_class, 'type': p_type})

    def create_forces(self, ph_val=6.8):
        """Construct the OpenMM force field with energy-decomposition groups.

        Force groups :
            group 0 : U_bond (HarmonicBondForce, composite stiffnesses)
            group 1 : U_LJ   (Lennard-Jones, CutoffNonPeriodic 3.0 nm)
            group 2 : U_dh   (Debye-Huckel: CustomNonbondedForce + bonded-pair
                      CustomBondForce restoration, same group so the
                      decomposition captures all electrostatics)
            group 3 : radial inflation force (excluded from decomposition)

        ph_val : bulk pH used to set the static protonation state of anionic
                 headgroups via Henderson-Hasselbalch (pKa ~ 4.0). At gastric
                 pH 2.5 the anionic charge approaches zero (q ~ -0.03).
        """
        # --- Henderson-Hasselbalch static partial-charge approximation -------
        # q_anionic = -1 / (1 + 10^(pKa - pH)); pKa of phosphatidic acid ~ 4.0.
        pka_pa = 4.0
        q_anionic = -1.0 / (1.0 + 10**(pka_pa - ph_val))

        charges = [
            q_anionic if info['class'] == 'Anionic' else 0.0
            for info in self.particle_info
        ]

        # --- U_bond: phase-composite harmonic bond network ------------------
        # U_bond(r_ij) = 1/2 * k_bond * (r_ij - r0)^2 ; k_bond = 0.5 * K_local
        # with K_local in {K_Lo, K_Ld, K_int} per pair.
        # The 0.5 scaling is an operational micro-stiffness assignment, not
        # the Seung-Nelson triangular-lattice equivalence; measure_mesh_modulus.py
        # shows it yields K_mesh ~ 1.5 * k_bond ~ 0.77 * K_local here
        # , a state-independent geometric prefactor.
        bond = HarmonicBondForce()
        bond.setForceGroup(0)

        # --- U_LJ: Lennard-Jones cohesion (residual; see module docstring) ---
        # Lorentz-Berthelot combining rules; CutoffNonPeriodic (3.0 nm)
        # enables GPU neighbour lists.
        vdw = CustomNonbondedForce(
            "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
            "sigma=0.5*(sig1+sig2); epsilon=sqrt(eps1*eps2)"
       )
        vdw.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
        vdw.setCutoffDistance(3.0 * nanometer)
        vdw.addPerParticleParameter("sig")
        vdw.addPerParticleParameter("eps")
        vdw.setForceGroup(1)

        # --- U_dh: screened Coulomb (Debye-Huckel) electrostatics -----------
        # Electrostatic prefactor = 1/(4*pi*eps0*eps_r) in OpenMM units;
        # 138.935 kJ nm mol^-1 e^-2 is the vacuum value (eps_r = 1.0), the
        # reference maximal-coupling choice used for all published runs.
        EPS_R = 1.0  # relative permittivity; set ~78.4 for aqueous dielectric
        elec = CustomNonbondedForce(f"({138.935 / EPS_R:.6f})*q1*q2*exp(-r/debye)/r")
        elec.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
        elec.setCutoffDistance(3.0 * nanometer)
        elec.addPerParticleParameter("q")
        elec.addGlobalParameter("debye", DEBYE_LENGTH)
        elec.setForceGroup(2)

        # --- Bonded-pair Debye-Huckel restoration  --------
        # Bonded pairs are excluded from `elec` below (together with `vdw`,
        # removing the bonded/non-bonded double counting); charged bonded
        # pairs get their U_dh interaction back through this CustomBondForce.
        # Same force group as `elec` so the U_dh decomposition remains
        # complete. Pairs with q_prod = 0 are filtered out.
        bonded_elec = CustomBondForce(f"({138.935 / EPS_R:.6f})*q_prod*exp(-r/debye)/r")
        bonded_elec.addPerBondParameter("q_prod")
        bonded_elec.addGlobalParameter("debye", DEBYE_LENGTH)
        bonded_elec.setForceGroup(2)

        # --- F_rad: simulated osmotic inflation (turgor) --------------------
        # A radial external body force F_rad = k_pressure * r_hat is applied
        # to every particle and ramped in the worker loop to drive the
        # vesicle to its yield point. The exact tension conversion is applied
        # at analysis time (see module docstring, "Inflation protocol").
        stressor = CustomExternalForce("-k_pressure * sqrt(x*x+y*y+z*z)")
        stressor.addGlobalParameter("k_pressure", 0.0)
        for i in range(self.system.getNumParticles()):
            stressor.addParticle(i, [])
        stressor.setForceGroup(3)

        # Per-particle non-bonded parameters. Particles must be registered in
        # the CustomNonbondedForce objects *before* exclusions are added.
        # epsilon_Lo = 15.0 kJ mol^-1 (~6 kBT at 300 K) for sterol-rich L_o
        # domains; epsilon_mix = 1.0 kJ mol^-1 for fluid L_d interactions.
        for i, info in enumerate(self.particle_info):
            p_type = info['type']
            eps = 15.0 if p_type == 'Lo' else 1.0
            vdw.addParticle([1.0 * nanometer, eps])
            elec.addParticle([charges[i]])

        # --- Bond network construction (nearest neighbours within 2.5 nm) ---
        # On this mesh (rho = 0.5 nm^-2) the 2.5 nm ball captures <z> ~ 8
        # neighbours per patch (measure_mesh_modulus.py reports the exact
        # coordination for each radius).
        coords = [p.value_in_unit(nanometer) for p in self.positions]
        tree = spatial.cKDTree(coords)
        pairs = list(tree.query_pairs(r=2.5))

        for i, j in pairs:
            r0 = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            type_i = self.particle_info[i]['type']
            type_j = self.particle_info[j]['type']

            # Phase-composite spring selection.
            if type_i == 'Lo' and type_j == 'Lo':
                k_local = self.k_lo_omm
            elif type_i == 'Ld' and type_j == 'Ld':
                k_local = self.k_ld_omm
            else:
                k_local = self.k_int_omm

            k_bond_val = k_local * 0.5
            bond.addBond(i, j, r0 * nanometer, k_bond_val * kilojoule_per_mole / nanometer**2)

            # Exclude bonded pairs from both non-bonded forces.
            vdw.addExclusion(int(i), int(j))
            elec.addExclusion(int(i), int(j))

            # Restore bonded-pair electrostatics for charged pairs only.
            q_prod = charges[i] * charges[j]
            if abs(q_prod) > 1e-6:
                bonded_elec.addBond(int(i), int(j), [float(q_prod)])

        self.system.addForce(bond)
        self.system.addForce(vdw)
        self.system.addForce(elec)
        self.system.addForce(bonded_elec)
        self.system.addForce(stressor)
        return self.system

# =============================================================================
# 3. TOPOLOGY ANALYSIS
# =============================================================================

def calculate_topology(pos, info, cutoff=2.5):
    """Quantify rigid-phase morphology: percolation fraction S_max and Moran's I.

    S_max (manuscript, Appendix A.2): fraction of all membrane patches
    belonging to the largest connected rigid (L_o) component, computed via
    connected components on the L_o subgraph of the 2.5 nm neighbour graph.

    Moran's I (manuscript, Appendix A.2): spatial-autocorrelation index,
        I = (N/W) * sum_ij w_ij (x_i - xbar)(x_j - xbar) / sum_i (x_i - xbar)^2
    where w_ij = 1 for neighbour pairs, 0 otherwise. I ~ 0 indicates random
    mixing (Spontaneous); I -> 1 indicates high spatial clustering (Seeded).
    """
    # Binary rigid-phase indicator: 1.0 for L_o, 0.0 for L_d.
    vals = np.array([1.0 if p['type'] == 'Lo' else 0.0 for p in info])
    lo_indices = [i for i, x in enumerate(vals) if x == 1.0]
    if not lo_indices:
        return 0.0, 0.0

    # Neighbour graph on the *measured* (post-equilibration) positions.
    tree = spatial.cKDTree(pos)
    pairs = list(tree.query_pairs(cutoff))

    # Moran's I over all neighbour pairs.
    mean_x = np.mean(vals)
    denom = np.sum((vals - mean_x)**2)
    numer, sum_w = 0.0, 0.0
    if len(pairs) > 0:
        i_idx = np.array([p[0] for p in pairs])
        j_idx = np.array([p[1] for p in pairs])
        numer = np.sum((vals[i_idx] - mean_x) * (vals[j_idx] - mean_x))
        sum_w = len(pairs)
    moran_i = (len(vals) / sum_w * numer / denom) if (denom != 0 and sum_w != 0) else 0.0

    # Percolation (S_max) via connected components on the L_o subgraph.
    row_idx, col_idx = [], []
    for i, j in pairs:
        if vals[i] == 1.0 and vals[j] == 1.0:
            row_idx.append(i)
            col_idx.append(j)

    if not row_idx:
        return (1.0 / len(info)), moran_i

    data = np.ones(len(row_idx))
    adj = sp.coo_matrix((data, (row_idx, col_idx)), shape=(len(info), len(info)))
    n_comps, labels = csgraph.connected_components(adj, directed=False)
    lo_labels = labels[lo_indices]
    largest = np.bincount(lo_labels).max() if len(lo_labels) > 0 else 0
    return (largest / len(info)), moran_i

# =============================================================================
# 4. SIMULATION WORKER TASK
# =============================================================================

def create_context(system, gpu_id):
    """Create an OpenMM Context, preferring GPU platforms with CPU fallback.

    A fresh Integrator is constructed for each platform attempt: OpenMM
    binds the integrator during Context construction, and a failed GPU
    attempt can leave it bound, which would otherwise poison the CPU
    fallback ("This Integrator is already bound to a context").
    """
    for plat_name in ['CUDA', 'OpenCL', 'CPU']:
        integrator = LangevinIntegrator(TEMP, FRICTION, TIMESTEP)
        if SEED is not None:
            integrator.setRandomNumberSeed((SEED + 997) % (2**31))
        try:
            platform = Platform.getPlatformByName(plat_name)
            if plat_name == 'CPU':
                print(f"[Worker Warning] GPU platforms failed for device {gpu_id}. Falling back to CPU.", flush=True)
                return Context(system, integrator, platform)
            props = {'DeviceIndex': str(gpu_id), 'Precision': 'mixed'}
            return Context(system, integrator, platform, props)
        except Exception:
            del integrator
            continue
    raise RuntimeError("No usable OpenMM platform (CUDA/OpenCL/CPU) available")

def run_simulation_task(task_dict):
    """Run a single (mode, state, pH, radius) replicate: build, equilibrate,
    load to yield.

    Protocol: energy minimisation -> EQUILIBRATION_STEPS equilibration ->
    ramped radial loading with Delta_f scaled by (20 nm / R) so the
    tension-space loading rate gamma_dot ~ rho R f_dot is identical across
    radii -> yield when the mean radius exceeds 1.15 r0 (~32% areal strain).
    Returns a one-element list with the result dict (or an error dict).
    """
    task_id = task_dict['task_id']
    # Optional deterministic seeding (SEED = None leaves the RNG unseeded,
    # matching the original release; see SEED in the configuration block).
    if SEED is not None:
        np.random.seed((SEED + task_id) % (2**32))
    try:
        gpu_id = task_dict['gpu_id']
        mission_name = task_dict['mission_name']
        mode = task_dict.get('mode', 'SEEDED')
        ph_val = task_dict.get('ph_val', 6.8)
        radius = task_dict.get('radius', VESICLE_RADIUS)

        profile = MISSION_PROFILES[mission_name]
        genes = profile['genes']
        fractions = profile['fractions']
        # Legacy Eq. 1 homogenised modulus K_eff = K_base + alpha*G_SMT
        # - beta*G_FAD, floored to stay physical. Reported for backwards
        # compatibility; the bonds actually use the composite map.
        ka_raw = GENE_COEFFS['K_base'] + (GENE_COEFFS['alpha_SMT'] * genes['SMT']) - (GENE_COEFFS['beta_FAD'] * genes['FAD'])
        ka_eff = max(10.0, ka_raw)

        builder = KEVContinuumBuilder(fractions, genes, radius=radius, density=PARTICLE_DENSITY)
        builder.build(mode=mode)
        system = builder.create_forces(ph_val=ph_val)

        context = create_context(system, gpu_id)
        context.setPositions(builder.positions)
        LocalEnergyMinimizer.minimize(context)

        # Equilibration (standardised across conditions).
        integrator = context.getIntegrator()
        integrator.step(EQUILIBRATION_STEPS)

        pos_eq = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
        perc, het = calculate_topology(pos_eq, builder.particle_info)

        # Yield reference radius r0 and initial area from the *equilibrated*
        # mesh (the shell contracts slightly during minimisation/equilibration;
        # see the appendix).
        r0 = np.mean(np.linalg.norm(pos_eq, axis=1))
        initial_area = 4.0 * np.pi * (r0**2)

        # Rate-equalised force ramp: Delta_f(R) = PRESSURE_INCREMENT * 20/R
        # keeps R * Delta_f (and hence gamma_dot ~ rho R f_dot) constant
        # across radii.
        delta_f = PRESSURE_INCREMENT * (20.0 / radius)

        failed = False
        rupture_tension = 0.0
        pressure_k = 0.0
        strain_history = []
        tension_history = []

        for _ in range(MAX_LOAD_STEPS):
            pressure_k += delta_f
            context.setParameter("k_pressure", pressure_k)
            integrator.step(LOAD_STEPS_PER_INCREMENT)

            state = context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            current_r = np.mean(np.linalg.norm(pos, axis=1))
            current_area = 4.0 * np.pi * (current_r**2)

            # Areal strain relative to the equilibrated reference area.
            strain = (current_area - initial_area) / initial_area
            # Exact instantaneous equatorial membrane tension
            # : gamma = N f / (8 pi r) = rho(r) f r / 2,
            # converted to mN m^-1. Equals the legacy convention
            # f*r/2 * OMM_TO_SI multiplied by rho(r) in nm^-2.
            tension = (builder.n_lipids * pressure_k / (8.0 * math.pi * current_r)) * UnitConverter.OMM_TO_SI

            strain_history.append(strain)
            tension_history.append(tension)

            # Geometric yield criterion (imposed, not emergent): 15% radial
            # (~32% areal) strain.
            if current_r > r0 * RUPTURE_RADIAL_STRAIN:
                failed = True
                rupture_tension = tension
                break

        if not failed:
            rupture_tension = tension

        rupture_strain = strain_history[-1] if strain_history else 0.0
        f_crit = pressure_k

        # Toughness = area under the tension-strain curve.
        # Integrate on the monotonised strain axis so thermal back-steps do
        # not subtract area; the raw (un-monotonised) curve is still written
        # to the CSV.
        toughness = float(integrate.trapezoid(tension_history, np.maximum.accumulate(strain_history)))

        # Potential-energy decomposition at yield via force groups
        #. Group 2 = non-bonded + bonded Debye-Huckel.
        u_bond = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        u_lj   = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        u_dh   = context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(kilojoule_per_mole)

        # Audit of the realised anionic composition (should match the profile
        # fraction of all patches: 5% State A, 40% State B). NB: a list
        # comprehension, not a generator - `from openmm import *` shadows the
        # builtin sum() with openmm.unit.unit_math.sum.
        n_anionic = len([p for p in builder.particle_info if p['class'] == 'Anionic'])

        # Explicit cleanup frees GPU VRAM for the next queued replicate.
        del context

        return [{
            'Replicate': task_id, 'Mission': mission_name, 'Mode': mode,
            'pH': ph_val, 'Radius_nm': radius, 'N_particles': builder.n_lipids,
            'Delta_f': delta_f,
            'Ka_Input_mNm': ka_eff,
            'K_Lo_mNm': builder.k_lo_raw, 'K_Ld_mNm': builder.k_ld_raw,
            'K_Int_mNm': builder.k_int_raw,
            'Anionic_Fraction': n_anionic / builder.n_lipids,
            'Percolation': perc, 'Heterogeneity': het,
            'Ruptured': failed, 'Rupture_Strain': rupture_strain,
            'Rupture_Tension_mNm': rupture_tension, 'f_crit_kJmolnm': f_crit,
            'Toughness': toughness,
            'Strain_Curve': strain_history, 'Tension_Curve': tension_history,
            'U_bond_kJmol': u_bond, 'U_bond_per_N': u_bond / builder.n_lipids,
            'U_LJ_kJmol': u_lj, 'U_dh_kJmol': u_dh
        }]
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[Worker Failure] Task ID {task_id}:\n{tb}", flush=True)
        return [{'error': str(e), 'traceback': tb, 'Replicate': task_id}]

# =============================================================================
# 5. MAIN DISPATCHER
# =============================================================================

if __name__ == "__main__":
    n_expected = len(MODES) * len(MISSIONS) * len(PH_LEVELS) * REPLICATES
    print(f"--- Unified PDEV Production Run (Radius = {VESICLE_RADIUS} nm, "
          f"N = {int(4 * math.pi * VESICLE_RADIUS**2 * PARTICLE_DENSITY)} patches, "
          f"Delta_f = {PRESSURE_INCREMENT * 20.0 / VESICLE_RADIUS:.3f} kJ/mol/nm, "
          f"{n_expected} replicates) ---")
    tasks = []
    task_id = 0
    for mode in MODES:
        for m in MISSIONS:
            for ph in PH_LEVELS:
                for i in range(REPLICATES):
                    tasks.append({
                        'task_id': task_id,
                        'gpu_id': task_id % NUM_GPUS,
                        'mission_name': m,
                        'mode': mode,
                        'ph_val': ph,
                        'radius': VESICLE_RADIUS
                    })
                    task_id += 1

    ctx = mp.get_context('spawn')
    all_results = []
    start = time.time()
    with ctx.Pool(processes=TOTAL_WORKERS) as pool:
        for res_batch in pool.imap_unordered(run_simulation_task, tasks):
            if res_batch:
                all_results.extend(res_batch)
            if len(all_results) % 25 == 0:
                print(f" > {len(all_results)}/{len(tasks)} completed...", flush=True)

    if all_results:
        df = pd.DataFrame(all_results)

        if 'error' in df.columns and 'Mission' not in df.columns:
            print("\n" + "!" * 70)
            print("[FATAL] All tasks failed. Traceback of first failed task:")
            print(df['traceback'].iloc[0])
            print("!" * 70)
        else:
            if 'error' in df.columns:
                n_failed = df['error'].notna().sum()
                if n_failed > 0:
                    print(f"\nWarning: {n_failed} replicates failed and were omitted.")
                df = df[df['error'].isna()].copy()

            # Output filename carries the radius so the R = 20 and R = 40
            # campaigns can be run back-to-back without overwriting each other.
            out_name = ARGS.out or f"pdev_unified_R{int(VESICLE_RADIUS)}_{ARGS.rate}x.csv"
            df.to_csv(out_name, index=False)
            print(f"\nDONE in {time.time()-start:.1f}s. Wrote {out_name} ({len(df)} replicates).")
            print("\n--- PRODUCTION SUMMARY (mean +/- sem) ---")
            print(df.groupby(['Mode', 'Mission', 'pH'])[
                ['Rupture_Tension_mNm', 'Toughness', 'U_bond_per_N', 'Percolation', 'U_dh_kJmol']
            ].agg(['mean', 'sem']))
