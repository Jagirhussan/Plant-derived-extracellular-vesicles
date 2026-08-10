"""
Genomic regulation of chemo-mechanical stability in plant-derived
extracellular vesicles (PDEVs): a multiscale model of composite reinforcement.

kev_mechanics.py
================

Simulated osmotic inflation of coarse-grained PDEV vesicles across genetic
states and pH levels to quantify the mechanical hierarchy and acid-shock
response reported in the manuscript (Sections "Mechanical hierarchy of states"
and "Acid shock and mechanical homeostasis"; Figs. 2, 5 and the pH stability
data behind Tables 1-4).

Framework
---------
A 2D Supra-molecular Coarse-Grained Molecular Dynamics (SCG-MD) mesh built on
OpenMM. Each simulated *particle* is a mesoscopic patch of the lipid continuum
(not a single atom or lipid molecule). The vesicle is a spherical shell
(R = 20 nm, surface density rho = 0.5 nm^-2, N ~ 2500 patches) whose total
potential energy is

    U_total = sum U_bond + sum U_LJ + sum U_dh

where
    U_bond : harmonic bond network (elastic continuum; k_bond ~ 0.5 K_A)
    U_LJ   : Lennard-Jones cohesion driving L_o/L_d phase separation
    U_dh   : screened (Debye-Huckel) Coulomb electrostatics

Gene expression is mapped to a network area-compressibility modulus via the
linear constitutive relation of the manuscript (Eq. 1):

    K_eff(G) = K_base + alpha * G_SMT - beta * G_FAD

with K_base = 240, alpha = 120, beta = 60 (all mN m^-1). G_PLD acts only on the
electrostatic potential U_dh (via pH-dependent headgroup charge) and therefore
does not appear in Eq. 1.

Output
------
Writes ``kev_mechanics.csv`` with one row per replicate containing:
    Replicate            : integer replicate id
    Mission              : genetic-state key ('STATE_A_RIPENING',
                           'STATE_B_DEFENSE', 'STATE_C_STRESS')
    pH                   : bulk pH of the simulated medium
    Ka_Input_mNm         : mapped network modulus K_eff (mN m^-1)
    Percolation          : rigid-domain percolation fraction S_max
    Heterogeneity        : Moran's I spatial-autocorrelation index
    Rupture_Tension_mNm  : critical rupture tension gamma_crit (mN m^-1)

Note on terminology
-------------------
The CSV column names and the 'Mission' string keys are kept stable for
backwards-compatibility with the downstream analysis scripts
(soft_matter_figures.py, soft_matter_tables.py, soft_matter_report.py). The
manuscript notation mapping is:

    Code identifier              Manuscript symbol / term
    ---------------------------  ---------------------------------------
    K_base, alpha_SMT, beta_FAD  K_base, alpha, beta           (Eq. 1)
    genes['SMT']                 G_SMT  (sterol methyltransferase)
    genes['FAD']                 G_FAD  (fatty-acid desaturase)
    genes['PLD']                 G_PLD  (phospholipase D)
    Ka_Input_mNm                 K_eff  (= K_A in the SCG mesh)
    Rupture_Tension_mNm          gamma_crit  (critical rupture tension)
    Percolation                  S_max  (rigid-domain percolation fraction)
    Heterogeneity                Moran's I  (spatial heterogeneity index)
    particle type 'Lo'           liquid-ordered (sterol-rich) phase, L_o
    particle type 'Ld'           liquid-disordered (fluid) phase, L_d

Requires OpenMM with CUDA support.
"""

import numpy as np
import scipy.spatial as spatial
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import pandas as pd
import multiprocessing as mp
from openmm import *
from openmm.app import *
from openmm.unit import *
import math
import time
import os

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Execution / hardware controls (do not affect the physics).

NUM_GPUS = 1
WORKERS_PER_GPU = 20     # Optimized for N=2500 particles
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

REPLICATES = 50         # Replicates per (state, pH) condition
# pH levels spanning gastric acid (2.5), near-neutral cytosolic/endosomal (6.8),
# and blood (7.4). Only pH 2.5 vs 6.8 are reported in the manuscript (Fig. 3);
# 7.4 is retained for completeness.
PH_LEVELS = [2.5, 6.8, 7.4]

# Genetic states swept by this script. 'STATE_A_RIPENING' (State A) and
# 'STATE_B_DEFENSE' (State B) are the two phenotypic archetypes defined in the
# manuscript. 'STATE_C_STRESS' is an exploratory intermediate archetype kept for
# parameter-space coverage; it is NOT reported in the manuscript and is excluded
# from the figure/table analysis scripts downstream.
GENETIC_PROFILES = ['STATE_A_RIPENING', 'STATE_B_DEFENSE', 'STATE_C_STRESS']

# Coefficients of the linear constitutive mapping (manuscript Eq. 1):
#   K_eff(G) = K_base + alpha * G_SMT - beta * G_FAD
# Units: mN m^-1. 'gamma_PLD' is retained for completeness but is intentionally
# unused here because G_PLD acts only on the electrostatic potential U_dh (via
# pH-dependent headgroup charge) and does not enter Eq. 1. See manuscript,
# "Chemo-mechanical mapping".
GENE_COEFFS = {
    'K_base': 240.0, 'alpha_SMT': 120.0, 'beta_FAD': 60.0, 'gamma_PLD': -0.05
}

# FINAL PROFILES (Optimal parameters)
# Each profile gives the relative fold-expression of the three rate-limiting
# enzymes (genes, mapped to G_PLD / G_SMT / G_FAD) and the resulting lipid
# composition fractions. 'S_sat' + 'Sterol' define the rigid (L_o) target
# fraction; 'S_unsat' is the fluid matrix; 'Anionic' is the anionic
# phosphatidic-acid fraction whose charge is pH-dependent (U_dh).
MISSION_PROFILES = {
    'STATE_A_RIPENING': {
        # State A (High-Fluidity / Homeostatic): wild-type maturation profile.
        # High G_FAD upregulates polyunsaturated fatty acids (S_unsat ~ 80%).
        'genes': {'PLD': 2.5, 'SMT': 0.5, 'FAD': 3.0},
        # 15% Rigid (Dispersed Rafts)
        'fractions': {'S_sat': 0.1, 'S_unsat': 0.8, 'Sterol': 0.05, 'Anionic': 0.05}
    },
    'STATE_B_DEFENSE':  {
        # State B (High-Barrier / Stress-Adapted): G_SMT at the upper bound
        # (4.0-fold) drives sterol accumulation and L_o domain formation.
        'genes': {'PLD': 0.5, 'SMT': 4.0, 'FAD': 0.5},
        # 50% Rigid (Armoured Plates - Discontinuous)
        'fractions': {'S_sat': 0.25, 'S_unsat': 0.1, 'Sterol': 0.25, 'Anionic': 0.4}
    },
    'STATE_C_STRESS':   {
        # Exploratory intermediate archetype (NOT reported in the manuscript).
        'genes': {'PLD': 1.0, 'SMT': 2.0, 'FAD': 1.5},
        # 30% Rigid (Network)
        'fractions': {'S_sat': 0.15, 'S_unsat': 0.4, 'Sterol': 0.15, 'Anionic': 0.3}
    }
}


class UnitConverter:
    """Unit-conversion helper.

    OMM_TO_SI scales the OpenMM internal force/energy units into the
    mN m^-1 tension units reported in the manuscript. It is a fixed unit
    conversion factor, not a tunable physical parameter.
    """
    OMM_TO_SI = 1.660539

# =============================================================================
# 2. PHYSICS ENGINE
# =============================================================================

class KEVContinuumBuilder:
    """Builds the SCG-MD vesicle system (geometry, topology, force field).

    The vesicle is a 2D elastic shell represented as a spherical mesh of
    mesoscopic lipid patches. Bonded nearest-neighbours form a quenched
    harmonic network whose stiffness derives from the gene-mapped area
    compressibility modulus K_eff (Eq. 1); non-bonded LJ + screened-Coulomb
    terms drive phase separation and electrostatics respectively.
    """

    def __init__(self, fractions, mechanics, radius=20.0, density=0.5):
        # Density reduced to 0.5 (N ~ 2500) to match Sigma=1.0nm
        self.fractions = fractions
        self.radius = radius
        self.system = System()
        self.positions = []
        self.particle_info = []
        # Number of mesoscopic patches: N = 4*pi*R^2 * rho (manuscript, "SCG-MD").
        self.n_lipids = int(4 * math.pi * (radius**2) * density)
        # Convert the mapped modulus K_eff (mN m^-1) into OpenMM internal units.
        self.ka_omm = mechanics['ka_mNm'] / UnitConverter.OMM_TO_SI

    def _assign_clustered_types(self, coords):
        """Pre-Seeded (Active-sorting) topology generation.

        Implements the cluster-growth algorithm of Appendix A.1 (Seeded mode):
        nucleation sites are selected at random, then a breadth-first expansion
        over the 2.0 nm neighbour ball assigns the liquid-ordered (L_o, 'Lo')
        identity until the target rigid fraction (S_sat + Sterol) is reached.
        Remaining patches are the liquid-disordered fluid matrix ('Ld').
        """
        n_total = len(coords)
        target_lo = self.fractions.get('S_sat', 0) + self.fractions.get('Sterol', 0)
        n_lo_target = int(n_total * target_lo)

        types = ['Ld'] * n_total
        # Adjust seed count for N=2500
        n_seeds = max(3, int(n_total / 25))
        seeds = np.random.choice(n_total, n_seeds, replace=False)

        active_front = set(seeds)
        assigned_count = 0
        for s in seeds:
            types[s] = 'Lo'
            assigned_count += 1

        tree = spatial.cKDTree(coords)

        while assigned_count < n_lo_target and active_front:
            current = list(active_front)[np.random.randint(len(active_front))]
            neighbors = tree.query_ball_point(coords[current], 2.0) # 2.0nm search radius

            growth_occurred = False
            for n in neighbors:
                if types[n] == 'Ld' and assigned_count < n_lo_target:
                    types[n] = 'Lo'
                    active_front.add(n)
                    assigned_count += 1
                    growth_occurred = True

            if not growth_occurred:
                active_front.remove(current)

        if assigned_count < n_lo_target:
            remaining = [i for i, t in enumerate(types) if t == 'Ld']
            fill = np.random.choice(remaining, n_lo_target - assigned_count, replace=False)
            for f in fill: types[f] = 'Lo'

        return types

    def build(self):
        """Generate the Fibonacci-sphere mesh and assign particle identities."""
        # Fibonacci spiral on the unit sphere, scaled to the vesicle radius R.
        phi_golden = math.pi * (3. - math.sqrt(5.))
        temp_coords = []
        for i in range(self.n_lipids):
            y = 1 - (i / float(self.n_lipids - 1)) * 2
            rad_at_y = math.sqrt(1 - y * y)
            theta = phi_golden * i
            pos = np.array([math.cos(theta) * rad_at_y, y, math.sin(theta) * rad_at_y]) * self.radius
            temp_coords.append(pos)

        # Seeded (active-sorting) topology: consolidated L_o domains.
        types = self._assign_clustered_types(temp_coords)

        for i, pos in enumerate(temp_coords):
            # Each particle is a mesoscopic lipid-continuum patch (mass is a
            # dummy inertial weight, not a physical lipid mass).
            h_idx = self.system.addParticle(100.0 * amu)
            self.positions.append(pos * nanometer)
            p_type = types[i]
            # Class label used only to set non-bonded parameters below.
            l_class = 'Sterol' if p_type == 'Lo' else 'S_unsat'
            if p_type == 'Ld' and np.random.rand() < 0.4: l_class = 'Anionic'
            self.particle_info.append({'idx': h_idx, 'class': l_class, 'type': p_type})

    def create_forces(self, ph_val):
        """Construct the OpenMM force field: U_bond + U_LJ + U_dh + F_rad.

        ph_val : bulk pH used to set the static protonation state of anionic
                 headgroups via Henderson-Hasselbalch (pKa ~ 4.0). At gastric
                 pH 2.5 the anionic charge approaches zero (q ~ -0.03).
        """
        # --- Henderson-Hasselbalch static partial-charge approximation -------
        # q_anionic = -1 / (1 + 10^(pKa - pH)); pKa of phosphatidic acid ~ 4.0.
        # This decouples lipid protonation from dynamic solvent electrokinetics
        # (manuscript, "Force fields and mesoscopic potentials").
        pka_pa = 4.0
        q_anionic = -1.0 / (1.0 + 10**(pka_pa - ph_val))

        # --- U_bond: harmonic bond network ----------------------------------
        # U_bond(r_ij) = 1/2 * k_bond * (r_ij - r0)^2 ;  k_bond ~ 0.5 K_A
        bond = HarmonicBondForce()

        # --- U_LJ: Lennard-Jones phase-separation cohesion ------------------
        # U_LJ = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]; sigma = 0.5*(sig1+sig2),
        # epsilon = sqrt(eps1*eps2) (Lorentz-Berthelot combining rules).
        vdw = CustomNonbondedForce(
            "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
            "sigma=0.5*(sig1+sig2); epsilon=sqrt(eps1*eps2)"
        )
        vdw.addPerParticleParameter("sig")
        vdw.addPerParticleParameter("eps")
        vdw.setCutoffDistance(3.0*nanometer)

        # --- U_dh: screened Coulomb (Debye-Huckel) electrostatics -----------
        # U_dh = (1/(4*pi*eps0*eps_r)) * q1*q2/r * exp(-r/lambda_D)
        # The '138.935' prefactor folds in 1/(4*pi*eps0*eps_r) in OpenMM units.
        # Debye length lambda_D = 1.0 nm (manuscript: ~150 mM gastric ionic
        # strength), fixed as a global static parameter.
        elec = CustomNonbondedForce("138.935*q1*q2*exp(-r/debye)/r")
        elec.addPerParticleParameter("q")
        elec.addGlobalParameter("debye", 1.0*nanometer)

        # --- F_rad: simulated osmotic inflation (turgor) --------------------
        # A radial external force F_rad = k_pressure * r_hat is applied to every
        # particle. k_pressure is ramped in the worker loop to drive the vesicle
        # to its yield point. The negative sign pulls particles outward.
        stressor = CustomExternalForce("-k_pressure * sqrt(x*x+y*y+z*z)")
        stressor.addGlobalParameter("k_pressure", 0.0)
        for i in range(self.system.getNumParticles()):
            stressor.addParticle(i, [])
        self.system.addForce(stressor)

        # --- Bond network construction (nearest neighbours within 2.5 nm) ---
        coords = [p.value_in_unit(nanometer) for p in self.positions]
        tree = spatial.cKDTree(coords)
        pairs = list(tree.query_pairs(r=2.5))

        # Bond Scaling 0.5: k_bond ~ 0.5 K_A (Seung & Nelson, 1988).
        k_bond_val = self.ka_omm * 0.5

        for i, j in pairs:
            r0 = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            bond.addBond(i, j, r0*nanometer, k_bond_val * kilojoule_per_mole/nanometer**2)

        # --- Per-particle non-bonded parameters -----------------------------
        # epsilon_Lo = 15.0 kJ mol^-1 (~6 kBT at 300 K) for sterol-rich L_o
        # domains; epsilon_mix = 1.0 kJ mol^-1 for fluid L_d interactions.
        for info in self.particle_info:
            p_type = info['type']
            eps = 15.0 if p_type == 'Lo' else 1.0
            q = q_anionic if info['class'] == 'Anionic' else 0.0
            vdw.addParticle([1.0*nanometer, eps])
            elec.addParticle([q])

        self.system.addForce(bond)
        self.system.addForce(vdw)
        self.system.addForce(elec)
        return self.system

# =============================================================================
# 3. ANALYSIS
# =============================================================================

def calculate_topology(pos, info, cutoff=2.5):
    """Quantify rigid-phase morphology: percolation fraction S_max and Moran's I.

    S_max (manuscript, Appendix A.2): fraction of all membrane patches belonging
    to the largest connected rigid (L_o) component, computed with a Union-Find
    (connected-components) algorithm on the sterol subgraph. S_max ~ 0.15
    indicates dispersed domains; S_max >= 0.45 indicates a monolithic percolating
    network (for a ~50% rigid composition).

    Moran's I (manuscript, Appendix A.2): spatial-autocorrelation index,
        I = (N/W) * sum_ij w_ij (x_i - xbar)(x_j - xbar) / sum_i (x_i - xbar)^2
    where w_ij = 1 for neighbour pairs, 0 otherwise. I ~ 0 indicates random
    mixing (Spontaneous); I -> 1 indicates high spatial clustering (Seeded).
    """
    if hasattr(pos, 'value_in_unit'): pos = pos.value_in_unit(nanometer)
    # Binary rigid-phase indicator: 1.0 for L_o, 0.0 for L_d.
    vals = np.array([1.0 if p['type'] == 'Lo' else 0.0 for p in info])

    tree = spatial.cKDTree(pos)
    pairs = list(tree.query_pairs(cutoff))

    # Moran's I
    mean_x = np.mean(vals)
    denom = np.sum((vals - mean_x)**2)
    numer, sum_w = 0.0, 0.0
    if len(pairs) > 0:
        i_idx = np.array([p[0] for p in pairs])
        j_idx = np.array([p[1] for p in pairs])
        term = (vals[i_idx] - mean_x) * (vals[j_idx] - mean_x)
        numer = np.sum(term)
        sum_w = len(pairs)
    moran_i = (len(vals) / sum_w * numer / denom) if (denom != 0 and sum_w != 0) else 0.0

    # Percolation (S_max) via connected components on the L_o subgraph.
    lo_indices = [i for i, x in enumerate(vals) if x == 1.0]
    if not lo_indices: return 0.0, moran_i

    # Fast Sparse Matrix construction
    row_idx, col_idx = [], []
    for i, j in pairs:
        if vals[i] == 1.0 and vals[j] == 1.0:
            row_idx.append(i)
            col_idx.append(j)

    if not row_idx: return (1.0/len(info)), moran_i

    data = np.ones(len(row_idx))
    adj = sp.coo_matrix((data, (row_idx, col_idx)), shape=(len(info), len(info)))
    n_comps, labels = csgraph.connected_components(adj, directed=False)

    lo_labels = labels[lo_indices]
    largest = np.bincount(lo_labels).max() if len(lo_labels) > 0 else 0
    return (largest / len(info)), moran_i

# =============================================================================
# 4. WORKER
# =============================================================================

def run_simulation_task(task_data):
    """Run a single (state, pH) replicate: build, equilibrate, inflate to rupture.

    Returns a one-element list with the result dict (or an error dict on
    failure), to be collected by the multiprocessing pool.
    """
    try:
        task_id, gpu_id, mission_name, ph_val = task_data

        # Setup
        profile = MISSION_PROFILES[mission_name]
        genes = profile['genes']
        fractions = profile['fractions']
        # Eq. 1: K_eff = K_base + alpha*G_SMT - beta*G_FAD  (G_PLD excluded).
        ka_raw = GENE_COEFFS['K_base'] + (GENE_COEFFS['alpha_SMT']*genes['SMT']) - (GENE_COEFFS['beta_FAD']*genes['FAD'])
        # Floor to keep the network modulus physical (avoid negative/zero K_A).
        ka_eff = max(10.0, ka_raw)

        # Build (N~2500)
        builder = KEVContinuumBuilder(fractions, {'ka_mNm': ka_eff}, radius=20.0)
        builder.build()
        system = builder.create_forces(ph_val)

        # --- Integrator & thermodynamics (Appendix B.1) --------------------
        # T = 300 K, Langevin friction gamma = 1.0 ps^-1, dt = 4 fs.
        platform = Platform.getPlatformByName('CUDA')
        props = {'DeviceIndex': str(gpu_id), 'Precision': 'mixed'}
        integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.004*picosecond)
        sim = Simulation(Topology(), system, integrator, platform, props)
        sim.context.setPositions(builder.positions)
        sim.minimizeEnergy()

        # Equilibration (10k steps sufficient for N=2500)
        sim.step(10000)

        # Post-equilibration topology (quenched mesh; diffusion is frozen on
        # the ns loading timescale, per manuscript Limitations).
        pos_eq = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
        perc, het = calculate_topology(pos_eq, builder.particle_info)

        # --- Simulated osmotic inflation to rupture -------------------------
        # F_rad = k_pressure * r_hat ramped at 0.5 kJ mol^-1 nm^-1 per 500
        # steps. With dt = 4 fs this is 0.25 kJ mol^-1 nm^-1 ps^-1 (manuscript,
        # "Simulated osmotic inflation"). Failure = irreversible plastic
        # expansion > 15% radial strain.
        r0 = 20.0
        rupture_tension = 0.0
        failed = False
        pressure_k = 0.0

        for step in range(150):
            pressure_k += 0.5
            sim.context.setParameter("k_pressure", pressure_k)
            sim.step(500)

            state = sim.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            current_r = np.mean(np.linalg.norm(pos, axis=1))
            # Young-Laplace: gamma = P * R / 2  (here pressure_k plays the role
            # of the transmembrane pressure difference; OMM_TO_SI converts to
            # mN m^-1).
            current_tension = pressure_k * current_r / 2.0

            if current_r > r0 * 1.15:
                failed = True
                rupture_tension = current_tension * UnitConverter.OMM_TO_SI
                break

        if not failed: rupture_tension = current_tension * UnitConverter.OMM_TO_SI

        return [{
            'Replicate': task_id, 'Mission': mission_name, 'pH': ph_val,
            'Ka_Input_mNm': ka_eff, 'Percolation': perc, 'Heterogeneity': het,
            'Rupture_Tension_mNm': rupture_tension
        }]
    except Exception as e:
        return [{'error': str(e), 'Replicate': task_id}]

# =============================================================================
# 5. MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"--- KEV Mechanics ---")
    tasks = []
    task_id = 0
    for m in GENETIC_PROFILES:
        for ph in PH_LEVELS:
            for i in range(REPLICATES):
                tasks.append((task_id, task_id % NUM_GPUS, m, ph))
                task_id += 1

    ctx = mp.get_context('spawn')
    all_results = []
    start = time.time()
    with ctx.Pool(processes=TOTAL_WORKERS) as pool:
        for res_batch in pool.imap_unordered(run_simulation_task, tasks):
            if res_batch: all_results.extend(res_batch)
            if len(all_results) % 25 == 0:
                print(f" > {len(all_results)}/{len(tasks)} complete...",flush=True)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("kev_mechanics.csv", index=False)
        print(f"\nDONE in {time.time()-start:.1f}s.")
        print(df.groupby('Mission')[['Percolation', 'Rupture_Tension_mNm']].mean())
