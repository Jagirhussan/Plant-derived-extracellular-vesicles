"""
Genomic regulation of chemo-mechanical stability in plant-derived
extracellular vesicles (PDEVs): a multiscale model of composite reinforcement.

kev_seeded_vs_spontaneous.py
============================

Topology-controlled simulated osmotic inflation of coarse-grained PDEV
vesicles. This script isolates the contribution of lipid *organisation* by
comparing two initialisation modes at identical lipid composition
(manuscript, "Topological initialisation: sorting vs. mixing" and Appendix A.1;
results in "Mechanical hierarchy of states" and "Topological reinforcement
mechanism"; Figs. 3, 4, 5 and the data behind Tables 1-4).

Initialisation modes
--------------------
    Seeded     (Active sorting)   : cluster-growth / BFS pre-nucleation of
                                     liquid-ordered (L_o) domains, mimicking
                                     Golgi/flippase-mediated raft sorting.
    Spontaneous (Entropic mixing) : uniform random assignment of lipid
                                     identities (numpy.random.shuffle analogue),
                                     the randomised control.

Both modes use the *same* lipid fractions and the *same* mapped network modulus
K_eff, so any mechanical difference is attributable to topology alone.

Framework
---------
A 2D Supra-molecular Coarse-Grained Molecular Dynamics (SCG-MD) mesh built on
OpenMM. Each simulated *particle* is a mesoscopic patch of the lipid continuum
(not a single atom or lipid molecule). The vesicle is a spherical shell
(R = 20 nm, surface density rho = 0.5 nm^-2, N ~ 2500 patches) whose total
potential energy is

    U_total = sum U_bond + sum U_LJ + sum U_dh

Gene expression is mapped to a network area-compressibility modulus via the
linear constitutive relation of the manuscript (Eq. 1):

    K_eff(G) = K_base + alpha * G_SMT - beta * G_FAD

with K_base = 240, alpha = 120, beta = 60 (all mN m^-1). G_PLD acts only on the
electrostatic potential U_dh and does not appear in Eq. 1.

Electrostatics are evaluated at a fixed neutral pH (6.8) so that the comparison
isolates topology; pH-dependent acid-shock mechanics are handled by
kev_mechanics.py.

Output
------
Writes ``kev_seeded_vs_spontaneous.csv`` with one row per replicate containing:
    Replicate            : integer replicate id
    Mission              : genetic-state key ('STATE_A_RIPENING' or
                           'STATE_B_DEFENSE')
    Mode                 : 'SEEDED' or 'SPONTANEOUS'
    Percolation          : rigid-domain percolation fraction S_max
    Heterogeneity        : Moran's I spatial-autocorrelation index
    Rupture_Tension_mNm  : critical rupture tension gamma_crit (mN m^-1)
    Toughness            : area under the tension-strain curve (integral of
                           gamma d epsilon)
    Strain_Curve         : areal strain trajectory (per loading step)
    Tension_Curve        : membrane-tension trajectory (per loading step)

Note on terminology
-------------------
The CSV column names and the 'Mission'/'Mode' string keys are kept stable for
backwards-compatibility with the downstream analysis scripts
(soft_matter_figures.py, soft_matter_tables.py). The manuscript notation
mapping is:

    Code identifier              Manuscript symbol / term
    ---------------------------  ---------------------------------------
    K_base, alpha_SMT, beta_FAD  K_base, alpha, beta           (Eq. 1)
    genes['SMT']                 G_SMT  (sterol methyltransferase)
    genes['FAD']                 G_FAD  (fatty-acid desaturase)
    genes['PLD']                 G_PLD  (phospholipase D)
    Rupture_Tension_mNm          gamma_crit  (critical rupture tension)
    Percolation                  S_max  (rigid-domain percolation fraction)
    Heterogeneity                Moran's I  (spatial heterogeneity index)
    Mode 'SEEDED'                Seeded (Active sorting)
    Mode 'SPONTANEOUS'           Spontaneous (Entropic mixing)
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

NUM_GPUS = 1
WORKERS_PER_GPU = 20   
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

# Topological initialisation modes (manuscript, "Topological initialisation").
MODES = ['SEEDED', 'SPONTANEOUS']
# Genetic states. Only the two manuscript archetypes are compared here;
# State C (exploratory intermediate) is handled by kev_mechanics.py.
MISSIONS = ['STATE_A_RIPENING', 'STATE_B_DEFENSE']
REPLICATES = 50       # Full statistical power

# Physics Parameters (Appendix B.1): T = 300 K, Langevin friction gamma = 1.0
# ps^-1, time step dt = 4 fs.
TEMP = 300*kelvin
FRICTION = 1.0/picosecond
TIMESTEP = 0.004*picosecond

# Debye screening length lambda_D for the screened Coulomb potential U_dh.
# The manuscript states lambda_D = 1.0 nm (representing the ~150 mM ionic
# strength of gastric fluid). 

DEBYE_LENGTH = 1.0*nanometer

# Coefficients of the linear constitutive mapping (manuscript Eq. 1):
#   K_eff(G) = K_base + alpha * G_SMT - beta * G_FAD
# Units: mN m^-1. 'gamma_PLD' is retained for completeness but is intentionally
# unused here because G_PLD acts only on the electrostatic potential U_dh (via
# pH-dependent headgroup charge) and does not enter Eq. 1.
GENE_COEFFS = {
    'K_base': 240.0, 'alpha_SMT': 120.0, 'beta_FAD': 60.0, 'gamma_PLD': -0.05
}

# Genetic-state profiles (identical to kev_mechanics.py for the two manuscript
# archetypes). 'S_sat' + 'Sterol' define the rigid (L_o) target fraction;
# 'S_unsat' is the fluid matrix; 'Anionic' is the pH-sensitive phosphatidic-
# acid fraction (here fixed at the pH 6.8 protonation state).
GENETIC_PROFILES = {
    'STATE_A_RIPENING': {
        # State A (High-Fluidity / Homeostatic): high G_FAD, polyunsaturated.
        'genes': {'PLD': 2.5, 'SMT': 0.5, 'FAD': 3.0},
        'fractions': {'S_sat': 0.1, 'S_unsat': 0.8, 'Sterol': 0.05, 'Anionic': 0.05}
    },
    'STATE_B_DEFENSE':  {
        # State B (High-Barrier / Stress-Adapted): high G_SMT, sterol-rich.
        'genes': {'PLD': 0.5, 'SMT': 4.0, 'FAD': 0.5},
        'fractions': {'S_sat': 0.25, 'S_unsat': 0.1, 'Sterol': 0.25, 'Anionic': 0.4}
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
        self.fractions = fractions
        self.radius = radius
        self.system = System()
        self.positions = []
        self.particle_info = []
        # Number of mesoscopic patches: N = 4*pi*R^2 * rho (manuscript, "SCG-MD").
        self.n_lipids = int(4 * math.pi * (radius**2) * density)
        # Convert the mapped modulus K_eff (mN m^-1) into OpenMM internal units.
        self.ka_omm = mechanics['ka_mNm'] / UnitConverter.OMM_TO_SI

    def _assign_types(self, coords, mode='SEEDED'):
        """Assign L_o / L_d particle identities according to the init mode.

        Seeded (Active sorting) -- Appendix A.1 cluster-growth algorithm:
            nucleation sites are selected at random, then a breadth-first
            expansion over the 2.0 nm neighbour ball assigns the
            liquid-ordered (L_o, 'Lo') identity until the target rigid
            fraction (S_sat + Sterol) is reached. Models Golgi/flippase-
            mediated pre-nucleation of raft domains.

        Spontaneous (Entropic mixing) -- randomised control:
            L_o identities are assigned via a uniform random draw
            (numpy.random.choice), simulating a membrane formed by pure
            entropic mixing without active cellular sorting.
        """
        n_total = len(coords)
        target_lo = self.fractions.get('S_sat', 0) + self.fractions.get('Sterol', 0)
        n_lo_target = int(n_total * target_lo)

        types = ['Ld'] * n_total

        if mode == 'SPONTANEOUS':
            # Random distribution (Control)
            indices = np.random.choice(n_total, n_lo_target, replace=False)
            for i in indices: types[i] = 'Lo'

        elif mode == 'SEEDED':
            # Cluster growth (Hypothesis)
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
                neighbors = tree.query_ball_point(coords[current], 2.0)
                growth = False
                for n in neighbors:
                    if types[n] == 'Ld' and assigned_count < n_lo_target:
                        types[n] = 'Lo'
                        active_front.add(n)
                        assigned_count += 1
                        growth = True
                if not growth: active_front.remove(current)

            if assigned_count < n_lo_target:
                rem = [i for i, t in enumerate(types) if t == 'Ld']
                fill = np.random.choice(rem, n_lo_target - assigned_count, replace=False)
                for f in fill: types[f] = 'Lo'

        return types

    def build(self, mode='SEEDED'):
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

        types = self._assign_types(temp_coords, mode=mode)

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

    def create_forces(self):
        """Construct the OpenMM force field: U_bond + U_LJ + U_dh + F_rad.

        Electrostatics are fixed at pH 6.8 (neutral) so that the Seeded vs.
        Spontaneous comparison isolates topology rather than pH effects.
        """
        # pH fixed at 6.8 for mechanics comparison.
        # Henderson-Hasselbalch static partial-charge approximation:
        # q_anionic = -1 / (1 + 10^(pKa - pH)); pKa of phosphatidic acid ~ 4.0.
        pka_pa = 4.0
        q_anionic = -1.0 / (1.0 + 10**(pka_pa - 6.8))

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
        # lambda_D = DEBYE_LENGTH .
        elec = CustomNonbondedForce("138.935*q1*q2*exp(-r/debye)/r")
        elec.addPerParticleParameter("q")
        elec.addGlobalParameter("debye", DEBYE_LENGTH)

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
# 3. WORKER
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
    vals = np.array([1.0 if p['type'] == 'Lo' else 0.0 for p in info])
    lo_indices = [i for i, x in enumerate(vals) if x == 1.0]
    if not lo_indices: return 0.0, 0.0

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

def run_simulation_task(task_data):
    """Run a single (state, mode) replicate: build, equilibrate, inflate.

    Records the full tension-strain trajectory so that Toughness (area under
    the curve) and the stress-strain curves (Fig. 4) can be computed.

    Returns a one-element list with the result dict (or an error dict on
    failure), to be collected by the multiprocessing pool.
    """
    try:
        task_id, gpu_id, mission_name, mode = task_data

        profile = GENETIC_PROFILES[mission_name]
        genes = profile['genes']
        fractions = profile['fractions']
        # Eq. 1: K_eff = K_base + alpha*G_SMT - beta*G_FAD  (G_PLD excluded).
        ka_raw = GENE_COEFFS['K_base'] + (GENE_COEFFS['alpha_SMT']*genes['SMT']) - (GENE_COEFFS['beta_FAD']*genes['FAD'])
        # Floor to keep the network modulus physical (avoid negative/zero K_A).
        ka_eff = max(10.0, ka_raw)

        builder = KEVContinuumBuilder(fractions, {'ka_mNm': ka_eff}, radius=20.0)
        builder.build(mode=mode)
        system = builder.create_forces()

        # --- Integrator & thermodynamics (Appendix B.1) --------------------
        platform = Platform.getPlatformByName('CUDA')
        props = {'DeviceIndex': str(gpu_id), 'Precision': 'mixed'}
        integrator = LangevinIntegrator(TEMP, FRICTION, TIMESTEP)
        sim = Simulation(Topology(), system, integrator, platform, props)
        sim.context.setPositions(builder.positions)
        sim.minimizeEnergy()

        # Extended Equilibration for Spontaneous Control.
        # 100k steps = ~400 ps. Sufficient for local diffusion in CG model so
        # that the randomised control can locally reorganise before loading;
        # Seeded vesicles equilibrate in 10k steps as the domains are
        # pre-nucleated.
        eq_steps = 100000 if mode == 'SPONTANEOUS' else 10000
        sim.step(eq_steps)

        # Post-equilibration topology (quenched mesh; diffusion is frozen on
        # the ns loading timescale, per manuscript Limitations).
        pos_eq = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
        perc, het = calculate_topology(pos_eq, builder.particle_info)

        # --- Simulated osmotic inflation to rupture -------------------------
        # F_rad = k_pressure * r_hat ramped at 0.5 kJ mol^-1 nm^-1 per 500
        # steps. With dt = 4 fs this is 0.25 kJ mol^-1 nm^-1 ps^-1 (manuscript,
        # "Simulated osmotic inflation"). Failure = irreversible plastic
        # expansion > 15% radial strain.
        # Stress-Strain Data
        r0_state = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
        r0 = np.mean(np.linalg.norm(r0_state, axis=1))
        initial_area = 4 * np.pi * (r0**2)

        rupture_tension = 0.0
        failed = False
        pressure_k = 0.0
        strain_history = []
        tension_history = []

        for step in range(150):
            pressure_k += 0.5
            sim.context.setParameter("k_pressure", pressure_k)
            sim.step(500)

            state = sim.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            current_r = np.mean(np.linalg.norm(pos, axis=1))
            current_area = 4 * np.pi * (current_r**2)

            # Areal strain epsilon = (A - A0) / A0.
            strain = (current_area - initial_area) / initial_area
            # Young-Laplace: gamma = P * R / 2  (OMM_TO_SI converts to mN m^-1).
            tension = (pressure_k * current_r / 2.0) * UnitConverter.OMM_TO_SI

            strain_history.append(strain)
            tension_history.append(tension)

            if current_r > r0 * 1.15:
                failed = True
                rupture_tension = tension
                break

        if not failed: rupture_tension = tension

        # Toughness ~ integral of gamma d epsilon (area under the curve).
        toughness = np.trapz(tension_history, strain_history)

        return [{
            'Replicate': task_id, 'Mission': mission_name, 'Mode': mode,
            'Percolation': perc, 'Heterogeneity': het,
            'Rupture_Tension_mNm': rupture_tension, 'Toughness': toughness,
            'Strain_Curve': strain_history,
            'Tension_Curve': tension_history
        }]
    except Exception as e:
        return [{'error': str(e), 'Replicate': task_id}]

# =============================================================================
# 5. MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"--- KEV CONTROL RUN (N={REPLICATES}) ---")
    tasks = []
    task_id = 0
    for mode in MODES:
        for m in MISSIONS:
            for i in range(REPLICATES):
                assigned_gpu = task_id % NUM_GPUS
                tasks.append((task_id, assigned_gpu, m, mode))
                task_id += 1

    ctx = mp.get_context('spawn')
    all_results = []

    print(f"Queue: {len(tasks)} simulations on {NUM_GPUS} GPUs.",flush=True)

    with ctx.Pool(processes=TOTAL_WORKERS) as pool:
        for res_batch in pool.imap_unordered(run_simulation_task, tasks):
            if res_batch: all_results.extend(res_batch)
            if len(all_results) % 25 == 0: print(f" > {len(all_results)} completed...")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("kev_seeded_vs_spontaneous.csv", index=False)
        print("\n--- PRODUCTION SUMMARY ---")
        print(df.groupby(['Mode', 'Mission'])[['Rupture_Tension_mNm', 'Toughness', 'Heterogeneity']].agg(['mean', 'sem']))
