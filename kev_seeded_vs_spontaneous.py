'''
Seeded vs Spontaneous configurations
'''
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

NUM_GPUS = 2
WORKERS_PER_GPU = 10   # Safe load for long runs
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

MODES = ['SEEDED', 'SPONTANEOUS'] 
MISSIONS = ['STATE_A_RIPENING', 'STATE_B_DEFENSE'] 
REPLICATES = 50       # Full statistical power

# Physics Parameters
TEMP = 300*kelvin
FRICTION = 1.0/picosecond
TIMESTEP = 0.004*picosecond
DEBYE_LENGTH = 0.8*nanometer # Physiological salt (approx 150mM)

GENE_COEFFS = {
    'K_base': 240.0, 'alpha_SMT': 120.0, 'beta_FAD': 60.0, 'gamma_PLD': -0.05 
}

GENETIC_PROFILES = {
    'STATE_A_RIPENING': {
        'genes': {'PLD': 2.5, 'SMT': 0.5, 'FAD': 3.0},
        'fractions': {'S_sat': 0.1, 'S_unsat': 0.8, 'Sterol': 0.05, 'Anionic': 0.05}
    },
    'STATE_B_DEFENSE':  {
        'genes': {'PLD': 0.5, 'SMT': 4.0, 'FAD': 0.5},
        'fractions': {'S_sat': 0.25, 'S_unsat': 0.1, 'Sterol': 0.25, 'Anionic': 0.4}
    }
}

class UnitConverter:
    OMM_TO_SI = 1.660539 

# =============================================================================
# 2. PHYSICS ENGINE
# =============================================================================

class KEVContinuumBuilder:
    def __init__(self, fractions, mechanics, radius=20.0, density=0.5): 
        self.fractions = fractions
        self.radius = radius 
        self.system = System()
        self.positions = []
        self.particle_info = [] 
        self.n_lipids = int(4 * math.pi * (radius**2) * density)
        self.ka_omm = mechanics['ka_mNm'] / UnitConverter.OMM_TO_SI

    def _assign_types(self, coords, mode='SEEDED'):
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
            h_idx = self.system.addParticle(100.0 * amu)
            self.positions.append(pos * nanometer)
            p_type = types[i]
            l_class = 'Sterol' if p_type == 'Lo' else 'S_unsat'
            if p_type == 'Ld' and np.random.rand() < 0.4: l_class = 'Anionic' 
            self.particle_info.append({'idx': h_idx, 'class': l_class, 'type': p_type})

    def create_forces(self):
        # pH fixed at 6.8 for mechanics comparison
        pka_pa = 4.0
        q_anionic = -1.0 / (1.0 + 10**(pka_pa - 6.8))
        
        bond = HarmonicBondForce()
        vdw = CustomNonbondedForce(
            "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
            "sigma=0.5*(sig1+sig2); epsilon=sqrt(eps1*eps2)"
        )
        vdw.addPerParticleParameter("sig")
        vdw.addPerParticleParameter("eps")
        vdw.setCutoffDistance(3.0*nanometer) 

        elec = CustomNonbondedForce("138.935*q1*q2*exp(-r/debye)/r") 
        elec.addPerParticleParameter("q")
        elec.addGlobalParameter("debye", DEBYE_LENGTH) 

        stressor = CustomExternalForce("-k_pressure * sqrt(x*x+y*y+z*z)") 
        stressor.addGlobalParameter("k_pressure", 0.0)
        for i in range(self.system.getNumParticles()):
            stressor.addParticle(i, [])
        self.system.addForce(stressor)

        coords = [p.value_in_unit(nanometer) for p in self.positions]
        tree = spatial.cKDTree(coords)
        pairs = list(tree.query_pairs(r=2.5))
        
        k_bond_val = self.ka_omm * 0.5
        
        for i, j in pairs:
            r0 = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            bond.addBond(i, j, r0*nanometer, k_bond_val * kilojoule_per_mole/nanometer**2)

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
    vals = np.array([1.0 if p['type'] == 'Lo' else 0.0 for p in info])
    lo_indices = [i for i, x in enumerate(vals) if x == 1.0]
    if not lo_indices: return 0.0, 0.0
    
    tree = spatial.cKDTree(pos)
    pairs = list(tree.query_pairs(cutoff))
    
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
    try:
        task_id, gpu_id, mission_name, mode = task_data
        
        profile = GENETIC_PROFILES[mission_name]
        genes = profile['genes']
        fractions = profile['fractions']
        ka_raw = GENE_COEFFS['K_base'] + (GENE_COEFFS['alpha_SMT']*genes['SMT']) - (GENE_COEFFS['beta_FAD']*genes['FAD'])
        ka_eff = max(10.0, ka_raw)

        builder = KEVContinuumBuilder(fractions, {'ka_mNm': ka_eff}, radius=20.0)
        builder.build(mode=mode)
        system = builder.create_forces()
        
        platform = Platform.getPlatformByName('CUDA')
        props = {'DeviceIndex': str(gpu_id), 'Precision': 'mixed'}
        integrator = LangevinIntegrator(TEMP, FRICTION, TIMESTEP)
        sim = Simulation(Topology(), system, integrator, platform, props)
        sim.context.setPositions(builder.positions)
        sim.minimizeEnergy()
        
        # Extended Equilibration for Spontaneous Control
        # 100k steps = ~400 ps. Sufficient for local diffusion in CG model.
        eq_steps = 100000 if mode == 'SPONTANEOUS' else 10000
        sim.step(eq_steps)
        
        pos_eq = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
        perc, het = calculate_topology(pos_eq, builder.particle_info)
        
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
            
            strain = (current_area - initial_area) / initial_area
            tension = (pressure_k * current_r / 2.0) * UnitConverter.OMM_TO_SI
            
            strain_history.append(strain)
            tension_history.append(tension)
            
            if current_r > r0 * 1.15: 
                failed = True
                rupture_tension = tension
                break
        
        if not failed: rupture_tension = tension
        
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