'''
 Test KEV rupture and digestion mechanics
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
WORKERS_PER_GPU = 10     # Optimized for N=2500 particles
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

REPLICATES = 50         
PH_LEVELS = [2.5, 6.8, 7.4] 
GENETIC_PROFILES = ['STATE_A_RIPENING', 'STATE_B_DEFENSE', 'STATE_C_STRESS']

GENE_COEFFS = {
    'K_base': 240.0, 'alpha_SMT': 120.0, 'beta_FAD': 60.0, 'gamma_PLD': -0.05 
}

# FINAL PROFILES (Goldilocks Tuned)
MISSION_PROFILES = {
    'STATE_A_RIPENING': {
        'genes': {'PLD': 2.5, 'SMT': 0.5, 'FAD': 3.0},
        # 15% Rigid (Dispersed Rafts)
        'fractions': {'S_sat': 0.1, 'S_unsat': 0.8, 'Sterol': 0.05, 'Anionic': 0.05}
    },
    'STATE_B_DEFENSE':  {
        'genes': {'PLD': 0.5, 'SMT': 4.0, 'FAD': 0.5},
        # 50% Rigid (Armoured Plates - Discontinuous)
        'fractions': {'S_sat': 0.25, 'S_unsat': 0.1, 'Sterol': 0.25, 'Anionic': 0.4}
    },
    'STATE_C_STRESS':   {
        'genes': {'PLD': 1.0, 'SMT': 2.0, 'FAD': 1.5},
        # 30% Rigid (Network)
        'fractions': {'S_sat': 0.15, 'S_unsat': 0.4, 'Sterol': 0.15, 'Anionic': 0.3}
    }
}

class UnitConverter:
    OMM_TO_SI = 1.660539 

# =============================================================================
# 2. PHYSICS ENGINE
# =============================================================================

class KEVContinuumBuilder:
    def __init__(self, fractions, mechanics, radius=20.0, density=0.5): 
        # Density reduced to 0.5 (N ~ 2500) to match Sigma=1.0nm
        self.fractions = fractions
        self.radius = radius 
        self.system = System()
        self.positions = []
        self.particle_info = [] 
        self.n_lipids = int(4 * math.pi * (radius**2) * density)
        self.ka_omm = mechanics['ka_mNm'] / UnitConverter.OMM_TO_SI

    def _assign_clustered_types(self, coords):
        """Pre-Seeded Topology Generation."""
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
        phi_golden = math.pi * (3. - math.sqrt(5.))
        temp_coords = []
        for i in range(self.n_lipids):
            y = 1 - (i / float(self.n_lipids - 1)) * 2
            rad_at_y = math.sqrt(1 - y * y)
            theta = phi_golden * i
            pos = np.array([math.cos(theta) * rad_at_y, y, math.sin(theta) * rad_at_y]) * self.radius
            temp_coords.append(pos)
            
        types = self._assign_clustered_types(temp_coords)
        
        for i, pos in enumerate(temp_coords):
            h_idx = self.system.addParticle(100.0 * amu)
            self.positions.append(pos * nanometer)
            p_type = types[i]
            l_class = 'Sterol' if p_type == 'Lo' else 'S_unsat'
            if p_type == 'Ld' and np.random.rand() < 0.4: l_class = 'Anionic' 
            self.particle_info.append({'idx': h_idx, 'class': l_class, 'type': p_type})

    def create_forces(self, ph_val):
        pka_pa = 4.0
        q_anionic = -1.0 / (1.0 + 10**(pka_pa - ph_val))
        
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
        elec.addGlobalParameter("debye", 1.0*nanometer) 

        stressor = CustomExternalForce("-k_pressure * sqrt(x*x+y*y+z*z)") 
        stressor.addGlobalParameter("k_pressure", 0.0)
        for i in range(self.system.getNumParticles()):
            stressor.addParticle(i, [])
        self.system.addForce(stressor)

        coords = [p.value_in_unit(nanometer) for p in self.positions]
        tree = spatial.cKDTree(coords)
        pairs = list(tree.query_pairs(r=2.5))
        
        # Bond Scaling 0.5
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
# 3. ANALYSIS 
# =============================================================================

def calculate_topology(pos, info, cutoff=2.5):
    if hasattr(pos, 'value_in_unit'): pos = pos.value_in_unit(nanometer)
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

    # Percolation
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
    try:
        task_id, gpu_id, mission_name, ph_val = task_data
        
        # Setup
        profile = MISSION_PROFILES[mission_name]
        genes = profile['genes']
        fractions = profile['fractions']
        ka_raw = GENE_COEFFS['K_base'] + (GENE_COEFFS['alpha_SMT']*genes['SMT']) - (GENE_COEFFS['beta_FAD']*genes['FAD'])
        ka_eff = max(10.0, ka_raw)

        # Build (N~2500)
        builder = KEVContinuumBuilder(fractions, {'ka_mNm': ka_eff}, radius=20.0)
        builder.build()
        system = builder.create_forces(ph_val)
        
        platform = Platform.getPlatformByName('CUDA')
        props = {'DeviceIndex': str(gpu_id), 'Precision': 'mixed'}
        integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.004*picosecond)
        sim = Simulation(Topology(), system, integrator, platform, props)
        sim.context.setPositions(builder.positions)
        sim.minimizeEnergy()
        
        # Equilibration (10k steps sufficient for N=2500)
        sim.step(10000) 
        
        pos_eq = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
        perc, het = calculate_topology(pos_eq, builder.particle_info)
        
        # Rupture
        r0 = 20.0
        rupture_tension = 0.0
        failed = False
        pressure_k = 0.0
        
        for step in range(100):
            pressure_k += 0.5
            sim.context.setParameter("k_pressure", pressure_k)
            sim.step(500)
            
            state = sim.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            current_r = np.mean(np.linalg.norm(pos, axis=1))
            current_tension = pressure_k * current_r / 2.0 
            
            if current_r > r0 * 1.10: 
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