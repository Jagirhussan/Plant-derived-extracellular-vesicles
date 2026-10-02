# Reproduction package

Transcriptional regulation of chemo-mechanical stability in plant-derived
extracellular vesicles: a multiscale model of composite reinforcement.

This folder contains every script needed to regenerate the simulation data,
figures and tables of the manuscript, plus a notebook that recomputes each
number quoted in the text directly from the per-replicate CSVs.

## The problem

Plant-derived extracellular vesicles (PDEVs) survive the gastrointestinal
tract far better than mammalian exosomes or synthetic liposomes, but the
biophysical rules that map plant gene regulation onto the mechanical
stability of the secreted vesicle are unknown. In particular it is unclear
whether stability follows from membrane composition alone or requires a
specific supramolecular organisation of rigid, sterol-rich domains.

## The solution

A multiscale chemo-mechanical model links a parameterised genetic state
space (sterol methyltransferase, fatty-acid desaturase and phospholipase D
expression) to supra-molecular coarse-grained molecular dynamics (SCG-MD)
of a composite membrane shell. Two genetic archetypes are simulated, a
fluid Ripening state and a rigid sterol-rich Defence state, each
instantiated with actively sorted (Seeded) or randomly mixed (Spontaneous)
domain topologies. Simulated osmotic inflation with an exact equatorial
force-balance tension measurement yields critical tensions, toughness and
stored elastic energy; a 64-fold loading-rate ladder separates dynamic
strength from the quasi-static material limit.

Key findings reproduced by this package:

- Both genetic states converge to finite quasi-static strengths
  (11.1 and 54.2 mN/m); their ratio (4.9) recovers most of the 5.75-fold
  genomic stiffness contrast, while the production-rate contrast (+13.5%)
  is a rate-compressed lower bound.
- Seeded and Spontaneous architectures are mechanically degenerate
  (within 1 to 2.5 percent) at every loading rate tested; domain layout
  leaves only an energetic, not a mechanical, signature.
- Stored elastic energy density is size-invariant from R = 20 to 40 nm
  and gastric acid shock yields a bounded +2 percent reactive toughening
  of the anionic-rich Defence state.

## Implementation

The membrane is a 2D over-coordinated Fibonacci mesh (mean coordination
about 8) of mesoscopic patches connected by a phase-composite harmonic
bond network. Bond stiffnesses follow a linear constitutive map of gene
expression (K_Lo, K_Ld, K_int with k_bond = 0.5 K_local; the emergent mesh
modulus is K_mesh = 1.545 k_bond). Residual Lennard-Jones cohesion and
screened Debye-Hueckel electrostatics act between non-bonded pairs, with
bonded pairs excluded and their electrostatics restored through an
auxiliary bonded term. Inflation applies a radial per-particle force ramp
with dwell, tension is gamma = N f / (8 pi r), and failure is declared at
15 percent radial expansion. Simulations run on OpenMM with CUDA.

## Scripts

| Script | Purpose | Output |
| --- | --- | --- |
| `run_pdev_simulation_r20.py` | Production campaign, R = 20 nm (both states, both modes, pH 2.5/5.0/6.8, n = 50) | `pdev_unified_R20.csv` |
| `run_pdev_simulation_r40.py` | Production campaign, R = 40 nm | `pdev_unified_R40.csv` |
| `run_pdev_simulation_ladder.py` | Loading-rate ladder engine (rates 1x/4x/16x/64x, selectable missions, modes, pH, n) | `pdev_unified_R20_<rate>x*.csv` |
| `measure_mesh_modulus.py` | Bare-mesh area-expansion modulus calibration | `mesh_modulus_calibration.csv` |
| `make_figures_tables.py` | Regenerates every manuscript figure (PDF/PNG) and statistical table, including the rate-convergence fits | `../figures/`, `../tables/` |
| `Manuscript_Numbers.ipynb` | Recomputes every number quoted in the manuscript from the CSVs | printed values |

The engines write their CSV next to the working directory from which they
are launched. The figure/table generator and the notebook expect the
campaign CSVs one directory above this folder.

## Reproducing the results

From this folder, with the environment below active:

```bash
# mesh calibration (CPU)
python measure_mesh_modulus.py

# production campaigns (GPU; each writes its CSV here)
python run_pdev_simulation_r20.py
python run_pdev_simulation_r40.py

# loading-rate ladder (GPU); examples:
python run_pdev_simulation_ladder.py --rate 4
python run_pdev_simulation_ladder.py --rate 16
python run_pdev_simulation_ladder.py --rate 64 --missions STATE_A_RIPENING --modes SEEDED --replicates 15 --out pdev_unified_R20_64x_A.csv
python run_pdev_simulation_ladder.py --rate 64 --missions STATE_B_DEFENSE --modes SEEDED --replicates 10 --out pdev_unified_R20_64x_B.csv
python run_pdev_simulation_ladder.py --rate 64 --modes SPONTANEOUS --replicates 10 --out pdev_unified_R20_64x_spont.csv

# figures, tables and manuscript numbers
python make_figures_tables.py
jupyter notebook numbers.ipynb
```

A single NVIDIA GPU reproduces the full campaign in a few days; the two
production campaigns dominate the cost. Setting the optional `SEED`
constant in the engines makes runs exactly reproducible (both the
domain-assignment and Langevin random streams are then seeded per
replicate); by default each replicate is an independent stochastic
realisation, as used for the manuscript data.

## Setup

Hardware

- NVIDIA GPU with CUDA toolkit (OpenMM CUDA platform); about 4 GB free
  VRAM per concurrent worker is sufficient at N = 2,513 particles
- Any modern multi-core Linux host; the engines parallelise replicates
  across GPU workers automatically

Python environment

```bash
conda create -n openmm python=3.10 -y
conda activate openmm
conda install -c conda-forge openmm numpy scipy pandas matplotlib seaborn jupyter -y
```

If a user-site installation shadows the environment packages, prefix
commands with `PYTHONNOUSERSITE=1` (for example
`PYTHONNOUSERSITE=1 python make_figures_tables.py`).
