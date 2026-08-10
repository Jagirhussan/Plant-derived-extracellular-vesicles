# Genomic regulation of chemo-mechanical stability in plant-derived extracellular vesicles: a multiscale model of composite reinforcement

This repository contains the source code for the multiscale biophysical model
presented in the accompanying manuscript. It couples a parameterised genetic
state space (enzyme expression) to Supra-molecular Coarse-Grained Molecular
Dynamics (SCG-MD) to simulate how plant-derived extracellular vesicles (PDEVs)
withstand the harsh physicochemical barriers of the human gastrointestinal
tract (osmotic pressure and acidic pH).

## Scientific context

Plant-derived EVs are remarkably robust compared with mammalian exosomes, often
surviving gastric transit to deliver bioactive cargo. This code tests the
hypothesis that this stability arises from a **Defence state** — a genetically
regulated lipid profile that creates a phase-separated, composite-material
architecture in which discontinuous rigid (sterol-rich, liquid-ordered, $L_o$)
domains arrest crack propagation within a fluid (liquid-disordered, $L_d$)
matrix.

The model defines two primary genetic states:

1. **State A (Ripening, high-fluidity / homeostatic):** high $G_{FAD}$
   activity, polyunsaturated lipids ($S_{unsat} \approx 80\%$), low sterols.
   Biologically associated with fruit softening.
2. **State B (Defence, high-barrier / stress-adapted):** $G_{SMT}$ at the
   upper bound (4.0-fold), sterol accumulation, $L_o$ domain formation.
   Biologically associated with pathogen / abiotic-stress resistance.

An optional exploratory intermediate archetype (`STATE_C_STRESS`) is retained
in `kev_mechanics.py` for parameter-space coverage; it is **not** reported in
the manuscript.

## Notation glossary (code ↔ manuscript)

The CSV column names and internal string keys are kept stable for backwards
compatibility; the rendered figures, tables and this README use the manuscript
notation.

- **Code identifier** → **Manuscript symbol / term**
- `K_base`, `alpha_SMT`, `beta_FAD` → $K_{base}$, $\alpha$, $\beta$ (Eq. 1)
- `genes['SMT']` → $G_{SMT}$ (sterol methyltransferase)
- `genes['FAD']` → $G_{FAD}$ (fatty-acid desaturase)
- `genes['PLD']` → $G_{PLD}$ (phospholipase D; acts on $U_{dh}$ only)
- `Ka_Input_mNm` → $K_{eff}$ (network area-compressibility modulus, mN m$^{-1}$)
- `Rupture_Tension_mNm` → $\gamma_{crit}$ (critical rupture tension, mN m$^{-1}$)
- `Percolation` → $S_{max}$ (rigid-domain percolation fraction)
- `Heterogeneity` → Moran's $I$ (spatial-autocorrelation index)
- `Mode 'SEEDED'` → Seeded (Active sorting) initialisation
- `Mode 'SPONTANEOUS'` → Spontaneous (Entropic mixing) initialisation
- particle type `'Lo'` → liquid-ordered (sterol-rich) phase, $L_o$
- particle type `'Ld'` → liquid-disordered (fluid) phase, $L_d$

## Dependencies

These scripts require **OpenMM with CUDA support** to run the molecular
dynamics simulations.

- Python 3.8+
- OpenMM 7.0+ (with CUDA toolkit)
- NumPy
- SciPy
- Pandas
- Matplotlib (figures)
- Seaborn (figures)

## Usage & execution order

The scripts are designed to be run in a specific order. The simulation scripts
(`kev_*.py`) generate CSV data files, which are then consumed by the analysis
and plotting scripts.

### 1. `kev_seeded_vs_spontaneous.py`

**Goal:** determine whether the *topology* of the lipid shell matters, or only
the composition.

- **What it does:** runs parallel simulations comparing **Seeded**
  initialisation (rigid lipids pre-clustered via a cluster-growth / BFS
  algorithm, mimicking biological raft sorting) against **Spontaneous**
  initialisation (uniform random assignment, the randomised control), at
  identical lipid composition. Electrostatics are fixed at pH 6.8 so the
  comparison isolates topology. Also records the full tension–strain
  trajectories used for Fig. 4.
- **Output:** `kev_seeded_vs_spontaneous.csv`
- **Execution:**
  ```bash
  python kev_seeded_vs_spontaneous.py
  ```

> **Re-run note:** the Debye screening length $\lambda_D$ in this script is set
> to 1.0 nm to match the manuscript. It was previously 0.8 nm; any CSV
> committed before this change is stale and should be regenerated before
> producing figures/tables.

### 2. `kev_mechanics.py`

**Goal:** test the environmental (pH) stability of different genetic profiles.

- **What it does:** simulates vesicle rupture under osmotic inflation across pH
  levels (2.5, 6.8, 7.4). It calculates how the Henderson–Hasselbalch
  protonation of anionic lipids (phosphatidic acid, $pK_a \approx 4.0$) at
  gastric pH (2.5) affects membrane tension, quantifying the stress-stiffening
  (jamming) response of the Ripening state versus the mechanical homeostasis of
  the Defence state.
- **Output:** `kev_mechanics.csv`
- **Execution:**
  ```bash
  python kev_mechanics.py
  ```
  *(Note: this script uses GPU multiprocessing. Ensure `NUM_GPUS` matches your
  hardware.)*

### 3. `soft_matter_figures.py`

**Goal:** visualise the results.

- **What it does:** reads the two CSVs above and produces the four
  publication-ready figures.
- **Output:** `Fig1_Mechanics_Control.pdf`, `Fig2_Topology_Scatter.pdf`,
  `Fig3_pH_Stability.pdf`, `Fig4_Stress_Strain.pdf`.
- **Execution:**
  ```bash
  python soft_matter_figures.py
  ```

### 4. `soft_matter_tables.py`

**Goal:** compute the statistical tables reported in the manuscript.

- **What it does:** reads `kev_seeded_vs_spontaneous.csv` and prints the
  rupture-tension ($\gamma_{crit}$), percolation ($S_{max}$), Moran's $I$
  summary statistics (mean ± SD, 95% CI) and the Welch two-sample $t$-tests
  (Defence vs Ripening) by initialisation mode.
- **Execution:**
  ```bash
  python soft_matter_tables.py
  ```

---

## Key findings & results

After running the pipeline, the generated figures illustrate the following
biophysical conclusions.

### 1. The composite advantage (Figs 1 & 2)

The **Defence state (Seeded)** significantly outperforms the **Ripening state**.
Crucially, Seeded Defence vesicles are mechanically stronger than Spontaneous
(randomised) vesicles of the *exact same* lipid composition.

- *Conclusion:* stability is not only chemical; it is topological. The
  formation of discontinuous rigid domains (a composite material) arrests crack
  propagation, increasing the critical rupture tension $\gamma_{crit}$ by ~39%
  over the Ripening state, and by ~23% over the randomised control.

### 2. Mechanical homeostasis (Fig 3)

The simulation compares vesicles at pH 6.8 (neutral) vs pH 2.5 (gastric acid).

- **Ripening state:** exhibits stress-stiffening (jamming). The membrane
  stiffens by ~6.9% under acid shock due to charge neutralisation of anionic
  lipids and steric jamming of the neutralised continuum.
- **Defence state:** exhibits **mechanical homeostasis**. The rigid sterol-rich
  composite network buffers the membrane against electrostatic fluctuations,
  maintaining stable mechanics ($\Delta < 2.5\%$).

### 3. Toughness (Fig 4)

The stress–strain curves show that the Defence state possesses a much higher
yield point and total toughness (area under the tension–strain curve) than the
fluid Ripening state and the randomised control.

> **Interpretive caveat:** the absolute rupture tensions reported here (e.g. ~367 mN m^{-1}) 
> are theoretical *comparative indices* rather than
> absolute physiological values. The nanosecond loading rates inherent to MD
> yield much higher thresholds than quasi-static macroscopic experiments; the
> logarithmic scaling of $\gamma_c$ with loading rate shifts absolute values
> downward while preserving the relative structural hierarchy. See the
> manuscript Discussion for details.

## File structure

- `kev_seeded_vs_spontaneous.py`: topology-controlled simulation (Seeded vs
  Spontaneous).
- `kev_mechanics.py`: pH / gene-expression sweep simulation.
- `soft_matter_figures.py`: figure generation.
- `soft_matter_tables.py`: statistical table generation.
- `Manuscript_values.ipynb`: Recomputes **every quantitative value cited in the manuscript** directly from the simulation CSV outputs.
- `kev_*.csv`: data files (generated after running simulations).

## Authors & citation

This code supports the research by:

**Jagir R. Hussan, Anand Rampadarath, David P. Nickerson, Peter J. Hunter**

Auckland Bioengineering Institute, University of Auckland, New Zealand.

If you use this model, please cite the associated manuscript:

> Hussan, J.R., Rampadarath, A., Nickerson, D.P. and Hunter, P.J., 2026. 
>"Genomic regulation of chemo-mechanical stability in plant-derived extracellular vesicles: a multiscale model of composite reinforcement".
> bioRxiv, pp.2026-07.
