# Chemo-Mechanical Model of Plant-Derived Extracellular Vesicles (PDEVs)

This repository contains the source code for a multiscale biophysical model investigating the mechanical stability of plant-derived extracellular vesicles. 

By coupling a parameterised genetic state space (enzyme expression) with Supra-molecular Coarse-Grained Molecular Dynamics (SCG-MD), this project simulates how vesicles withstand the harsh physicochemical barriers of the human gastrointestinal tract (e.g., osmotic pressure and acidic pH).

## Scientific Context

Plant-derived EVs are remarkably robust compared to mammalian exosomes, often surviving gastric transit to deliver bioactive cargo. This code tests the hypothesis that this stability arises from a **"Defense State"**—a genetically regulated lipid profile that creates a phase-separated, composite material architecture.

The model defines two primary genetic states:
1.  **State A (Ripening):** High fluidity, low sterols. Biologically associated with fruit softening.
2.  **State B (Defense):** High sterol content, forming rigid domains. Biologically associated with pathogen resistance.

## Dependencies

To run these simulations, you need a Python environment with the following packages installed. **Note:** These scripts require **OpenMM with CUDA support** to run the molecular dynamics simulations.

* **Python 3.8+**
* **OpenMM 7.0+** (with CUDA toolkit)
* **NumPy**
* **SciPy**
* **Pandas**
* **Matplotlib** (for plotting)
* **Seaborn** (for plotting)

## Usage & Execution Order

The scripts are designed to be run in a specific order. The simulation scripts (`kev_*.py`) generate CSV data files, which are then consumed by the plotting script.

### 1. `kev_seeded_vs_spontaneous.py`
**Goal:** Determine if the *topology* of the lipid shell matters, or just the composition.
* **What it does:** Runs parallel simulations comparing "Seeded" initialization (where rigid lipids are pre-clustered, mimicking biological rafts) against "Spontaneous" initialization (random distribution).
* **Output:** `kev_seeded_vs_spontaneous.csv`
* **Execution:**
    ```bash
    python kev_seeded_vs_spontaneous.py
    ```

### 2. `kev_mechanics.py`
**Goal:** Test the environmental stability of different genetic profiles.
* **What it does:** Simulates vesicle rupture under osmotic inflation across different pH levels (2.5, 6.8, 7.4). It calculates how the protonation of anionic lipids at gastric pH (2.5) affects membrane tension.
* **Output:** `kev_mechanics.csv`
* **Execution:**
    ```bash
    python kev_mechanics.py
    ```
    *(Note: This script utilizes multiprocessing on GPUs. Ensure your `NUM_GPUS` variable in the script matches your hardware).*

### 3. `soft_matter_figures.py`
**Goal:** Visualize the results.
* **What it does:** Reads the two CSV files generated above and produces publication-ready figures.
* **Output:** `Fig1_Mechanics_Control.pdf`, `Fig2_Topology_Scatter.pdf`, `Fig3_pH_Stability.pdf`, `Fig4_Stress_Strain.pdf`.
* **Execution:**
    ```bash
    python soft_matter_figures.py
    ```

---

##  Key Findings & Results

After running the pipeline, the generated figures will illustrate the following biophysical conclusions:

### 1. The "Composite" Advantage (Fig 1 & 2)
You will observe that the **Defense State (Seeded)** significantly outperforms the **Ripening State**. Crucially, the "Seeded" Defense vesicles are mechanically stronger than "Spontaneous" vesicles of the exact same lipid composition.
* *Conclusion:* Stability is not just chemical; it is topological. The formation of discontinuous rigid domains (a composite material) arrests crack propagation, increasing rupture tension by ~39%.

### 2. Mechanical Homeostasis (Fig 3)
The simulation compares vesicles at pH 6.8 (neutral) vs. pH 2.5 (gastric acid).
* **Ripening State:** Exhibits "Stress Stiffening" (Jamming). The membrane stiffens by ~9% due to charge neutralization of anionic lipids.
* **Defense State:** Exhibits **Homeostasis**. The rigid sterol network buffers the membrane against electrostatic fluctuations, maintaining stable mechanics ($\Delta < 2.5\%$).

### 3. Toughness (Fig 4)
The stress-strain curves show that the Defense state possesses a much higher yield point and total toughness (area under the curve) compared to the fluid Ripening state.

##  File Structure

* `kev_seeded_vs_spontaneous.py`: Simulation logic for topology comparison.
* `kev_mechanics.py`: Simulation logic for pH and gene-expression sweeping.
* `soft_matter_figures.py`: Plotting logic.
* `kev_*.csv`: Data files (generated after running simulations).

##  Authors & Citation

This code supports the research by:
**Jagir R. Hussan, Maryam Alavi, David A. Nickerson, Anand Rampadarath, and Peter J. Hunter**

If you use this model, please cite the associated manuscript:
> "Genomic Regulation of Chemo-Mechanical Stability in Plant-Derived Extracellular Vesicles: A Multiscale Model of Composite Reinforcement" (2026).