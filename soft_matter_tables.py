"""
Genomic regulation of chemo-mechanical stability in plant-derived
extracellular vesicles (PDEVs): a multiscale model of composite reinforcement.

soft_matter_tables.py
=====================

Computes and prints the statistical tables reported in the manuscript
(Statistical tables, Tables 1-4) from the Seeded-vs-Spontaneous simulation
output ``kev_seeded_vs_spontaneous.csv``:

    Table 1 (tab:rupture)     : Rupture tension (gamma_crit) statistics,
                                 n=50 per group; mean +/- SD, 95% CI.
    Table 2 (tab:percolation) : Percolation fraction (S_max) statistics.
    Table 3 (tab:morans)      : Moran's I spatial-autocorrelation statistics.
    Table 4 (tab:welch)       : Welch two-sample t-tests
                                 (Defence vs Ripening) by initialisation mode.

Notation
--------
Display headers use the manuscript symbols gamma_crit (critical rupture
tension, mN m^-1), S_max (rigid-domain percolation fraction) and Moran's I
(spatial heterogeneity). The underlying CSV column names are kept as-is
(Rupture_Tension_mNm, Percolation, Heterogeneity, Mission, Mode) for
backwards-compatibility; the mapping is:

    CSV column              Manuscript symbol / term
    ---------------------   ---------------------------------------
    Rupture_Tension_mNm     gamma_crit  (critical rupture tension)
    Percolation             S_max       (rigid-domain percolation fraction)
    Heterogeneity           Moran's I   (spatial heterogeneity index)
    Mission                 State (STATE_A_RIPENING / STATE_B_DEFENSE)
    Mode                    Init. (SEEDED / SPONTANEOUS)

Requires pandas, numpy and scipy.
"""
import pandas as pd
import numpy as np
from scipy import stats

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d for two groups (pooled-SD standardisation)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # Calculate the pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_sd

def get_summary_stats(series):
    """Return mean, SD, and 95% CI."""
    n = len(series)
    mean = np.mean(series)
    sd = np.std(series, ddof=1)
    se = sd / np.sqrt(n)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
    return mean, sd, ci

def generate_tables(csv_file='kev_seeded_vs_spontaneous.csv'):
    """Print Tables 1-4 (manuscript Statistical tables) to stdout.

    No statistics logic is altered relative to the original implementation:
    grouping, Welch's t-test (equal_var=False) and Cohen's d are unchanged.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run the simulation first.")
        return

    modes = ['SEEDED', 'SPONTANEOUS']
    # (CSV Mission key, manuscript table label)
    missions = [('STATE_A_RIPENING', 'A (Rip.)'), ('STATE_B_DEFENSE', 'B (Def.)')]

    # --- TABLE 1: Rupture Tension (gamma_crit) -- manuscript tab:rupture ---
    print("-" * 60)
    print("TABLE 1: Rupture tension statistics (gamma_crit)")
    print(f"{'Init.':<12} | {'State':<10} | {'Mean':<8} | {'SD':<6} | {'95% CI'}")
    print("-" * 60)
    for mode in modes:
        for mission, label in missions:
            data = df[(df['Mode'] == mode) & (df['Mission'] == mission)]['Rupture_Tension_mNm']
            mean, sd, ci = get_summary_stats(data)
            print(f"{mode.capitalize():<12} | {label:<10} | {mean:>8.1f} | {sd:>6.1f} | [{ci[0]:.1f}, {ci[1]:.1f}]")

    # --- TABLE 2: Percolation Fraction (S_max) -- manuscript tab:percolation ---
    print("\n" + "-" * 60)
    print("TABLE 2: Percolation fraction statistics (S_max)")
    print(f"{'Init.':<12} | {'State':<10} | {'Mean':<8} | {'SD':<6} | {'95% CI'}")
    print("-" * 60)
    for mode in modes:
        for mission, label in missions:
            data = df[(df['Mode'] == mode) & (df['Mission'] == mission)]['Percolation']
            mean, sd, ci = get_summary_stats(data)
            print(f"{mode.capitalize():<12} | {label:<10} | {mean:>8.3f} | {sd:>6.3f} | [{ci[0]:.3f}, {ci[1]:.3f}]")

    # --- TABLE 3: Moran's I (Heterogeneity) -- manuscript tab:morans ---
    print("\n" + "-" * 60)
    print("TABLE 3: Moran's I statistics")
    print(f"{'Init.':<12} | {'State':<10} | {'Mean':<8} | {'SD':<6} | {'95% CI'}")
    print("-" * 60)
    for mode in modes:
        for mission, label in missions:
            data = df[(df['Mode'] == mode) & (df['Mission'] == mission)]['Heterogeneity']
            mean, sd, ci = get_summary_stats(data)
            print(f"{mode.capitalize():<12} | {label:<10} | {mean:>8.3f} | {sd:>6.3f} | [{ci[0]:.3f}, {ci[1]:.3f}]")

    # --- TABLE 4: Welch Two-Sample t-tests -- manuscript tab:welch ---
    print("\n" + "-" * 60)
    print("TABLE 4: Welch two-sample t-tests (Defence vs Ripening)")
    print(f"{'Mode':<12} | {'Delta':<8} | {'t':<8} | {'p-value':<12} | {'Cohens d'}")
    print("-" * 60)
    for mode in modes:
        data_def = df[(df['Mode'] == mode) & (df['Mission'] == 'STATE_B_DEFENSE')]['Rupture_Tension_mNm']
        data_rip = df[(df['Mode'] == mode) & (df['Mission'] == 'STATE_A_RIPENING')]['Rupture_Tension_mNm']

        delta = np.mean(data_def) - np.mean(data_rip)
        t_stat, p_val = stats.ttest_ind(data_def, data_rip, equal_var=False) # Welch's t-test
        d = calculate_cohens_d(data_def, data_rip)

        print(f"{mode.capitalize():<12} | {delta:>8.1f} | {t_stat:>8.2f} | {p_val:>12.2e} | {d:.2f}")

if __name__ == "__main__":
    generate_tables()
