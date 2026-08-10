"""
Genomic regulation of chemo-mechanical stability in plant-derived
extracellular vesicles (PDEVs): a multiscale model of composite reinforcement.

soft_matter_figures.py
======================

Produces the four publication-ready figures of the manuscript from the CSV
outputs of the SCG-MD simulation scripts:
    
    Fig2_Stress_Strain.pdf      -> Fig. 2 (Comparative stress-strain trajectories)
    Fig3_Mechanics_Control.pdf  -> Fig. 3 (Mechanical hierarchy of PDEV states)
    Fig4_Topology_Scatter.pdf   -> Fig. 4 (Topological reinforcement; S_max)
    Fig5_pH_Stability.pdf       -> Fig. 5 (Environmental stability under acid shock)
    

Inputs
------
    kev_mechanics.csv               (from kev_mechanics.py; used for Fig. 5)
    kev_seeded_vs_spontaneous.csv   (from kev_seeded_vs_spontaneous.py;
                                     used for Figs. 2, 3, 4)

Notation
--------
Display labels follow the manuscript ("Defence" / "Randomised").
Axis labels use the manuscript symbols gamma_crit (critical rupture tension),
S_max (rigid-domain percolation fraction) and units mN m^-1. The underlying CSV
column names are kept as-is (Rupture_Tension_mNm, Percolation, Heterogeneity,
Mission, Mode) for backwards-compatibility; only the rendered strings change.

Requires matplotlib and seaborn.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import ast

def parse_numpy_list(s):
    """
    Parses a string representation of a list containing np.float64(...) elements.
    Example: '[np.float64(0.1), np.float64(0.2)]' -> [0.1, 0.2]
    """
    if pd.isna(s) or s == '[]' or s == '':
        return []
    try:
        # Evaluate the string using numpy in the context
        return eval(s, {'np': np})
    except Exception as e:
        print(f"Error parsing string: {s[:50]}... {e}")
        return []

def generate_all_figures():
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'grid.alpha': 0.3
    })

    # Load Data
    try:
        df_mechanics = pd.read_csv('kev_mechanics.csv')
        df_seededvsspontaneous = pd.read_csv('kev_seeded_vs_spontaneous.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: CSV files not found.")
        return

    # Colors
    c_rip = '#2ecc71' # Green (Ripening)
    c_def = '#e74c3c' # Red (Defence)
    c_rnd = '#95a5a6' # Grey (Control)
    c_acid = '#8e44ad' # Purple (Acid)

    # =========================================================================
    # FIG 3: Mechanics Control (Violin/Strip Plot) -- manuscript Fig. 3
    # Critical rupture tension (gamma_crit) for the three conditions compared
    # in "Mechanical hierarchy of states": Ripening (State A, Seeded),
    # Defence (State B) Seeded (Composite), and Defence (State B) Spontaneous
    # (Randomised control). N=50 per group.
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    
    # Filter pertinent data from control study
    subset = df_seededvsspontaneous[
        ((df_seededvsspontaneous['Mission'] == 'STATE_A_RIPENING') & (df_seededvsspontaneous['Mode'] == 'SEEDED')) |
        ((df_seededvsspontaneous['Mission'] == 'STATE_B_DEFENSE') & (df_seededvsspontaneous['Mode'] == 'SEEDED')) |
        ((df_seededvsspontaneous['Mission'] == 'STATE_B_DEFENSE') & (df_seededvsspontaneous['Mode'] == 'SPONTANEOUS'))
    ].copy()

    # Label mapping (display strings use manuscript spelling: Defence / Randomised)
    subset['Condition'] = subset.apply(lambda x: 
        'Ripening\n(Fluid)' if x['Mission'] == 'STATE_A_RIPENING' else 
        ('Defence\n(Composite)' if x['Mode'] == 'SEEDED' else 'Defence\n(Randomised)'), axis=1)

    order = ['Ripening\n(Fluid)', 'Defence\n(Randomised)', 'Defence\n(Composite)']
    pal = [c_rip, c_def, c_rnd]

    sns.violinplot(data=subset, x='Condition', y='Rupture_Tension_mNm', order=order, palette=pal, hue='Condition',
                   inner=None, alpha=0.3, linewidth=0, ax=ax1)
    sns.stripplot(data=subset, x='Condition', y='Rupture_Tension_mNm', order=order, palette=pal, hue='Condition',
                  size=4, jitter=0.2, alpha=0.7, ax=ax1)

    # Statistical bars
    y_max = subset['Rupture_Tension_mNm'].max()
    
    # A vs B Seeded
    x1, x2 = 0, 2
    y, h = y_max + 15, 5
    ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax1.text((x1+x2)/2, y+h+2, "p < 10$^{-60}$", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # B Spont vs B Seeded
    x1, x2 = 1, 2
    y, h = y_max - 50, 5
    ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax1.text((x1+x2)/2, y+h+2, "Topology Effect\n(+22%)", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Critical Rupture Tension, $\\gamma_{crit}$ (mN m$^{-1}$)', fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylim(200, 400)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig('Fig3_Mechanics_Control.pdf',dpi=300)
    print("Generated Fig3")


    # ----------------------------
    # Create 3-level grouping variable
    # ----------------------------

    # Adjust this threshold if needed
    SPONT_THRESHOLD = 0.45

    df = df_seededvsspontaneous.copy()

    df['Case'] = 'Unknown'

    # Ripening (seeded)
    df.loc[
        df['Mission'] == 'STATE_A_RIPENING',
        'Case'
    ] = 'Ripening (Seeded)'

    # Defence (seeded)
    df.loc[
        (df['Mission'] == 'STATE_B_DEFENSE') &
        (df['Mode'] == 'SEEDED'),
        'Case'
    ] = 'Defence (Seeded)'

    # Defence (spontaneous) – cluster 1
    df.loc[
        (df['Mission'] == 'STATE_B_DEFENSE') &
        (df['Mode'] == 'SPONTANEOUS') &
        (df['Percolation'] < SPONT_THRESHOLD),
        'Case'
    ] = 'Defence (Randomised) – Cluster 1'

    # Defence (spontaneous) – cluster 2
    df.loc[
        (df['Mission'] == 'STATE_B_DEFENSE') &
        (df['Mode'] == 'SPONTANEOUS') &
        (df['Percolation'] >= SPONT_THRESHOLD),
        'Case'
    ] = 'Defence (Randomised) – Cluster 2'

    # --------------------------------------------------
    # 2. COLOR PALETTE
    # --------------------------------------------------
    palette_case = {
        'Ripening (Seeded)': c_rip,
        'Defence (Seeded)': c_def,
        'Defence (Randomised) – Cluster 1': c_rnd,
        'Defence (Randomised) – Cluster 2': c_rnd
    }

    # --------------------------------------------------
    # 3. JOINT GRID
    # --------------------------------------------------
    g = sns.JointGrid(
        data=df,
        x='Percolation',
        y='Rupture_Tension_mNm',
        height=6
    )

    # --------------------------------------------------
    # 4. JOINT SCATTER (SEEDED)
    # --------------------------------------------------
    seeded = df[df['Mode'] == 'SEEDED']

    sns.scatterplot(
        data=seeded,
        x='Percolation',
        y='Rupture_Tension_mNm',
        hue='Mission',
        palette={
            'STATE_A_RIPENING': c_rip,
            'STATE_B_DEFENSE': c_def
        },
        s=80,
        alpha=0.6,
        edgecolor='k',
        legend=False,
        ax=g.ax_joint
    )

    # --------------------------------------------------
    # 5. JOINT SCATTER (SPONTANEOUS – X markers)
    # --------------------------------------------------
    spont = df[df['Mode'] == 'SPONTANEOUS']

    g.ax_joint.scatter(
        spont['Percolation'],
        spont['Rupture_Tension_mNm'],
        color=c_rnd,
        marker='X',
        s=80,
        alpha=0.6,
        edgecolor='k'
    )

    # --------------------------------------------------
    # 6. MARGINAL KDEs (NOW 4 DISTINCT DENSITIES)
    # --------------------------------------------------
    sns.kdeplot(
        data=df,
        x='Percolation',
        hue='Case',
        palette=palette_case,
        fill=True,
        alpha=0.3,
        bw_adjust=0.6,
        legend=False,
        ax=g.ax_marg_x
    )

    sns.kdeplot(
        data=df,
        y='Rupture_Tension_mNm',
        hue='Case',
        palette=palette_case,
        fill=True,
        alpha=0.3,
        bw_adjust=0.6,
        legend=False,
        ax=g.ax_marg_y
    )

    # --------------------------------------------------
    # 7. REFERENCE LINES & TEXT
    # --------------------------------------------------
    g.ax_joint.axvline(0.5, color='k', linestyle='--', alpha=0.5)

    g.ax_joint.text(
        0.52, 240, 'Monolithic\nLimit',
        fontsize=9
    )

    g.ax_joint.text(
        0.48, 240, 'Composite\nRange',
        fontsize=9, ha='right'
    )

    # --------------------------------------------------
    # 8. AXIS LABELS & LIMITS
    # --------------------------------------------------
    g.set_axis_labels(
        'Rigid Domain Percolation ($S_{max}$)',
        'Critical Rupture Tension, $\\gamma_{crit}$ (mN m$^{-1}$)',
        fontweight='bold'
    )

    g.ax_joint.set_ylim(200, 420)
    g.ax_marg_y.set_ylim(200, 420)

    # --------------------------------------------------
    # 9. MANUAL LEGEND
    # --------------------------------------------------
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor=c_rip, markersize=10,
            label='Ripening (Seeded)'),

        Line2D([0], [0], marker='o', color='w',
            markerfacecolor=c_def, markersize=10,
            label='Defence (Seeded)'),

        Line2D([0], [0], marker='X', color='w',
            markerfacecolor=c_rnd, markersize=10,
            label='Defence (Randomised)')
    ]

    g.ax_joint.legend(
        handles=legend_elements,
        loc='upper left',
        frameon=False
    )



    plt.savefig('Fig4_Topology_Scatter.pdf',dpi=300)
    print("Generated Fig4")

    # =========================================================================
    # FIG 5: pH Stability (Bar) -- manuscript Fig. 5
    # Environmental stability under acid shock (pH 6.8 vs 2.5). The Ripening
    # state (A) stress-stiffens (jamming) under acid shock; the Defence state
    # (B) exhibits mechanical homeostasis (Delta < 2.5%).
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    subset_ph = df_mechanics[df_mechanics['pH'].isin([2.5, 6.8])]
    
    means = subset_ph.groupby(['Mission', 'pH'])['Rupture_Tension_mNm'].mean().reset_index()
    
    sns.barplot(data=subset_ph, x='Mission', y='Rupture_Tension_mNm', hue='pH',
                order=['STATE_A_RIPENING', 'STATE_B_DEFENSE'],
                palette={2.5: c_acid, 6.8: '#bdc3c7'},
                edgecolor='k', capsize=0.1, errorbar=('ci', 95), ax=ax3)
    
    # Annotations
    h_a_68 = means[(means['Mission']=='STATE_A_RIPENING') & (means['pH']==6.8)]['Rupture_Tension_mNm'].values[0]
    h_a_25 = means[(means['Mission']=='STATE_A_RIPENING') & (means['pH']==2.5)]['Rupture_Tension_mNm'].values[0]
    
    ax3.annotate(f'+{((h_a_25-h_a_68)/h_a_68)*100:.1f}% (Jamming)', 
                 xy=(-0.2, h_a_25), xytext=(-0.2, h_a_25+40),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')

    h_b_25 = means[(means['Mission']=='STATE_B_DEFENSE') & (means['pH']==2.5)]['Rupture_Tension_mNm'].values[0]
    ax3.text(0.8, h_b_25+15, r"Homeostasis\n($\Delta = 1.3\%$)", ha='center', color=c_def, fontweight='bold')

    ax3.set_ylabel('Critical Rupture Tension, $\\gamma_{crit}$ (mN m$^{-1}$)', fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_xticklabels(['Ripening (A)', 'Defence (B)'])
    ax3.legend(title='pH', loc='upper left')
    ax3.set_ylim(0, 500)
    sns.despine()
    plt.tight_layout()
    plt.savefig('Fig5_pH_Stability.pdf',dpi=300)
    print("Generated Fig5")

    # =========================================================================
    # FIG 2: Stress-Strain (Curves) -- manuscript Fig. 2
    # Comparative stress-strain trajectories. State B (Defence, Composite /
    # Seeded) exhibits a higher yield point and greater toughness (area under
    # the curve) than the fluid Ripening state (A) and the randomised control.
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    
    def plot_curve_group(mission, mode, color, label, style='-'):
        subset = df_seededvsspontaneous[(df_seededvsspontaneous['Mission'] == mission) & (df_seededvsspontaneous['Mode'] == mode)]
        
        all_strains = []
        all_tensions = []
        
        for _, row in subset.iterrows():
            s = parse_numpy_list(row['Strain_Curve'])
            t = parse_numpy_list(row['Tension_Curve'])
            
            if len(s) > 10:
                all_strains.append(s)
                all_tensions.append(t)
        
        if not all_strains:
            print(f"No curves for {label}")
            return

        # Interpolate to average
        max_s = max(max(s) for s in all_strains)
        common_s = np.linspace(0, max_s, 200)
        interp_ts = []
        for s, t in zip(all_strains, all_tensions):
            # clean duplicates in s for numpy interp
            # simplistic approach: assume monotonic increasing
            interp_ts.append(np.interp(common_s, s, t))
        
        mean_t = np.mean(interp_ts, axis=0)
        std_t = np.std(interp_ts, axis=0)
        
        ax4.plot(common_s*100, mean_t, color=color, linestyle=style, label=label, linewidth=2)
        ax4.fill_between(common_s*100, mean_t-std_t, mean_t+std_t, color=color, alpha=0.1)
        
        # Mark Yield
        ax4.scatter(common_s[-1]*100, mean_t[-1], color=color, s=40, zorder=5)

    plot_curve_group('STATE_A_RIPENING', 'SEEDED', c_rip, 'Ripening (Fluid)')
    plot_curve_group('STATE_B_DEFENSE', 'SPONTANEOUS', c_rnd, 'Defence (Randomised)', '--')
    plot_curve_group('STATE_B_DEFENSE', 'SEEDED', c_def, 'Defence (Composite)')

    ax4.set_xlabel('Areal Strain (%)', fontweight='bold')
    ax4.set_ylabel('Membrane Tension, $\\gamma$ (mN m$^{-1}$)', fontweight='bold')
    ax4.legend(loc='lower right', frameon=True)
    
    # Annotate Toughness
    ax4.annotate(r"Toughness $\sim \int \gamma d\epsilon$", xy=(15, 100), xytext=(5, 300),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', fontsize=10)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('Fig2_Stress_Strain.pdf',dpi=300)
    print("Generated Fig2")

if __name__ == "__main__":
    generate_all_figures()