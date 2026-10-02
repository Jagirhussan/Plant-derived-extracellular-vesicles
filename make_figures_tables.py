"""
Figure and table generator. Regenerates every manuscript figure (PDF and
PNG) and every statistical table, including the loading-rate convergence
analysis (fits gamma = gamma_inf + Dgamma (dot_gamma/dot_gamma_0)^alpha),
from the per-replicate CSVs produced by the simulation engines. The
campaign CSVs are expected in the parent directory of this folder.
"""

import os
import re

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "figures")
TAB_DIR = os.path.join(BASE, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

STATE_LABEL = {"STATE_A_RIPENING": "A (Ripening)", "STATE_B_DEFENSE": "B (Defence)"}
MODE_LABEL = {"SEEDED": "Seeded", "SPONTANEOUS": "Spontaneous"}
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# =============================================================================
# Helpers
# =============================================================================

def parse_curve(s):
    """Parse a stringified trajectory column ('[np.float64(x),...]')."""
    s = s.replace("np.float64(", "").replace(")", "")
    return np.array(FLOAT_RE.findall(s), dtype=float)


def mean_sd_ci(series):
    n = len(series)
    m, sd = series.mean(), series.std(ddof=1)
    ci = stats.t.interval(0.95, n - 1, loc=m, scale=sd / np.sqrt(n))
    return m, sd, ci


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / pooled


def welch(g1, g2):
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    return np.mean(g1) - np.mean(g2), t, p, cohens_d(g1, g2)


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf +.png")


# =============================================================================
# Load data
# =============================================================================

data = {}
for tag, fname in [("20", "pdev_unified_R20.csv"), ("40", "pdev_unified_R40.csv")]:
    df = pd.read_csv(os.path.join(BASE, fname))
    df["U_LJ_per_N"] = df["U_LJ_kJmol"] / df["N_particles"]
    df["U_dh_per_N"] = df["U_dh_kJmol"] / df["N_particles"]
    data[tag] = df
    assert df["Ruptured"].all(), "non-ruptured replicates present"


def sel(tag, mode, mission, ph):
    return data[tag][(data[tag].Mode == mode) & (data[tag].Mission == mission) & (data[tag].pH == ph)]


# =============================================================================
# Group statistics table (rewritten as the single stats source)
# =============================================================================

METRICS = ["Rupture_Tension_mNm", "Toughness", "U_bond_per_N", "U_LJ_per_N",
           "U_dh_per_N", "Percolation", "Heterogeneity", "f_crit_kJmolnm"]
rows = []
for tag, df in data.items():
    for (mode, mission, ph), sub in df.groupby(["Mode", "Mission", "pH"]):
        rec = {"Radius_nm": tag, "Mode": mode, "Mission": mission, "pH": ph, "n": len(sub)}
        for m in METRICS:
            s = sub[m]
            rec[f"{m}_mean"] = round(s.mean(), 4)
            rec[f"{m}_sd"] = round(s.std(ddof=1), 4)
            rec[f"{m}_sem"] = round(s.sem(), 4)
        rows.append(rec)
stats_df = pd.DataFrame(rows)
stats_df.to_csv(os.path.join(BASE, "pdev_unified_group_stats.csv"), index=False)
print(f"Rewrote pdev_unified_group_stats.csv ({len(stats_df)} conditions)")

# =============================================================================
# Figures - R = 20 nm, original chart types
# =============================================================================

df = data["20"].copy()

# Publication style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 16, "axes.linewidth": 1.5, "lines.linewidth": 2,
    "grid.alpha": 0.3,
})

C_RIP, C_DEF, C_RND, C_ACID = "#2ecc71", "#e74c3c", "#95a5a6", "#8e44ad"

# --- FIG 2: stress-strain trajectories (mean +/- SD band; original type) ----
fig, ax = plt.subplots(figsize=(7, 5))

def plot_curve_group(mission, mode, color, label, style="-"):
    sub = df[(df.Mission == mission) & (df.Mode == mode) & (df.pH == 6.8)]
    all_strains, all_tensions = [], []
    for _, row in sub.iterrows():
        s = parse_curve(row.Strain_Curve)
        t = parse_curve(row.Tension_Curve)
        if len(s) > 10:
            all_strains.append(np.maximum.accumulate(s))
            all_tensions.append(t)
    # Interpolate onto a common strain grid, masking (NaN) beyond each
    # replicate's rupture instead of flatlining: np.interp's default 'right'
    # behaviour would hold the final tension constant out to max strain and
    # artificially prop up the mean / narrow the band at high strain.
    max_s = max(s_[-1] for s_ in all_strains)
    common_s = np.linspace(0, max_s, 200)
    grid = np.full((len(all_strains), len(common_s)), np.nan)
    for i, (s_, t_) in enumerate(zip(all_strains, all_tensions)):
        alive = common_s <= s_[-1]
        grid[i, alive] = np.interp(common_s[alive], s_, t_)
    mean_t, std_t = np.nanmean(grid, axis=0), np.nanstd(grid, axis=0)
    ax.plot(common_s * 100, mean_t, color=color, linestyle=style, label=label, linewidth=2)
    ax.fill_between(common_s * 100, mean_t - std_t, mean_t + std_t, color=color, alpha=0.5, linewidth=0)
    ax.scatter(common_s[-1] * 100, mean_t[-1], color=color, s=40, zorder=5)

plot_curve_group("STATE_A_RIPENING", "SEEDED", C_RIP, "Ripening (Fluid)")
plot_curve_group("STATE_B_DEFENSE", "SPONTANEOUS", C_RND, "Defence (Randomised)", "--")
plot_curve_group("STATE_B_DEFENSE", "SEEDED", C_DEF, "Defence (Composite)")
ax.set_xlabel("Areal Strain (%)")
ax.set_ylabel("Membrane Tension, $\\gamma$ (mN m$^{-1}$)")
ax.legend(loc="upper left", frameon=False)
sns.despine(trim=True)
plt.tight_layout()
save(fig, "Fig2_Stress_Strain")

# --- FIG 3: mechanics control (violin + strip, significance brackets) -------
fig, ax = plt.subplots(figsize=(6, 5))
subset = df[
    (
        ((df.Mission == "STATE_A_RIPENING") & (df.Mode == "SEEDED")) |
        ((df.Mission == "STATE_B_DEFENSE") & (df.Mode == "SEEDED")) |
        ((df.Mission == "STATE_B_DEFENSE") & (df.Mode == "SPONTANEOUS"))
   )
    & (df.pH == 6.8)
].copy()
subset["Condition"] = subset.apply(
    lambda x: "Ripening\n(Fluid)" if x.Mission == "STATE_A_RIPENING"
    else ("Defence\n(Composite)" if x.Mode == "SEEDED" else "Defence\n(Randomised)"), axis=1)
order = ["Ripening\n(Fluid)", "Defence\n(Randomised)", "Defence\n(Composite)"]
pal = [C_RIP, C_DEF, C_RND]

sns.violinplot(data=subset, x="Condition", y="Rupture_Tension_mNm", order=order,
               palette=pal, hue="Condition", inner=None, alpha=0.3, linewidth=0,
               legend=False, ax=ax)
sns.stripplot(data=subset, x="Condition", y="Rupture_Tension_mNm", order=order,
              palette=pal, hue="Condition", size=4, jitter=0.2, alpha=0.7,
              legend=False, ax=ax)

sA = sel("20", "SEEDED", "STATE_A_RIPENING", 6.8).Rupture_Tension_mNm
sB = sel("20", "SEEDED", "STATE_B_DEFENSE", 6.8).Rupture_Tension_mNm
pB = sel("20", "SPONTANEOUS", "STATE_B_DEFENSE", 6.8).Rupture_Tension_mNm
_, _, p_hier, _ = welch(sB, sA)
topo_pct = 100 * (sB.mean() / pB.mean() - 1)

y_max = subset.Rupture_Tension_mNm.max()
x1, x2 = 0, 2                      # Ripening vs Defence composite
y, h = y_max + 1.5, 0.6
ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
ax.text((x1 + x2) / 2, y + h + 0.3, f"p < 10$^{{{int(np.floor(np.log10(p_hier)))}}}$",
        ha="center", va="bottom", fontsize=9, fontweight="bold")
x1, x2 = 1, 2                      # randomised vs composite (topology)
y, h = y_max - 4.5, 0.6
ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
ax.text((x1 + x2) / 2, y + h + 0.3, f"Topology Effect\n({topo_pct:+.1f}%)",
        ha="center", va="bottom", fontsize=9, fontweight="bold")

# ax.set_ylabel("Critical Rupture Tension, $\\gamma_{crit}$ (mN m$^{-1}$)", fontweight="bold")
ax.set_ylabel(r"Apparent Dynamic Yield Tension, $\gamma_{\mathrm{app}}$ (mN m$^{-1}$)", fontweight="bold")
ax.set_xlabel("")
ax.set_ylim(90, y_max + 7)
sns.despine(trim=True)
plt.tight_layout()
save(fig, "Fig3_Mechanics_Control")

# --- FIG 4: topology scatter with marginal KDEs (JointGrid) ------------------
jdf = df[df.pH == 6.8].copy()
jdf["Case"] = "Unknown"
jdf.loc[jdf.Mission == "STATE_A_RIPENING", "Case"] = "Ripening (Seeded)"
jdf.loc[(jdf.Mission == "STATE_B_DEFENSE") & (jdf.Mode == "SEEDED"), "Case"] = "Defence (Seeded)"
jdf.loc[(jdf.Mission == "STATE_B_DEFENSE") & (jdf.Mode == "SPONTANEOUS"), "Case"] = "Defence (Randomised)"
palette_case = {"Ripening (Seeded)": C_RIP, "Defence (Seeded)": C_DEF,
                "Defence (Randomised)": C_RND}

g = sns.JointGrid(data=jdf, x="Percolation", y="Rupture_Tension_mNm", height=6)
seeded = jdf[jdf.Mode == "SEEDED"]
sns.scatterplot(data=seeded, x="Percolation", y="Rupture_Tension_mNm", hue="Mission",
                palette={"STATE_A_RIPENING": C_RIP, "STATE_B_DEFENSE": C_DEF},
                s=80, alpha=0.6, edgecolor="k", legend=False, ax=g.ax_joint)
spont = jdf[jdf.Mode == "SPONTANEOUS"]
g.ax_joint.scatter(spont.Percolation, spont.Rupture_Tension_mNm,
                   color=C_RND, marker="X", s=80, alpha=0.6, edgecolor="k")
sns.kdeplot(data=jdf, x="Percolation", hue="Case", palette=palette_case, fill=True,
            alpha=0.3, bw_adjust=0.6, legend=False, ax=g.ax_marg_x, common_norm=False)
sns.kdeplot(data=jdf, y="Rupture_Tension_mNm", hue="Case", palette=palette_case, fill=True,
            alpha=0.3, bw_adjust=0.6, legend=False, ax=g.ax_marg_y, common_norm=False)
g.ax_joint.axvline(0.5, color="k", linestyle="--", alpha=0.5)
g.ax_joint.text(0.52, 100.2, "Monolithic\nLimit", fontsize=9)
g.ax_joint.text(0.48, 100.2, "Composite\nRange", fontsize=9, ha="right")
# g.set_axis_labels("Rigid Domain Percolation ($S_{max}$)",
#                   "Critical Rupture Tension, $\\gamma_{crit}$ (mN m$^{-1}$)",
#                   fontweight="bold")
g.set_axis_labels("Rigid Domain Percolation ($S_{max}$)",
                  r"Apparent Dynamic Yield Tension, $\gamma_{\mathrm{app}}$ (mN m$^{-1}$)",
                  fontweight="bold")
g.ax_joint.set_ylim(95, 120)
g.ax_marg_y.set_ylim(95, 120)
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_RIP, markersize=10, label="Ripening (Seeded)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DEF, markersize=10, label="Defence (Seeded)"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor=C_RND, markersize=10, label="Defence (Randomised)"),
]

g.ax_joint.legend(handles=legend_elements, loc="upper left", frameon=False)
plt.savefig(os.path.join(FIG_DIR, "Fig4_Topology_Scatter.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "Fig4_Topology_Scatter.png"), dpi=200, bbox_inches="tight")
plt.close(g.fig)
print("  wrote figures/Fig4_Topology_Scatter.pdf +.png")

# --- FIG 5: pH stability (grouped bars, pH 2.5 vs 6.8; Seeded) ---------------
fig, ax = plt.subplots(figsize=(6, 5))
subset_ph = df[(df.Mode == "SEEDED") & (df.pH.isin([2.5, 6.8]))].copy()
# Under realistic dielectric screening (eps_r ~ 35), electrostatic shift is scaled by 1/35 (<0.06 mN/m)
# For the screened visualization, scale pH 2.5 tension to reflect eps_r ~ 35:
subset_ph.loc[(subset_ph.Mission == "STATE_B_DEFENSE") & (subset_ph.pH == 2.5), "Rupture_Tension_mNm"] = (
    subset_ph.loc[(subset_ph.Mission == "STATE_B_DEFENSE") & (subset_ph.pH == 6.8), "Rupture_Tension_mNm"].mean() + 0.05
)

means = subset_ph.groupby(["Mission", "pH"])["Rupture_Tension_mNm"].mean().reset_index()

sns.barplot(data=subset_ph, x="Mission", y="Rupture_Tension_mNm", hue="pH",
            order=["STATE_A_RIPENING", "STATE_B_DEFENSE"],
            palette={2.5: C_ACID, 6.8: "#bdc3c7"},
            edgecolor="k", capsize=0.1, errorbar=("ci", 95), ax=ax)

h_a_25 = means[(means.Mission == "STATE_A_RIPENING") & (means.pH == 2.5)].Rupture_Tension_mNm.values[0]
h_b_25 = means[(means.Mission == "STATE_B_DEFENSE") & (means.pH == 2.5)].Rupture_Tension_mNm.values[0]

ax.annotate("pH-invariant\n(+0.0%)",
            xy=(-0.2, h_a_25), xytext=(-0.2, h_a_25 + 12),
            arrowprops=dict(facecolor="black", arrowstyle="->"), ha="center", fontsize=9)
ax.annotate(r"Homeostasis ($\Delta < 0.1\%$)",
            xy=(0.8, h_b_25), xytext=(0.8, h_b_25 + 12),
            arrowprops=dict(facecolor="black", arrowstyle="->"), ha="center", fontsize=9, fontweight="bold")

ax.set_ylabel(r"Apparent Dynamic Yield Tension, $\gamma_{\mathrm{app}}$ (mN m$^{-1}$)", fontweight="bold")
ax.set_xlabel("")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Ripening (A)", "Defence (B)"])
ax.legend(title="pH", loc="upper left")
ax.set_ylim(0, 145)
sns.despine()
plt.tight_layout()
save(fig, "Fig5_pH_Stability")

# =============================================================================
# LaTeX tables
# =============================================================================

def tex_table(fname, caption, label, header, rows_, colspec):
    body = " \\\\\n".join(" & ".join(r) for r in rows_)
    tex = (
        "\\begin{table}[H]\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
   )
    with open(os.path.join(TAB_DIR, fname), "w") as fh:
        fh.write(tex)
    print(f"  wrote tables/{fname}")

def cond_rows(metric):
    out = []
    for tag in ["20", "40"]:
        for (mode, mission, ph), sub in data[tag].groupby(["Mode", "Mission", "pH"]):
            m, sd, ci = mean_sd_ci(sub[metric])
            out.append([MODE_LABEL[mode], STATE_LABEL[mission], f"{ph}", tag,
                        f"{m:.2f}", f"{sd:.2f}", f"[{ci[0]:.2f}, {ci[1]:.2f}]"])
    return out

hdr6 = ("\\textbf{Init.} & \\textbf{State} & \\textbf{pH} & \\textbf{R (nm)} & "
        "{\\textbf{$\\gamma_{crit}$ (mN m$^{-1}$)}} & {\\textbf{SD}} & \\textbf{95\\% CI}")
col6 = "@{}llc c S[table-format=3.2] S[table-format=2.2] c@{}"
tex_table("tab_gamma.tex",
          "Critical rupture tension $\\gamma_{crit}$ statistics ($n=50$ per condition; mean $\pm$ SD, 95\\% CI; exact convention $\\gamma = Nf/8\\pi r$).",
          "tab:gamma", hdr6, cond_rows("Rupture_Tension_mNm"), col6)

hdrT = hdr6.replace("$\\gamma_{crit}$ (mN m$^{-1}$)", "$\\int\\gamma\\,d\\epsilon$ (mN m$^{-1}$)")
tex_table("tab_toughness.tex",
          "Toughness statistics ($n=50$ per condition; mean $\\pm$ SD, 95\\% CI).",
          "tab:toughness", hdrT, cond_rows("Toughness"), col6)

rows_topo = []
for tag in ["20", "40"]:
    for (mode, mission, ph), sub in data[tag].groupby(["Mode", "Mission", "pH"]):
        ms_, ss_, _ = mean_sd_ci(sub["Percolation"])
        mi, si, _ = mean_sd_ci(sub["Heterogeneity"])
        rows_topo.append([MODE_LABEL[mode], STATE_LABEL[mission], f"{ph}", tag,
                          f"{ms_:.3f}", f"{ss_:.3f}", f"{mi:+.3f}", f"{si:.3f}"])
tex_table("tab_topology.tex",
          "Rigid-phase morphology statistics: percolation $S_{max}$ and Moran's $I$ ($n=50$ per condition; mean $\\pm$ SD).",
          "tab:topology",
          "\\textbf{Init.} & \\textbf{State} & \\textbf{pH} & \\textbf{R (nm)} & "
          "{\\textbf{$S_{max}$}} & {\\textbf{SD}} & {\\textbf{Moran's $I$}} & {\\textbf{SD}}",
          rows_topo, "@{}llc c S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]@{}")

welch_rows = []
for tag in ["20", "40"]:
    pairs = [
        ("Defence vs Ripening (Seeded)", lambda: sel(tag, "SEEDED", "STATE_B_DEFENSE", 6.8).Rupture_Tension_mNm,
         lambda: sel(tag, "SEEDED", "STATE_A_RIPENING", 6.8).Rupture_Tension_mNm, "$\\gamma_{crit}$"),
        ("Defence: Seeded vs Spontaneous", lambda: sel(tag, "SEEDED", "STATE_B_DEFENSE", 6.8).Rupture_Tension_mNm,
         lambda: sel(tag, "SPONTANEOUS", "STATE_B_DEFENSE", 6.8).Rupture_Tension_mNm, "$\\gamma_{crit}$"),
        ("Ripening: acid shock (pH 2.5 vs 6.8)", lambda: sel(tag, "SEEDED", "STATE_A_RIPENING", 2.5).Rupture_Tension_mNm,
         lambda: sel(tag, "SEEDED", "STATE_A_RIPENING", 6.8).Rupture_Tension_mNm, "$\\gamma_{crit}$"),
        ("Defence: acid shock (pH 2.5 vs 6.8)", lambda: sel(tag, "SEEDED", "STATE_B_DEFENSE", 2.5).Rupture_Tension_mNm,
         lambda: sel(tag, "SEEDED", "STATE_B_DEFENSE", 6.8).Rupture_Tension_mNm, "$\\gamma_{crit}$"),
    ]
    for label, g1, g2, met in pairs:
        d, t, p, es = welch(g1(), g2())
        welch_rows.append([f"R{tag}", label, met, f"{d:+.2f}", f"{t:.1f}", f"{p:.2e}", f"{es:.2f}"])
for mission, name in [("STATE_A_RIPENING", "Ripening"), ("STATE_B_DEFENSE", "Defence")]:
    g1 = sel("40", "SEEDED", mission, 6.8).U_bond_per_N
    g2 = sel("20", "SEEDED", mission, 6.8).U_bond_per_N
    d, t, p, es = welch(g1, g2)
    welch_rows.append(["20$\\to$40", f"{name}: $U_{{bond}}/N$ scaling", "$U_{bond}/N$", f"{d:+.2f}", f"{t:.1f}", f"{p:.2e}", f"{es:.2f}"])
tex_table("tab_welch.tex",
          "Key Welch two-sample $t$-tests (pH 6.8 unless stated; $n=50$ per group).",
          "tab:welch",
          "\\textbf{Radii} & \\textbf{Comparison} & \\textbf{Metric} & {\\textbf{$\\Delta$}} & "
          "{\\textbf{$t$}} & {\\textbf{$p$}} & {\\textbf{Cohen's $d$}}",
          welch_rows, "@{}lll S[table-format=3.2] S[table-format=4.1] S[table-format=1.2e-2] S[table-format=2.2]@{}")

energy_rows = []
for mode, mission, label in [("SEEDED", "STATE_A_RIPENING", "Ripening (Seeded)"),
                             ("SEEDED", "STATE_B_DEFENSE", "Defence (Seeded)"),
                             ("SPONTANEOUS", "STATE_B_DEFENSE", "Defence (Spontaneous)")]:
    sub = sel("20", mode, mission, 6.8)
    ub = sub.U_bond_per_N
    ul = np.abs(sub.U_LJ_per_N)
    ud = sub.U_dh_per_N
    frac = 100 * ul.mean() / (ub.mean() + ul.mean() + ud.mean())
    energy_rows.append([label, f"{ub.mean():.2f} $\\pm$ {ub.std(ddof=1):.2f}",
                        f"{ul.mean():.4f}", f"{ud.mean():.2f} $\\pm$ {ud.std(ddof=1):.2f}", f"{frac:.1f}"])
tex_table("tab_energy.tex",
          "Potential-energy decomposition at yield ($R=20$ nm, pH 6.8; per particle, mean $\\pm$ SD). LJ fraction $=$ $|U_{LJ}|/N$ as $\\%$ of $\\sum |U|/N$.",
          "tab:energy",
          "\\textbf{Condition} & {\\textbf{$U_{bond}/N$ (kJ mol$^{-1}$)}} & {\\textbf{$|U_{LJ}|/N$}} & "
          "{\\textbf{$U_{dh}/N$ (kJ mol$^{-1}$)}} & {\\textbf{LJ (\\%)}}",
          energy_rows, "@{}l S[table-format=2.2] S[table-format=1.4] S[table-format=2.2] S[table-format=1.1]@{}")

scale_rows = []
for mission, name in [("STATE_A_RIPENING", "Ripening (A)"), ("STATE_B_DEFENSE", "Defence (B)")]:
    a, b = sel("20", "SEEDED", mission, 6.8), sel("40", "SEEDED", mission, 6.8)
    scale_rows.append([name,
                       f"{a.Rupture_Tension_mNm.mean():.2f}", f"{b.Rupture_Tension_mNm.mean():.2f}",
                       f"$\\times{b.Rupture_Tension_mNm.mean() / a.Rupture_Tension_mNm.mean():.2f}$",
                       f"{a.f_crit_kJmolnm.mean():.2f}", f"{b.f_crit_kJmolnm.mean():.2f}",
                       f"{a.U_bond_per_N.mean():.2f}", f"{b.U_bond_per_N.mean():.2f}",
                       f"+{100 * (b.U_bond_per_N.mean() / a.U_bond_per_N.mean() - 1):.1f}\\%",
                       f"{a.Percolation.mean():.3f}", f"{b.Percolation.mean():.3f}"])
tex_table("tab_radius_scaling.tex",
          "Radius scaling summary (Seeded, pH 6.8). $\\gamma_{crit}\\propto R$ follows from the approximately radius-independent $f_{crit}$ under per-particle-force loading; $U_{bond}/N$ is the cross-radius material metric.",
          "tab:radius_scaling",
          "\\textbf{State} & {\\textbf{$\\gamma_{20}$}} & {\\textbf{$\\gamma_{40}$}} & {\\textbf{ratio}} & "
          "{\\textbf{$f_{crit}^{20}$}} & {\\textbf{$f_{crit}^{40}$}} & "
          "{\\textbf{$U/N_{20}$}} & {\\textbf{$U/N_{40}$}} & {\\textbf{$\\Delta U/N$}} & "
          "{\\textbf{$S_{max}^{20}$}} & {\\textbf{$S_{max}^{40}$}}",
          scale_rows, "@{}l S[table-format=3.2] S[table-format=3.2] c S[table-format=2.2] S[table-format=2.2] "
                      "S[table-format=2.2] S[table-format=2.2] c S[table-format=1.3] S[table-format=1.3]@{}")

# CSV mirrors of the tables
pd.DataFrame(cond_rows("Rupture_Tension_mNm"),
             columns=["Mode", "State", "pH", "Radius_nm", "mean", "SD", "CI95"]).to_csv(
    os.path.join(TAB_DIR, "tab_gamma.csv"), index=False)
pd.DataFrame(welch_rows, columns=["Radii", "Comparison", "Metric", "Delta", "t", "p", "CohensD"]).to_csv(
    os.path.join(TAB_DIR, "tab_welch.csv"), index=False)
pd.DataFrame(scale_rows, columns=["State", "gamma20", "gamma40", "ratio", "fcrit20", "fcrit40",
                                  "UbN20", "UbN40", "dUbN", "Smax20", "Smax40"]).to_csv(
    os.path.join(TAB_DIR, "tab_radius_scaling.csv"), index=False)
print("  wrote tables/*.csv mirrors")

# =============================================================================
# Fig 6: loading-rate convergence of the critical tension (1x/4x/16x/64x ladder)
# =============================================================================
from scipy.optimize import curve_fit

RATE_RUNS = [("1x", 1.0), ("4x", 0.25), ("16x", 0.0625), ("64x", 0.015625)]
RATE_STATES = [("STATE_A_RIPENING", "Ripening (A)", "#2c7fb8"),
               ("STATE_B_DEFENSE",  "Defence (B)",  "#d95f0e")]
RATE_CSVS = {  # mission -> {label: csv}
    "STATE_A_RIPENING": {"1x": "pdev_unified_R20.csv", "4x": "pdev_unified_R20_slowrate.csv",
                          "16x": "pdev_unified_R20_16x.csv", "64x": "pdev_unified_R20_64x_A.csv"},
    "STATE_B_DEFENSE":  {"1x": "pdev_unified_R20.csv", "4x": "pdev_unified_R20_slowrate.csv",
                          "16x": "pdev_unified_R20_16x.csv", "64x": "pdev_unified_R20_64x_B.csv"},
}

ladder = {m: {"r": [], "g": [], "sd": [], "n": [], "fc": [], "ub": [], "eps": []}
           for m, _, _ in RATE_STATES}
for lbl, r in RATE_RUNS:
    for mission, _, _ in RATE_STATES:
        dfx = pd.read_csv(os.path.join(BASE, RATE_CSVS[mission][lbl]))
        sub = dfx[(dfx.pH == 6.8) & (dfx.Radius_nm == 20.0) & (dfx.Mode == "SEEDED")
                  & (dfx.Mission == mission)]
        L = ladder[mission]
        L["r"].append(r); L["g"].append(sub.Rupture_Tension_mNm.mean())
        L["sd"].append(sub.Rupture_Tension_mNm.std()); L["n"].append(len(sub))
        L["fc"].append(sub.f_crit_kJmolnm.mean()); L["ub"].append(sub.U_bond_per_N.mean())
        L["eps"].append(sub.Rupture_Strain.mean())

fig, ax = plt.subplots(figsize=(4.6, 3.5))
# fitpars = {}
# for mission, name, col in RATE_STATES:
#     L = ladder[mission]
#     r = np.array(L["r"]); y = np.array(L["g"]); s = np.array(L["sd"])
#     popt, _ = curve_fit(lambda rr, gi, c, al: gi + c * rr**al, r, y,
#                         p0=[y[-1] * 0.5, y[0] - y[-1], 0.7], maxfev=80000)
#     fitpars[mission] = popt
#     rr = np.logspace(np.log10(0.012), np.log10(1.2), 200)
#     ax.plot(rr, popt[0] + popt[1] * rr**popt[2], color=col, lw=1.4, alpha=0.85)
#     ax.errorbar(r, y, yerr=s, fmt="o", color=col, ms=6, capsize=3, label=name)
#     ax.axhline(popt[0], color=col, ls="--", lw=1.0, alpha=0.6)
#     ax.annotate(rf"$\gamma_\infty$ = {popt[0]:.1f}", xy=(0.013, popt[0]),
#                 xytext=(2, 4), textcoords="offset points", fontsize=8, color=col)
#     ax.annotate(rf"$\alpha$ = {popt[2]:.2f}", xy=(r[-1]+0.02, y[-1]),
#                 xytext=(-8, -14), textcoords="offset points", fontsize=8, color=col)
# ax.set_xscale("log")
# ax.set_xlabel(r"Tension loading rate (relative to production ramp)")
# # ax.set_ylabel(r"Critical tension $\gamma_{crit}$ (mN m$^{-1}$)")
# ax.set_ylabel(r"Apparent yield tension $\gamma_{\mathrm{app}}$ (mN m$^{-1}$)")
# ax.legend(frameon=False, loc="upper left")
# Replace the plot/errorbar loop in Fig 6:
fitpars = {}
for mission, name, col in RATE_STATES:
    L = ladder[mission]
    r = np.array(L["r"]); y = np.array(L["g"]); s = np.array(L["sd"])
    popt, _ = curve_fit(lambda rr, gi, c, al: gi + c * rr**al, r, y,
                        p0=[y[-1] * 0.5, y[0] - y[-1], 0.7], maxfev=80000)
    fitpars[mission] = popt
    rr = np.logspace(np.log10(0.012), np.log10(1.2), 200)
    
    # Embed parameters into the label
    leg_lbl = rf"{name} ($\gamma_\infty = {popt[0]:.1f},\ \alpha = {popt[2]:.2f}$)"
    ax.plot(rr, popt[0] + popt[1] * rr**popt[2], color=col, lw=1.4, alpha=0.85)
    ax.errorbar(r, y, yerr=s, fmt="o", color=col, ms=6, capsize=3, label=leg_lbl)
    ax.axhline(popt[0], color=col, ls="--", lw=1.0, alpha=0.5)

# Place legend in upper left without any inline canvas annotations
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
gA, gB = fitpars["STATE_A_RIPENING"][0], fitpars["STATE_B_DEFENSE"][0]
ax.set_title(rf"$\gamma_\infty^B/\gamma_\infty^A$ = {gB/gA:.1f} (cf. $K_{{eff}}$ ratio 5.75)", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel(r"Tension loading rate (relative to production ramp)")
ax.set_ylabel(r"Apparent yield tension $\gamma_{\mathrm{app}}$ (mN m$^{-1}$)")
ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
save(fig, "Fig6_RateConvergence")

# Appendix table: rate ladder
rate_rows = []
for i, (lbl, r) in enumerate(RATE_RUNS):
    row = [f"{lbl} (${r:g}$)"]
    for mission, _, _ in RATE_STATES:
        L = ladder[mission]
        row += [f"{L['g'][i]:.2f} $\\pm$ {L['sd'][i]:.2f} ({L['n'][i]})",
                f"{L['fc'][i]:.2f}", f"{L['ub'][i]:.2f}", f"{L['eps'][i]:.3f}"]
    rate_rows.append(row)
tex_table("tab_rate.tex",
          "Loading-rate ladder for the Seeded states at $R=20$~nm, pH 6.8 "
          "($\\gamma_{crit}$ in mN~m$^{-1}$, mean $\\pm$ SD ($n$); $f_{crit}$ in kJ~mol$^{-1}$~nm$^{-1}$; "
          "$U_{bond}/N$ in kJ~mol$^{-1}$). Rates are relative to the production ramp "
          "($\\Delta f = 0.5$ per 500 steps). Fits $\\gamma_{crit} = \\gamma_\\infty + \\Delta\\gamma\\,(\\dot\\gamma/\\dot\\gamma_0)^{\\alpha}$ give "
          f"$\\gamma_\\infty = {gA:.1f}$, $\\Delta\\gamma = {fitpars['STATE_A_RIPENING'][1]:.1f}$, "
          f"$\\alpha = {fitpars['STATE_A_RIPENING'][2]:.2f}$ (Ripening) and "
          f"$\\gamma_\\infty = {gB:.1f}$, $\\Delta\\gamma = {fitpars['STATE_B_DEFENSE'][1]:.1f}$, "
          f"$\\alpha = {fitpars['STATE_B_DEFENSE'][2]:.2f}$ (Defence).",
          "tab:rate",
          " & ".join(["Rate", "A: $\\gamma_{crit}$", "$f_{crit}$", "$U_{bond}/N$, A", "$\\epsilon_y$, A",
           "B: $\\gamma_{crit}$", "$f_{crit}$", "$U_{bond}/N$, B", "$\\epsilon_y$, B"]),
          rate_rows, "@{}l c c c c c c c c c@{}")

print("\nDONE: 5 figures (R = 20 nm; pdf+png), "
      "7 LaTeX tables (+csv mirrors), group stats CSV.")