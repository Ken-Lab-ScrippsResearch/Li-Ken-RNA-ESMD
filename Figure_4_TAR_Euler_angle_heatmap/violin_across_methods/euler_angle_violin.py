#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

files = {
    "AlphaFold": Path("data/selected_euler_angles_Alphafold.csv"),
    "cMD": Path("data/selected_euler_angles_cMD.csv"),
    "GaMD": Path("data/selected_euler_angles_GaMD.csv"),
    "Rex-GaMD": Path("data/selected_euler_angles_Rex_GaMD.csv"),
    "REST2": Path("data/selected_euler_angles_REST2.csv"),
    "T-REMD": Path("data/selected_euler_angles_T-REMD.csv"),
}

outdir = Path(".")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8.5
mpl.rcParams["axes.linewidth"] = 0.8

method_order = ["AlphaFold", "cMD", "GaMD", "Rex-GaMD", "REST2", "T-REMD"]
angle_order = ["alpha", "beta", "gamma"]
label_map = {
    "alpha": r"$\alpha$ (°)",
    "beta": r"$\beta$ (°)",
    "gamma": r"$\gamma$ (°)",
}

cmap = plt.cm.Purples
shade_positions = [0.25, 0.40, 0.55, 0.70, 0.83, 0.96]
method_colors = {m: cmap(v) for m, v in zip(method_order, shade_positions)}

def wrap180(x):
    x = np.asarray(x, dtype=float)
    return ((x + 180.0) % 360.0) - 180.0

required_cols = ["alpha", "beta", "gamma"]
frames = []
for method, path in files.items():
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    tmp = df.copy()
    for c in required_cols:
        tmp[c] = wrap180(tmp[c].to_numpy())
    tmp["method"] = method
    frames.append(tmp[["method", "alpha", "beta", "gamma"]])

all_df = pd.concat(frames, ignore_index=True)
long_df = all_df.melt(
    id_vars="method",
    value_vars=angle_order,
    var_name="angle",
    value_name="value_deg"
)

for angle in angle_order:
    fig = plt.figure(figsize=(3.6, 2), constrained_layout=True)
    ax = fig.add_subplot(111)

    data = [
        long_df[(long_df["angle"] == angle) & (long_df["method"] == m)]["value_deg"].dropna().to_numpy()
        for m in method_order
    ]

    parts = ax.violinplot(
        data,
        positions=np.arange(1, len(method_order) + 1),
        widths=0.82,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    for body, method in zip(parts["bodies"], method_order):
        body.set_facecolor(method_colors[method])
        body.set_edgecolor("black")
        body.set_linewidth(0.65)
        body.set_alpha(0.75)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.0)

    # ax.set_title(f"{label_map[angle]} across methods", fontsize=10, pad=4)
    ax.set_xticks(np.arange(1, len(method_order) + 1))
    ax.set_xticklabels(method_order, rotation=45, ha="right", fontsize=7.8)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel(label_map[angle], fontsize=9)
    ax.tick_params(axis="both", width=0.75, length=2.7, labelsize=7.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    legend_handles = [Patch(facecolor=method_colors[m], edgecolor="black", label=m) for m in method_order]
    # ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.22),
    #           ncol=3, frameon=False, fontsize=7.2, handlelength=1.0, columnspacing=1.1)

    fig.savefig(outdir / f"figure_4_{angle}_violin.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
