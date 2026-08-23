#!/usr/bin/env python3
"""
Generate a publication-style Mahalanobis-distance heatmap using only the
reversed Seaborn Rocket colormap.

The heatmap contains the six ligand-bound PDB structures only. No mean column
is calculated or displayed.

Run with:
    python mahalanobis_distance_heatmap.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------------------------------
# Data loading and Mahalanobis-distance calculation
# -------------------------------------------------------------------------
def wrap_deg(x):
    """Wrap angular values into the interval [-180, 180)."""
    return ((x + 180) % 360) - 180


def detect_cols(df):
    """Detect alpha, beta, and non-negated gamma columns."""
    cols = df.columns.str.lower()

    alpha_matches = df.columns[cols.str.contains("alpha")]
    beta_matches = df.columns[cols.str.contains("beta")]
    gamma_matches = df.columns[
        cols.str.contains("gamma") & ~cols.str.contains("neg")
    ]

    if (
        len(alpha_matches) == 0
        or len(beta_matches) == 0
        or len(gamma_matches) == 0
    ):
        raise ValueError(
            "Could not identify alpha, beta, and gamma columns. "
            f"Available columns: {list(df.columns)}"
        )

    return alpha_matches[0], beta_matches[0], gamma_matches[0]


def load_data(filepath):
    """Load and preprocess Euler-angle data from one CSV file."""
    df = pd.read_csv(filepath)

    # Retain only successfully processed rows when an 'ok' column is present.
    if "ok" in df.columns:
        ok_values = pd.to_numeric(df["ok"], errors="coerce")
        df = df.loc[ok_values == 1].copy()

    alpha_col, beta_col, gamma_col = detect_cols(df)

    abg = df[[alpha_col, beta_col, gamma_col]].copy()
    abg.columns = ["alpha", "beta", "gamma"]
    abg = abg.apply(pd.to_numeric, errors="coerce").dropna()

    if abg.empty:
        raise ValueError(f"No valid Euler-angle data found in {filepath}")

    abg = abg.apply(wrap_deg).reset_index(drop=True)
    abg["neg_gamma"] = -abg["gamma"]

    return abg


ensembles = {
    "1ANR": "data/abg_results_1ANR.csv",
    "FARFAR": "data/abg_results_FARFAR.csv",
    "cMD": "data/abg_results_cMD.csv",
    "GaMD": "data/abg_results_GaMD.csv",
    "REST2": "data/abg_results_REST2.csv",
    "Rex-GaMD": "data/abg_results_Rex_GaMD.csv",
    "T-REMD": "data/abg_results_T-REMD.csv",
    "AlphaFold": "data/abg_results_Alphafold.csv",
}

pdbs = {
    "1ARJ": "data/abg_results_1ARJ.csv",
    "1LVJ": "data/abg_results_1LVJ.csv",
    "1QD3": "data/abg_results_1QD3.csv",
    "1UTS": "data/abg_results_1UTS.csv",
    "1UUD": "data/abg_results_1UUD.csv",
    "1UUI": "data/abg_results_1UUI.csv",
}


print("Loading data...")

ens_data = {
    name: load_data(filepath)
    for name, filepath in ensembles.items()
}

pdb_data = {}

for name, filepath in pdbs.items():
    df = load_data(filepath)

    pdb_data[name] = {
        "alpha": df["alpha"].mean(),
        "beta": df["beta"].mean(),
        "neg_gamma": df["neg_gamma"].mean(),
    }


ens_names = list(ensembles.keys())
pdb_names = list(pdbs.keys())


print("Calculating Mahalanobis distances...")

mahal_matrix = np.zeros(
    (len(ens_names), len(pdb_names)),
    dtype=float,
)

for i, ens_name in enumerate(ens_names):
    X = ens_data[ens_name][
        ["alpha", "beta", "neg_gamma"]
    ].values

    ensemble_mean = X.mean(axis=0)
    covariance_inverse = np.linalg.inv(np.cov(X.T))

    for j, pdb_name in enumerate(pdb_names):
        pdb_point = np.array(
            [
                pdb_data[pdb_name]["alpha"],
                pdb_data[pdb_name]["beta"],
                pdb_data[pdb_name]["neg_gamma"],
            ]
        )

        difference = pdb_point - ensemble_mean

        mahal_matrix[i, j] = np.sqrt(
            difference @ covariance_inverse @ difference
        )


# -------------------------------------------------------------------------
# Publication-style Rocket heatmap
# -------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
    }
)

VMIN = 0
VMAX = 5

# Reversed Rocket keeps low distances light and high distances dark.
rocket_cmap = sns.color_palette(
    "rocket_r",
    as_cmap=True,
)


def auto_text_color(cmap, value):
    """Choose black or white text based on the cell-color luminance."""
    normalized_value = (
        np.clip(value, VMIN, VMAX) - VMIN
    ) / (VMAX - VMIN)

    red, green, blue, _ = cmap(normalized_value)

    luminance = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    return "black" if luminance > 0.55 else "white"


def make_heatmap():
    """Create the six-column Rocket heatmap without a mean column."""
    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    image = ax.imshow(
        mahal_matrix,
        cmap=rocket_cmap,
        aspect="auto",
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(len(pdb_names))
    )
    ax.set_yticks(
        np.arange(len(ens_names))
    )

    ax.set_xticklabels(
        pdb_names,
        fontsize=12,
    )
    ax.set_yticklabels(
        ens_names,
        fontsize=12,
    )

    ax.tick_params(
        length=0
    )

    # Add the Mahalanobis-distance value to each cell.
    for i in range(len(ens_names)):
        for j in range(len(pdb_names)):
            value = mahal_matrix[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=auto_text_color(
                    rocket_cmap,
                    value,
                ),
            )

    ax.set_xlabel(
        "Ligand-Bound PDB Structure",
        fontsize=14,
        labelpad=10,
    )

    ax.set_ylabel(
        "Ensemble",
        fontsize=14,
        labelpad=10,
    )

    # White borders between heatmap cells.
    ax.set_xticks(
        np.arange(len(pdb_names) + 1) - 0.5,
        minor=True,
    )
    ax.set_yticks(
        np.arange(len(ens_names) + 1) - 0.5,
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=2,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
        pad=0.03,
    )

    colorbar.set_label(
        "Mahalanobis Distance",
        fontsize=12,
    )

    colorbar.ax.tick_params(
        labelsize=10
    )

    colorbar.outline.set_linewidth(0.6)

    fig.suptitle(
        "Mahalanobis Distance: "
        "Compatibility with Experimental Structures",
        fontsize=14,
        y=0.95,
    )

    output_png = (
        "figure_6_mahalanobis_distance_heatmap.png"
    )

    fig.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_png}")


make_heatmap()

print("Done.")
