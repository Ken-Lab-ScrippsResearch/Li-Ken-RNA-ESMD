#!/usr/bin/env python3
"""
Create a publication-style heatmap of the minimum geodesic rotation distance
between each TAR ensemble and each ligand-bound experimental structure.

Figure format:
    - 9 x 6 inch canvas
    - DejaVu Sans
    - 600 dpi PNG
    - 8 ensemble rows
    - 6 ligand-bound PDB columns
    - no Mean column
    - white cell borders
    - bold, automatically contrasted annotations
    - light-peach to dark-purple color scale used for the published panel

Example:
    python geodesic_distance_heatmap.py \
        --data-dir data \
        --output figure_6_geodesic_distance_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
    }
)

VMIN = 0.0
DEFAULT_VMAX = 40.0
EULER_SEQUENCE = "ZXZ"

ENSEMBLE_FILES: Dict[str, str] = {
    "1ANR": "abg_results_1ANR.csv",
    "FARFAR": "abg_results_FARFAR.csv",
    "cMD": "abg_results_cMD.csv",
    "GaMD": "abg_results_GaMD.csv",
    "REST2": "abg_results_REST2.csv",
    "Rex-GaMD": "abg_results_Rex_GaMD.csv",
    "T-REMD": "abg_results_T-REMD.csv",
    "AlphaFold": "abg_results_Alphafold.csv",
}

REFERENCE_FILES: Dict[str, str] = {
    "1ARJ": "abg_results_1ARJ.csv",
    "1LVJ": "abg_results_1LVJ.csv",
    "1QD3": "abg_results_1QD3.csv",
    "1UTS": "abg_results_1UTS.csv",
    "1UUD": "abg_results_1UUD.csv",
    "1UUI": "abg_results_1UUI.csv",
}


def wrap_deg(values: pd.Series) -> pd.Series:
    """Wrap angles to [-180, 180)."""
    return ((values + 180.0) % 360.0) - 180.0


def detect_angle_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """Detect alpha, beta, and non-negated gamma columns."""
    lower = df.columns.str.lower()

    alpha_hits = df.columns[lower.str.contains("alpha")]
    beta_hits = df.columns[lower.str.contains("beta")]
    gamma_hits = df.columns[
        lower.str.contains("gamma") & ~lower.str.contains("neg")
    ]

    if len(alpha_hits) == 0 or len(beta_hits) == 0 or len(gamma_hits) == 0:
        raise ValueError(
            "Could not detect alpha, beta, and gamma columns. "
            f"Columns found: {list(df.columns)}"
        )

    return alpha_hits[0], beta_hits[0], gamma_hits[0]


def load_euler_angles(filepath: Path) -> np.ndarray:
    """Load valid alpha, beta, and gamma values from one CSV file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing input file: {filepath}")

    df = pd.read_csv(filepath)

    if "ok" in df.columns:
        ok_numeric = pd.to_numeric(df["ok"], errors="coerce")
        df = df.loc[ok_numeric == 1].copy()

    alpha_col, beta_col, gamma_col = detect_angle_columns(df)

    angles = df[[alpha_col, beta_col, gamma_col]].copy()
    angles.columns = ["alpha", "beta", "gamma"]
    angles = angles.apply(pd.to_numeric, errors="coerce").dropna()

    if angles.empty:
        raise ValueError(f"No valid Euler-angle rows found in {filepath}")

    for column in ("alpha", "beta", "gamma"):
        angles[column] = wrap_deg(angles[column])

    return angles[["alpha", "beta", "gamma"]].to_numpy(dtype=float)


def to_rotation_matrices(euler_angles: np.ndarray) -> np.ndarray:
    """Convert Euler-angle triplets in degrees to rotation matrices."""
    return Rotation.from_euler(
        EULER_SEQUENCE,
        euler_angles,
        degrees=True,
    ).as_matrix()


def pairwise_geodesic_distances(
    ensemble_angles: np.ndarray,
    reference_angles: np.ndarray,
) -> np.ndarray:
    """
    Calculate all pairwise geodesic rotation distances in degrees.

    d(R1, R2) = arccos[(Tr(R1^T R2) - 1) / 2]
    """
    ensemble_rotations = to_rotation_matrices(ensemble_angles)
    reference_rotations = to_rotation_matrices(reference_angles)

    relative_rotations = np.einsum(
        "aij,bjk->abik",
        np.transpose(ensemble_rotations, (0, 2, 1)),
        reference_rotations,
    )

    traces = np.trace(relative_rotations, axis1=2, axis2=3)
    cosine = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def calculate_nearest_distance_matrix(data_dir: Path) -> np.ndarray:
    """
    For every ensemble-reference pair, return the smallest distance between
    any ensemble conformer and any experimental model in that PDB entry.
    """
    ensemble_data = {
        name: load_euler_angles(data_dir / filename)
        for name, filename in ENSEMBLE_FILES.items()
    }
    reference_data = {
        name: load_euler_angles(data_dir / filename)
        for name, filename in REFERENCE_FILES.items()
    }

    matrix = np.zeros(
        (len(ENSEMBLE_FILES), len(REFERENCE_FILES)),
        dtype=float,
    )

    for row_index, ensemble_name in enumerate(ENSEMBLE_FILES):
        for column_index, reference_name in enumerate(REFERENCE_FILES):
            pairwise = pairwise_geodesic_distances(
                ensemble_data[ensemble_name],
                reference_data[reference_name],
            )
            matrix[row_index, column_index] = float(np.min(pairwise))

    return matrix


def panel_b_colormap() -> mcolors.LinearSegmentedColormap:
    """Light peach for small distances and dark purple for large distances."""
    anchors = [
        (0.00, "#F6E2D0"),
        (0.20, "#ECB591"),
        (0.35, "#E5825E"),
        (0.50, "#D74D46"),
        (0.65, "#B63156"),
        (0.80, "#85295A"),
        (0.90, "#56224F"),
        (0.96, "#281634"),
        (1.00, "#0D0F18"),
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "panel_b_peach_purple",
        anchors,
    )


def auto_text_color(
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    value: float,
) -> str:
    """Choose black or white annotation text from cell luminance."""
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance > 0.55 else "white"


def make_heatmap(
    nearest_matrix: np.ndarray,
    output_file: Path,
    vmax: float,
) -> None:
    """Render the six-column direct-distance heatmap without a Mean column."""
    ensemble_names = list(ENSEMBLE_FILES.keys())
    pdb_names = list(REFERENCE_FILES.keys())

    cmap = panel_b_colormap()
    norm = mcolors.Normalize(vmin=VMIN, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    image = ax.imshow(
        nearest_matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(len(pdb_names)),
        labels=pdb_names,
        fontsize=12,
    )
    ax.set_yticks(
        np.arange(len(ensemble_names)),
        labels=ensemble_names,
        fontsize=12,
    )
    ax.tick_params(length=0)

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

    ax.set_xticks(
        np.arange(len(pdb_names) + 1) - 0.5,
        minor=True,
    )
    ax.set_yticks(
        np.arange(len(ensemble_names) + 1) - 0.5,
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

    for row_index in range(len(ensemble_names)):
        for column_index in range(len(pdb_names)):
            value = nearest_matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=auto_text_color(cmap, norm, value),
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
        pad=0.03,
    )
    colorbar.set_label(
        "Minimum Geodesic Rotation Distance (°)",
        fontsize=12,
    )
    colorbar.ax.tick_params(labelsize=10)
    colorbar.outline.set_linewidth(0.6)

    fig.suptitle(
        "Minimum Geodesic Rotation Distance: "
        "Direct Coverage of Experimental Structures",
        fontsize=14,
        y=0.95,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate and plot the nearest geodesic rotation distance "
            "between TAR ensembles and ligand-bound experimental structures."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the abg_results_*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure_6_geodesic_distance_heatmap.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=DEFAULT_VMAX,
        help="Maximum value of the fixed heatmap color scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading Euler-angle data and calculating direct distances...")
    nearest_matrix = calculate_nearest_distance_matrix(args.data_dir)

    result_table = pd.DataFrame(
        nearest_matrix,
        index=list(ENSEMBLE_FILES.keys()),
        columns=list(REFERENCE_FILES.keys()),
    )

    print("\nMinimum geodesic rotation distances in degrees:")
    print(result_table.round(2).to_string())

    make_heatmap(
        nearest_matrix=nearest_matrix,
        output_file=args.output,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
