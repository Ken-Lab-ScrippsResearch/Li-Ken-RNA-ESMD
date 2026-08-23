#!/usr/bin/env python3
"""
General Euler-angle heatmap script.

Inputs
------
1) All-frame ABG CSV:
   must contain at least columns: frame, alpha, beta, gamma
   optional: zeta_alpha_plus_gamma

2) SAS selection TXT:
   format per row:
   ensemble_size  RDC_RMSD  frame_1 frame_2 ... frame_N

Outputs
-------
- PNG figure
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------
# Plot style
# -----------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 7.9
mpl.rcParams["axes.linewidth"] = 0.65


def wrap180(x):
    x = np.asarray(x, dtype=float)
    return ((x + 180.0) % 360.0) - 180.0


def read_sas_frames(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    ncols = {len(r) for r in rows}
    if len(ncols) != 1:
        raise ValueError(f"Inconsistent column counts in SAS file: {sorted(ncols)}")

    ensemble_sizes = np.array([int(r[0]) for r in rows], dtype=int)
    rmsd = np.array([float(r[1]) for r in rows], dtype=float)
    frames = np.array([[int(x) for x in r[2:]] for r in rows], dtype=int)
    rep_ids = np.arange(1, len(rows) + 1, dtype=int)

    return ensemble_sizes, rmsd, frames, rep_ids


def load_all_frames(path):
    df = pd.read_csv(path)

    required = {"frame", "alpha", "beta", "gamma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ABG CSV missing required columns: {sorted(missing)}")

    for col in ["alpha", "beta", "gamma", "zeta_alpha_plus_gamma"]:
        if col in df.columns:
            df[col] = wrap180(df[col].to_numpy())

    return df


def expand_selected_frames(abg_df, rmsd, frames, rep_ids):
    frame_map = abg_df.set_index("frame", drop=False)

    unique_frames = np.unique(frames)
    missing = np.setdiff1d(unique_frames, frame_map.index.to_numpy())
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} selected frames are missing from the ABG table. "
            f"Example missing frames: {missing[:10].tolist()}"
        )

    expanded = frame_map.loc[frames.ravel()].reset_index(drop=True).copy()
    expanded["replicate"] = np.repeat(rep_ids, frames.shape[1])
    expanded["sas_rdc_rmsd"] = np.repeat(rmsd, frames.shape[1])
    expanded["selection_rank"] = np.tile(np.arange(1, frames.shape[1] + 1), frames.shape[0])
    return expanded


def build_boundary_segments(mask, xedges, yedges):
    nxbins, nybins = mask.shape
    vertical_segments = []
    horizontal_segments = []

    for i in range(nxbins):
        for j in range(nybins):
            if not mask[i, j]:
                continue

            if j == 0 or not mask[i, j - 1]:
                horizontal_segments.append((yedges[j], xedges[i], xedges[i + 1]))
            if j == nybins - 1 or not mask[i, j + 1]:
                horizontal_segments.append((yedges[j + 1], xedges[i], xedges[i + 1]))
            if i == 0 or not mask[i - 1, j]:
                vertical_segments.append((xedges[i], yedges[j], yedges[j + 1]))
            if i == nxbins - 1 or not mask[i + 1, j]:
                vertical_segments.append((xedges[i + 1], yedges[j], yedges[j + 1]))

    return vertical_segments, horizontal_segments


def compute_hist_population(x, y, binwidth=20):
    edges = np.arange(-180, 181, binwidth)
    H, xedges, yedges = np.histogram2d(x, y, bins=[edges, edges], density=False)

    total = H.sum()
    if total == 0:
        raise ValueError("Histogram is empty; cannot compute population.")

    H = H / total * 100.0
    return H, xedges, yedges


def draw_hist(ax, H, xedges, yedges, vmin, vmax, title, xlabel, show_y_ticks=False):
    mask = H > 0
    Hm = np.ma.masked_where(~mask, H)

    mesh = ax.pcolormesh(
        xedges,
        yedges,
        Hm.T,
        cmap="Reds",
        shading="flat",
        vmin=vmin,
        vmax=vmax
    )

    vseg, hseg = build_boundary_segments(mask, xedges, yedges)
    for x, y0, y1 in vseg:
        ax.vlines(x, y0, y1, color="#565656", linewidth=0.30, zorder=3)
    for y, x0, x1 in hseg:
        ax.hlines(y, x0, x1, color="#565656", linewidth=0.30, zorder=3)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks([-120, 0, 120])
    ax.set_yticks([-120, 0, 120])

    ax.set_xlabel(xlabel, fontsize=7.9, labelpad=1.1)
    ax.set_ylabel("")

    if show_y_ticks:
        ax.tick_params(
            axis="y",
            labelleft=True,
            width=0.55,
            length=2.2,
            labelsize=6.9,
            pad=0.8
        )
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    ax.tick_params(
        axis="x",
        width=0.55,
        length=2.2,
        labelsize=6.9,
        pad=0.8
    )

    ax.set_aspect("equal")
    ax.set_facecolor("white")

    for s in ax.spines.values():
        s.set_linewidth(0.55)

    ax.set_title(title, fontsize=7.6, pad=2.2)
    return mesh


def save_png(fig, base):
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight", facecolor="white")


def make_onecol_row(df, outbase, binwidth=20):
    pairs = [
        ("alpha", "beta",  r"$\alpha$–$\beta$",  r"$\alpha$ (°)"),
        ("beta",  "gamma", r"$\beta$–$\gamma$",  r"$\beta$ (°)"),
        ("alpha", "gamma", r"$\alpha$–$\gamma$", r"$\alpha$ (°)"),
    ]

    hists = [compute_hist_population(df[x], df[y], binwidth=binwidth) for x, y, *_ in pairs]
    vmax = max(H.max() for H, _, _ in hists)

    # One-column width, matching your preferred size setup
    fig = plt.figure(figsize=(3.35, 1.46), facecolor="white")
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1, 1, 1, 0.08],
        left=0.055,
        right=0.985,
        bottom=0.25,
        top=0.88,
        wspace=0.055
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    mesh = None
    for i, (ax, (xcol, ycol, title, xlabel), (H, xedges, yedges)) in enumerate(zip(axes, pairs, hists)):
        mesh = draw_hist(
            ax,
            H,
            xedges,
            yedges,
            vmin=0,
            vmax=vmax,
            title=title,
            xlabel=xlabel,
            show_y_ticks=(i == 0)
        )

    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Population (%)", fontsize=7.9, labelpad=2.8)
    cb.ax.tick_params(labelsize=6.9, width=0.5, length=2.0, pad=0.8)
    cb.outline.set_linewidth(0.55)

    save_png(fig, outbase)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Make one-column Euler-angle heatmaps from ABG CSV + SAS TXT.")
    parser.add_argument("abg_csv", help="All-frame ABG CSV")
    parser.add_argument("sas_txt", help="SAS frame-selection TXT")
    parser.add_argument("-o", "--outdir", default="euler_general_out", help="Output directory")
    parser.add_argument("--binwidth", type=int, default=20, help="Histogram bin width in degrees")
    parser.add_argument("--prefix", default="euler_onecol_row_population", help="Output file prefix")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    abg_df = load_all_frames(args.abg_csv)
    ensemble_sizes, sas_rmsd, frames, rep_ids = read_sas_frames(args.sas_txt)
    selected_df = expand_selected_frames(abg_df, sas_rmsd, frames, rep_ids)

    # Make the figure
    make_onecol_row(
        selected_df,
        outdir / f"{args.prefix}_bin{args.binwidth}",
        binwidth=args.binwidth
    )

    print(f"Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
