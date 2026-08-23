#!/usr/bin/env bash
# Reproduce the Figure 4 Rex_GaMD panel.
python ../euler_angle_heatmap.py \
    data/abg_results.csv \
    data/selected_frame_numbers.txt \
    -o . \
    --binwidth 20 \
    --prefix figure_4_Rex_GaMD_euler_angle
