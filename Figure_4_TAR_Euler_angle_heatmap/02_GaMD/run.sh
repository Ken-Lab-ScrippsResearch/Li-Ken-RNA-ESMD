#!/usr/bin/env bash
# Reproduce the Figure 4 GaMD panel.
python ../euler_angle_heatmap.py \
    data/abg_results.csv \
    data/selected_frame_numbers.txt \
    -o . \
    --binwidth 20 \
    --prefix figure_4_GaMD_euler_angle
