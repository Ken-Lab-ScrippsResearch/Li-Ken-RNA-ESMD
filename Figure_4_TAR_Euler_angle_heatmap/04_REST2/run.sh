#!/usr/bin/env bash
# Reproduce the Figure 4 REST2 panel.
python ../euler_angle_heatmap.py \
    data/abg_results.csv \
    data/selected_frame_numbers.txt \
    -o . \
    --binwidth 20 \
    --prefix figure_4_REST2_euler_angle
