#!/bin/bash
# Sync figures with Google Drive or other folders so everyone can see them
gdrive="/Volumes/GoogleDrive/Shared drives/Stanford Computational Imaging/Projects/single_spad_depth/figures/";
rsync captured/midas/8_30_small_lab_scene/teaser.eps "$gdrive"
rsync full_pipeline/full_pipeline.jpeg "$gdrive"
shopt -s extglob  #Allows for | in bash pattern matching below
rsync -r comparison/densedepth_?(468|194|258)_comparison.png "$gdrive/comparison"
rsync prototype_single_col.png "$gdrive"
