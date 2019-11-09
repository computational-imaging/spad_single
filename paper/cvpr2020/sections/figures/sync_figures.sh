#!/usr/bin/env bash
# Sync figures with Google Drive or other folders so everyone can see them
gdrive="/Volumes/GoogleDrive/Shared drives/Stanford Computational Imaging/Projects/single_spad_depth/figures/";
rsync captured/midas/8_30_small_lab_scene/teaser.pdf "$gdrive"
rsync full_pipeline/full_pipeline.jpeg "$gdrive"
shopt -s extglob  #Allows for | in bash pattern matching below
rsync -r comparison/*.pdf "$gdrive/comparison"
rsync comparison.pdf "$gdrive"
rsync prototype_single_col.pdf "$gdrive"
rsync -R ././captured/**/**/?(gt_z_proj_crop_depth_fig|z_init_depth_fig|z_med_scaled_depth_fig|z_pred_depth_fig|rgb_cropped_fig|z_init_diff_fig|z_med_scaled_diff_fig|z_pred_diff_fig).png "$gdrive"
rsync -R ././captured/**/**/?(diff_colorbar|depth_colorbar).pdf "$gdrive" 
rsync method.pdf "$gdrive"
rsync captured.pdf "$gdrive"
rsync dither.pdf "$gdrive"
