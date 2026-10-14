# Chapter 4 tables and figures (parts ABC)

Generated 2026-10-14 00:04 by make_chapter4_tables.py from the files under C:\Users\Admin\Documents\Developments\thesis_project

## Produced (33)
- table: a1_dataset_flow 7 rows
- table: a2_split_counts 4 rows
- table: a2b_combined_training_set 3 rows
- table: a3_patient_leakage 3 rows
- table: a4_demographics 3 rows
- table: a5_setup4_subset 4 rows
- table: a5b_setup4_checks 4 rows
- table: a5c_setup4_demographics 3 rows
- table: a6_train_stats 3 rows
- table: b1_stylegan_options_full 53 rows
- table: b1_stylegan_options_key 15 rows
- table: b2_fid_by_snapshot 11 rows
- figure: b2_fid_curve 
- table: b3_training_summary 5 rows
- figure: b3_training_curves 
- table: b4_log_excerpt.txt 36 lines
- table: b5_synthetic_set 2 rows
- figure: b5_real_vs_generated 
- figure: b5_snapshot_reals.png copied from results\stylegan2-ada\png\reals.png
- figure: b5_snapshot_fakes_last.png kimg 5000, copied from results\stylegan2-ada\png\fakes005000.png
- table: c1_cluster_sizes 3 rows
- table: c2_cluster_quality 2 rows
- table: c3_scalar_means_per_cluster 4 rows
- table: c9_feature_and_cluster_settings 23 rows
- table: c4_manual_check 2 rows
- table: c5_marks_by_round 4 rows
- figure: c4_artifact_share_per_cluster 
- table: c6_final_domains 2 rows
- table: c7_domains_by_split 3 rows
- table: c8_purity_per_cluster 4 rows
- table: c8b_purity_per_domain_pool 2 rows
- figure: cluster_grids/ 8 contact sheets copied
- figure: c10_lung_mask_overlays 

## Skipped (0)

## Notes
- NIH metadata file: data\1-raw\Data_Entry_2017_v2020.csv
- c8_purity: estimates scale 50-image samples per cluster up to the whole cluster; ballpark only

## Assumptions to state in the thesis
- Class labels are read from the split files; 'Pneumonia' and 'No Finding' are matched by name.
- Purity figures (c8) scale 50-image mark samples to whole clusters; they are estimates.
- Chehade's cluster sizes and agreement are typed in once at the top of this script from their Table 1 and Fig. 5.
- Any table that is missing was skipped because its source file or column was not found (see above).
