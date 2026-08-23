# Enhanced-Sampling Molecular Dynamics Recovers Rare Functional RNA Conformations Across Diverse Structural Contexts

Figure code and input data for the paper. 
Every figure directory is self-contained. The data sits next to the code that reads it, all paths are relative, and each script writes a 600 dpi PNG into its own directory. 

## Figures

| Directory | Content | Run |
|---|---|---|
| `Figure_2_TAR_RDC_correlation/` | Measured vs calculated RDCs for TAR, one panel per method | `rdc_correlation.ipynb` in each method directory |
| `Figure_3_TAR_chemical_shift_correlation/` | Chemical-shift correlations and weighted metrics | `chemical_shift_correlation.ipynb` |
| `Figure_4_TAR_Euler_angle_heatmap/` | Inter-helical Euler-angle heatmaps | `bash run.sh` in each method directory |
| `Figure_4_TAR_Euler_angle_heatmap/violin_across_methods/` | Euler-angle distributions across methods | `python euler_angle_violin.py` |
| `Figure_5_TAR_excited_state_population/` | Excited-state populations | `excited_state_population.ipynb` |
| `Figure_6_TAR_Euler_angle_ensemble_comparison/` | Ensemble vs ligand-bound structure distances | `python geodesic_distance_heatmap.py`, `python mahalanobis_distance_heatmap.py` |
| `Figure_6_TAR_Euler_angle_ensemble_comparison/EBVS_docking_ROC/` | Virtual-screening ROC curves | `ebvs_roc_analysis.ipynb` in each method directory |
| `Figure_7_benchmark_RNA_RDC_correlation/` | RDC correlations for the 2KOC and 2L1V benchmark RNAs | `rdc_correlation.ipynb` in each method directory |
| `ensemble/` | The 20-conformer ensembles behind the Figure 2 and 7 panels | input files, nothing to run |
