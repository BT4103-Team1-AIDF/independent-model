# Results Artifacts (Independent Models)

This folder contains the final evaluation artifacts used in client reporting and slides.

## 1) `8H_Benchmark_Static/`
Static-split benchmark outputs across 8 horizons (1, 3, 6, 12, 24, 36, 48, 60) for 5 independent models.

Key files:
- `benchmark_summary_all_horizons.csv`
- `metrics_summary_all_horizons.csv`
- `best_model_per_horizon.csv`
- `yearly_auc_8horizons_5models_subplots.png`
- `best_model_by_horizon.png`

## 2) `Comp1_vs_Comp2_12m_Tuned5/`
Comparison outputs for tuned 5 models (Logistic, RF, XGBoost, LightGBM, CatBoost) between:
- Comp1 (1Y static)
- Comp2 (1Y rolling)

Key files:
- `tuned5_static_rolling_results.csv`
- `tuned5_auc_all_static_vs_rolling_6dp.png`
- `tuned5_auc_rollmean_static_vs_rolling_6dp.png`
- `tuned5_yearly_auc_long.csv`
- `summary.md`

These artifacts are packaged for reproducibility and presentation support.
