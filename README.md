# independent-model

Independent Models repository for BT4103 / AIDF.

This repo now contains **both**:
1. upload-ready competition/submission assets, and  
2. original training/tuning source code.

---

## Repository Structure

### A) `Final_Submission/` (for Codabench operations)

- `Final_Submission/Competition/Static`
  - `Ingestion.zip`, `Scoring.zip`, `input_data.zip`, `reference_data.zip`
- `Final_Submission/Competition/Rolling`
  - `Ingestion.zip`, `Scoring.zip`, `input_data.zip`, `reference_data.zip`
  - `competition2_rolling_config.json`, `COMP2_UI_CHECKLIST.md`
- `Final_Submission/Models`
  - Model ZIPs: `Logistic.zip`, `RandomForest.zip`, `LightGBM.zip`, `XGBoost.zip`, `CatBoost.zip`
  - Model cards: `*_model_card.pdf`

Use this section when creating competitions or uploading submissions.

### B) `Training_Tuning_Source/` (for research reproducibility)

- Original benchmark training/tuning pipeline code used for independent models.
- Includes multi-horizon runs, tuning config/grid, and evaluation exports.
- See `Training_Tuning_Source/README.md` for run commands.

Use this section when teammates want to reproduce experiments or continue model development.

---

## Recommended workflow

1. Develop/benchmark in `Training_Tuning_Source/`.
2. Package stable submission models into `Final_Submission/Models/`.
3. Upload with corresponding model cards.

---

## Notes

- Existing `Final_Submission` content remains unchanged.
- Sensitive data and generated artifacts should not be committed.
- This layout follows the handover style used across team repos (`codabench`, `classifier-chain`, `survival-analysis`).
