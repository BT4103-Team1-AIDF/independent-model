Competition phase files package

Files:
- ingestion_program.zip
- scoring_program.zip
- input_data.zip
- reference_data.zip

Source mapping:
- ingestion_program.zip <- updated ingestion (dynamic train/test split support + target auto-detect)
- scoring_program.zip <- updated scoring (labels.csv/test_outcomes auto-detect + dynamic join keys)
- input_data.zip <- Kexuan competition data seeting/input_data.zip
- reference_data.zip <- Kexuan competition data seeting/reference_data.zip

Schema notes:
- input_data provides full features panel (features.csv)
- reference_data provides labels table (labels.csv)
- train/test split is created internally by CodaBench static split configuration
- train labels are merged internally by shared join columns
