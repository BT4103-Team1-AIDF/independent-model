Comp2 UI checklist (local Codabench)

Details tab
- Title: AIDF Local - Comp2 1Y Rolling (or any name)
- Training Mode: Rolling window
- Competition Docker Image: codalab/codalab-legacy:py37
- Competition Type: Competition
- Publish: checked (after setup)

Rolling parameters (important)
- period_col: yyyy
- rolling_start_period: 2015
- rolling_end_period: 2024
- rolling_window_size: 8

Participation tab
- Keep default terms or paste your team terms

Pages tab
- Add at least 1 page (e.g., Overview)

Phases tab
- Add phase: Main Phase
- Add one task to this phase after creating task bundle

Task creation (inside phase)
Upload these 4 files:
- ingestion_program.zip
- scoring_program.zip
- input_data.zip
- reference_data.zip

Execution time
- Recommended: 1200 sec (20 min)

Compatibility note
- With the latest team Codabench (`aidf-codabench` commit `fbcfcda4`), no manual compute-worker key patch is needed.
- Keep leaderboard columns aligned to scoring outputs: `AUC_ALL`, `AUC_ROLL_MEAN`, `VALID_YEARS`.
