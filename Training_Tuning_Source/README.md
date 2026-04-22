# Independent Models — Training & Tuning Source

This folder contains the original training/tuning pipeline source code used by the independent-model stream.

It is separated from `Final_Submission/` to keep Codabench upload packages stable while preserving full research reproducibility.

## What is included

- `benchmark.py`  
  Main benchmarking entrypoint (multi-model, multi-horizon run logic).
- `run_benchmarks.py`  
  CLI wrapper for rolling/submission-style runs.
- `benchmarking/`  
  Modular pipeline components:
  - `config.py`: default configs and hyperparameter grids
  - `data.py`: data loading helpers
  - `features.py`: feature selection and time-aware splitting
  - `models.py`: model builders and probability alignment
  - `tuning.py`: hyperparameter tuning logic
  - `metrics.py`: AUC/summary metrics
  - `evaluation.py`: plotting and metrics export
  - `runner.py`: static benchmark orchestration

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_train_tune.txt
```

### 2) Run a multi-horizon benchmark

```bash
python run_benchmarks.py \
  --mode rolling \
  --data-path /path/to/train.csv \
  --output-dir outputs \
  --horizons 1 3 6 12 24 36 48 60 \
  --models logistic random_forest xgboost lightgbm \
  --max-tuning-trials 12
```

### 3) Run one submission-style evaluation

```bash
python run_benchmarks.py \
  --mode submission \
  --data-path /path/to/train.csv \
  --test-data-path /path/to/test.csv \
  --output-dir outputs \
  --horizons 12 \
  --models lightgbm
```

## Notes

- Keep private datasets outside this repo.
- Generated outputs are intentionally excluded by `.gitignore`.
- For Codabench uploads, use `Final_Submission/Models/*.zip` instead of this source folder.
