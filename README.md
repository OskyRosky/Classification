# Classification — Data Science Repository

A didactic and production-ready repository for **supervised classification**:
- Clear theory (short), **mathematical core**, algorithms, **library implementations** (functions + parameters),
- minimal code (Python/R), **evaluation**, pitfalls, and **when to use / not use** each technique.

## Structure
See `docs/` for theory and `src/` for reusable code. Use `notebooks/compare_all.ipynb` to benchmark methods on the same dataset.

## Quickstart (Python)
cd /Users/sultan/DataScience/Classification
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_synthetic.py
python scripts/train_eval.py
