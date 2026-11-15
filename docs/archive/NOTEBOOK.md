# Archived competition notebook

[baseline_modeling.ipynb](../../notebooks/archive/baseline_modeling.ipynb)
preserves the original competition experiment source. It is retained to explain
how the project developed and is outside the supported baseline workflow.

The public copy adds a historical notice and removes saved outputs, execution
counts, execution metadata and widget state. The original source cells are
unchanged.

The notebook includes preprocessing and supervised feature selection before
validation, GAIN imputation, SMOTE and hyperparameter search. Its historical
scores are subject to those validation limitations. Clearing outputs does not
repair the experimental design.

Its code reads bare `train.csv` and `test.csv` filenames relative to the working
directory. These historical paths differ from the current baseline default
`data/train.csv`. Moving or opening the notebook does not update its source or
make it a reproduction of the current baseline. Do not use its final submission
cells as the supported prediction interface.

For the supported experiment, run `baseline_cv.py` from the repository root using
the [reproduction instructions](../REPRODUCE.md). The notebook is not required
for baseline installation, training, OOF analysis or the missingness comparison.
The full `requirements.txt` records dependencies of the legacy scripts; it does
not establish that this archived notebook runs unchanged on a new environment.
