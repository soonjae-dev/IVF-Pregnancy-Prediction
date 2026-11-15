# Controlled comparison: numeric missingness indicators

## Question and fixed protocol

Does adding numeric missingness indicators change prediction ranking? The single
change is `SimpleImputer(add_indicator=True)` for numeric columns. Indicator
columns are learned from the training fold only; validation-only missingness does
not add a new column. Categorical missing values retain the existing handling.

Both variants use all 256,351 rows, the same shuffled five stratified folds (seed
42), and the same LightGBM settings: 250 trees, learning rate 0.05, 31 leaves.
No parameters were searched for this comparison. OOF row identities, labels and
fold assignments are checked for exact alignment before a report is saved.

## Results

| Fold | Baseline AUC | With indicators | Difference |
|---|---:|---:|---:|
| 1 | 0.737002 | 0.737005 | +0.000003 |
| 2 | 0.741959 | 0.742271 | +0.000312 |
| 3 | 0.739233 | 0.739577 | +0.000345 |
| 4 | 0.737799 | 0.737859 | +0.000060 |
| 5 | 0.740195 | 0.740196 | +0.000000 |
| Pooled OOF | 0.739219 | 0.739365 | +0.000146 |

Mean fold difference: +0.000144; sample standard deviation of
fold differences: 0.000170. These five folds share training data;
this standard deviation is descriptive, not a confidence interval or significance
test. Pooled AUC is not the mean of fold AUCs.

Sequential elapsed times were 15.6 seconds for the baseline
and 15.7 seconds with indicators. These include fitting and
scoring; they exclude CSV output and are not a controlled runtime benchmark.

## Decision

Retain the simpler baseline as the portfolio reference. The observed AUC gain
is small, and one seed is insufficient to claim a reliable improvement. This
experiment does not establish that missingness is uninformative, nor does it
measure the benefit of GAIN. It demonstrates a controlled component comparison
with a qualified conclusion.

## Reproduce and inspect

```bash
python compare_missingness.py
```

A new timestamped output directory contains both OOF files and a JSON report
with dataset/source hashes, environment versions and paired results. Existing
runs are preserved. Raw data and OOF files are not included in Git.

[Recorded comparison](../../results/missingness_20261110T144629_494691Z/comparison.json) · [Current baseline](../../README.md)

The original comparison report is unchanged. Copies of both executed source files
are now preserved with hashes verified against that report; see
[recorded runs and source preservation](../RESULTS.md).
