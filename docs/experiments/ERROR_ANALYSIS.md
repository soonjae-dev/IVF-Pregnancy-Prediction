# Error analysis by subgroup

Where does the best model (`nan_blend`, OOF ROC-AUC 0.74052) gain over the baseline
(0.73922), and where do both models fail?

- Script: [`error_analysis.py`](../../error_analysis.py). It does no training: it reads
  the out-of-fold predictions the [missing-value types run](MISSING_TYPES.md) saved and
  checks them against the SHA-256 hashes recorded in that run's `comparison.json`.
- Result: [`results/error_analysis_20261115T162321Z/summary.json`](../../results/error_analysis_20261115T162321Z/summary.json).
  It holds aggregates only, with no input rows or per-row predictions.
- Re-run on 2026-11-16 after a fix: the script now reads the OOF files with exact
  float parsing (`float_precision="round_trip"`). The default parser could change the
  last digit of a score and moved AUC by about 1e-6. Of the numbers below, only one
  interval bound changed at four decimals (0 transferred: +0.1515 → +0.1514).
- Same rows and same folds for both models (seed 42, 5 folds). The interval is a paired
  bootstrap over rows (200 rounds, positives and negatives resampled separately). It
  shows how stable the difference is on these rows, not on new data or another split.
  Intervals are per group and not corrected for the number of groups.
- `blend` is a rank average over the whole training set. Ranks are preserved inside any
  subgroup, so AUC and AP within a group are well defined.
- AP depends on the positive rate. It is compared between the two models within a
  group, never between groups.

## Overall, and with the rows where pregnancy was not possible removed

| Rows | Count | Positives | Rate | AUC baseline → blend | Δ AUC (95% interval) | AP baseline → blend |
|---|---:|---:|---:|---|---|---|
| All training rows | 256,351 | 66,228 | 25.8% | 0.7392 → 0.7405 | +0.0013 (+0.0010 to +0.0017) | 0.4487 → 0.4511 |
| Embryo transferred, or DI | 219,807 | 66,198 | 30.1% | 0.6776 → 0.6792 | +0.0016 (+0.0012 to +0.0020) | 0.4488 → 0.4512 |

36,544 IVF rows record `이식된 배아 수 = 0`. Only 30 of them (0.08%) are pregnancies. A
model that learns "no embryo transferred → not pregnant" gets these rows almost entirely
right. That accounts for a large share of the headline AUC. **Among the rows where an
embryo was transferred (or DI was performed), both models rank at about 0.68, not 0.74.**

The improvement does not come from those easy rows. It is the same size or slightly
larger once they are removed (+0.0016).

## By subgroup

The 6,291 DI rows are the same rows as `난자 출처 = 알 수 없음`, `동결 배아 사용 여부`
blank and `이식된 배아 수` blank (type 1 in [MISSING_TYPES.md](MISSING_TYPES.md)), so
those four lines repeat one result.

| Axis | Group | Rows | Positives | Rate | AUC baseline → blend | Δ AUC (95% interval) | AP baseline → blend |
|---|---|---:|---:|---:|---|---|---|
| Treatment | IVF | 250,060 | 65,417 | 26.2% | 0.7373 → 0.7386 | +0.0013 (+0.0010 to +0.0016) | 0.4502 → 0.4526 |
| | DI | 6,291 | 811 | 12.9% | 0.6838 → 0.6903 | +0.0065 (+0.0022 to +0.0120) | 0.2326 → 0.2331 |
| Age | 18–34 | 102,476 | 33,061 | 32.3% | 0.7023 → 0.7036 | +0.0014 (+0.0008 to +0.0019) | 0.4810 → 0.4837 |
| | 35–37 | 57,780 | 16,086 | 27.8% | 0.7011 → 0.7022 | +0.0011 (+0.0004 to +0.0020) | 0.4367 → 0.4386 |
| | 38–39 | 39,247 | 8,522 | 21.7% | 0.7083 → 0.7102 | +0.0019 (+0.0009 to +0.0029) | 0.3755 → 0.3788 |
| | 40–42 | 37,348 | 5,953 | 15.9% | 0.7349 → 0.7380 | +0.0031 (+0.0018 to +0.0045) | 0.3289 → 0.3344 |
| | 43–44 | 12,253 | 1,446 | 11.8% | 0.8273 → 0.8291 | +0.0018 (+0.0000 to +0.0034) | 0.3783 → 0.3785 |
| | 45–50 | 6,918 | 1,160 | 16.8% | 0.8314 → 0.8321 | +0.0007 (−0.0016 to +0.0029) | 0.4178 → 0.4186 |
| | unknown | 329 | 0 | 0% | — (one class) | — | — |
| Egg source | own | 234,291 | 60,443 | 25.8% | 0.7406 → 0.7419 | +0.0013 (+0.0010 to +0.0016) | 0.4498 → 0.4521 |
| | donor | 15,769 | 4,974 | 31.5% | 0.6797 → 0.6816 | +0.0019 (−0.0004 to +0.0043) | 0.4543 → 0.4582 |
| Frozen embryo | no | 209,934 | 56,212 | 26.8% | 0.7466 → 0.7476 | +0.0010 (+0.0007 to +0.0013) | 0.4629 → 0.4653 |
| | yes | 40,126 | 9,205 | 22.9% | 0.6500 → 0.6546 | +0.0046 (+0.0026 to +0.0066) | 0.3315 → 0.3364 |
| Embryos transferred | 0 | 36,544 | 30 | 0.08% | 0.5856 → 0.6122 | +0.0265 (−0.0957 to +0.1514) | 0.0011 → 0.0043 |
| | 1 | 93,791 | 29,408 | 31.4% | 0.6823 → 0.6843 | +0.0020 (+0.0014 to +0.0027) | 0.4550 → 0.4582 |
| | 2 | 110,845 | 34,483 | 31.1% | 0.6543 → 0.6557 | +0.0014 (+0.0008 to +0.0020) | 0.4475 → 0.4493 |
| | 3 or more | 8,880 | 1,496 | 16.8% | 0.6630 → 0.6667 | +0.0037 (−0.0003 to +0.0071) | 0.2848 → 0.2895 |

The "0 transferred" group has only 30 positives (flagged as small in the JSON), so its
numbers carry no weight. The "unknown age" group has no pregnancies at all.

### What the table shows

1. **The blend gains a little almost everywhere, and does not lose clearly anywhere.**
   Every group with a defined interval has a positive point estimate. The interval
   includes 0 in four smaller groups: age 45–50, donor eggs, 3 or more transferred, and
   0 transferred (age 43–44 touches 0).
2. **The gain is larger where the baseline is weakest**: DI (+0.0065), frozen-embryo
   cycles (+0.0046), age 40–42 (+0.0031). These are the groups where categorical or
   sparse inputs matter more, which is where CatBoost differs from one-hot LightGBM. That
   is an interpretation, not a tested cause.
3. **Hardest groups for both models**: frozen-embryo cycles (AUC 0.65), two embryos
   transferred (0.65), donor eggs (0.68) and DI (0.68–0.69). The model ranks these
   patients much worse than the headline 0.74 suggests.
4. **Older age groups look easy (0.83) because most of them do not become pregnant.**
   AUC within a group says how well the model ranks inside that group. It does not mean
   the model is more useful for those patients.

## How much one variable ranks on its own

ROC-AUC of a score that is only the positive rate of the row's group, with rates learned
on the other folds:

| Variable | AUC alone |
|---|---:|
| `이식된 배아 수` (bucketed) | 0.615 |
| `시술 당시 나이` | 0.596 |
| `동결 배아 사용 여부` | 0.520 |
| `난자 출처` | 0.516 |
| `시술 유형` | 0.508 |

The number of embryos transferred is the strongest single variable here, ahead of age.
It is recorded at the transfer, after most of the treatment has happened.

## When each input becomes known

The competition asks to predict pregnancy from every column. That is a valid
prediction task, but its model is not a model that could be used **before treatment**.
The table below groups the inputs by the earliest point at which they would be known.
This is my reading of the column names. The official data dictionary was not available,
so the boundaries are assumptions (for example, whether PGS or frozen-embryo use is
decided at planning or later).

| Stage | Inputs |
|---|---|
| Before treatment: patient history | `시술 시기 코드`, `시술 당시 나이`, `임신 시도 또는 마지막 임신 경과 연수`, all infertility-cause columns, past treatment, pregnancy and birth counts, `난자 출처`, `정자 출처`, donor ages, `대리모 여부` |
| Treatment plan | `시술 유형`, `특정 시술 유형`, `배란 자극 여부`, `배란 유도 유형`, `배아 생성 주요 이유`, PGS / PGD and pre-implantation testing flags, `동결 / 신선 / 기증 배아 사용 여부` |
| Lab stage (after egg collection and fertilization) | egg and embryo counts (`수집된 신선 난자 수`, `혼합된 난자 수`, `총 생성 배아 수`, microinjection counts, stored and thawed counts), the egg collection, thawing and mixing days, `배아 해동 경과일` |
| At transfer | `이식된 배아 수`, `미세주입 배아 이식 수`, `단일 배아 이식 여부`, `배아 이식 경과일` |

Consequences for how the results should be read:

- The 0.74 AUC is for a model that sees the full record of a cycle, including how many
  embryos were transferred. It is not the accuracy of a pre-treatment prediction.
- Rows with no transfer make the task look easier than it is for the cycles that matter.
  Among cycles with a transfer (or DI), AUC is about 0.68.
- A pre-treatment model would use only the first two stages. It was not trained here,
  and its accuracy is unknown; it would be lower.
- The data has no patient or clinic identifier. Rows from the same patient can fall in
  both training and validation folds, so generalization to new patients or clinics is
  not measured.
- Importance of a variable here means it helps ranking. It does not mean the variable
  causes pregnancy; the number of embryos transferred, for example, reflects decisions
  made on how the cycle was going.

## Reproduce

```bash
python error_analysis.py
```

Defaults: the latest `results/missing_types_*` run, `baseline` vs `nan_blend`, 200
bootstrap rounds. It needs that run's OOF CSV files, which are git-ignored and are
written by `missing_types_cv.py` (see [REPRODUCE.md](../REPRODUCE.md)). About 75 seconds
on the original machine.
