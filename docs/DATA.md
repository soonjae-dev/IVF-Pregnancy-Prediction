# Data access and interpretation

## Obtain the data files

The project uses the LG Aimers 6th Hackathon competition data hosted on DACON.
Use the official [competition data page](https://dacon.io/competitions/official/236452/data)
and review the [competition rules](https://dacon.io/competitions/official/236452/overview/rules)
before downloading or reusing the files. Authentication or browser verification
may be required. The competition evaluates ROC-AUC and restricts training to the
provided training data; the models are trained on `train.csv` only.

Extract the downloaded CSVs and place them at these paths, relative to the
repository root:

```text
data/
├── train.csv
└── test.csv
```

Cross-validation (`baseline_cv.py`, `compare_missingness.py`, `improve_cv.py`)
needs only `train.csv`. `predict_test.py` additionally reads `test.csv` to write a
submission file. None of these needs a GAIN cache or saved OOF predictions.
To use another training-file location:

```bash
python baseline_cv.py --data /path/to/train.csv --sample 5000 --folds 3
```

Use the installation instructions in [README.md](../README.md) first. The
synthetic-data tests run without downloading the competition data.

## Checked local file

These are observations from the local CSV used for the recorded baseline, not
independently verified definitions from the official data dictionary.

| Item | Observed value |
|---|---|
| Rows | 256,351 |
| Columns | 69: identifier, target, and 67 input features |
| Identifier | `ID`; 256,351 distinct values, no missing values |
| Target | `임신 성공 여부`; integer values 0 and 1, no missing values |
| Target counts | 190,123 zeros; 66,228 ones (25.8349% positive) |
| Input types | 47 numeric and 20 object columns, as inferred by pandas 2.2.1 |
| File size | 84,117,328 bytes |
| SHA-256 | `205c73f52ce2f157916b9b75b4cfededb634702f67c7f1cdc153b37c9f1492e2` |

`baseline_cv.py` removes `ID` and the target before modeling. Input feature
counts above precede one-hot encoding. Numeric missing values are imputed with
training-fold medians; categorical missing values use `__MISSING__`, and category
vocabularies are fitted within each training fold. The hash identifies the exact
file bytes, so an equivalent CSV saved with different formatting can have a
different hash.

<details>
<summary>Input column names in the checked CSV (67 columns)</summary>

```text
시술 시기 코드
시술 당시 나이
임신 시도 또는 마지막 임신 경과 연수
시술 유형
특정 시술 유형
배란 자극 여부
배란 유도 유형
단일 배아 이식 여부
착상 전 유전 검사 사용 여부
착상 전 유전 진단 사용 여부
남성 주 불임 원인
남성 부 불임 원인
여성 주 불임 원인
여성 부 불임 원인
부부 주 불임 원인
부부 부 불임 원인
불명확 불임 원인
불임 원인 - 난관 질환
불임 원인 - 남성 요인
불임 원인 - 배란 장애
불임 원인 - 여성 요인
불임 원인 - 자궁경부 문제
불임 원인 - 자궁내막증
불임 원인 - 정자 농도
불임 원인 - 정자 면역학적 요인
불임 원인 - 정자 운동성
불임 원인 - 정자 형태
배아 생성 주요 이유
총 시술 횟수
클리닉 내 총 시술 횟수
IVF 시술 횟수
DI 시술 횟수
총 임신 횟수
IVF 임신 횟수
DI 임신 횟수
총 출산 횟수
IVF 출산 횟수
DI 출산 횟수
총 생성 배아 수
미세주입된 난자 수
미세주입에서 생성된 배아 수
이식된 배아 수
미세주입 배아 이식 수
저장된 배아 수
미세주입 후 저장된 배아 수
해동된 배아 수
해동 난자 수
수집된 신선 난자 수
저장된 신선 난자 수
혼합된 난자 수
파트너 정자와 혼합된 난자 수
기증자 정자와 혼합된 난자 수
난자 출처
정자 출처
난자 기증자 나이
정자 기증자 나이
동결 배아 사용 여부
신선 배아 사용 여부
기증 배아 사용 여부
대리모 여부
PGD 시술 여부
PGS 시술 여부
난자 채취 경과일
난자 해동 경과일
난자 혼합 경과일
배아 이식 경과일
배아 해동 경과일
```

</details>

## Interpretation still requiring verification

- A row and its unique `ID` are not established as a unique patient. Repeated
  patients, treatment cycles, clinic groups, and temporal dependence have not
  been verified. `source_row_index` in OOF files is a CSV row position, not a
  patient identifier.
- The model treats label 1 as positive. The clinical endpoint and ascertainment
  window behind “pregnancy success” remain unverified against the official data
  dictionary; do not equate the label with live birth or another clinical endpoint.
- Define the intended prediction time before claiming practical utility. Fields
  such as embryo counts, transferred embryos, and elapsed procedure days require
  an availability review for that time. Their presence does not establish a
  pretreatment prediction setting.
- Row-wise stratified cross-validation does not establish generalization to new
  patients, clinics, or future periods. Group/time splits require verified metadata.

The official data dictionary could not be inspected during this documentation
review because the data page required browser verification. These questions are
open rather than inferred from column names.

## Publication and terms

Raw competition files and row-level OOF prediction CSVs remain local and are
ignored by Git. Published results contain aggregate metrics and provenance.
Nothing in this repository grants rights to the competition data. Data access,
use, and redistribution remain subject to the provider's terms. No redistribution permission is asserted here.
