"""
Synthetic data with the competition's schema, for end-to-end tests in CI.

Same 69 columns, types and category labels as the competition files, and the
same structural gaps documented in docs/experiments/MISSING_TYPES.md: DI rows
have no embryo or egg fields, flag columns record only 1 or blank, elapsed-day
columns are blank when their step did not happen, and a few gaps are random.
Values are drawn at random; no row of the real data is used or reproduced.
The target follows a simple made-up rule so the models have something to learn.
"""
import numpy as np
import pandas as pd

TARGET = "임신 성공 여부"
AGES = ["만18-34세", "만35-37세", "만38-39세", "만40-42세", "만43-44세", "만45-50세", "알 수 없음"]
COUNTS = ["0회", "1회", "2회", "3회", "4회", "5회", "6회 이상"]
CAUSES = ["남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인",
          "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
          "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
          "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
          "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
          "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태"]
COUNT_COLUMNS = ["총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
                 "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수",
                 "IVF 출산 횟수", "DI 출산 횟수"]
EMBRYO_COUNTS = ["총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
                 "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
                 "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
                 "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수"]
COLUMN_ORDER = (
    ["ID", "시술 시기 코드", "시술 당시 나이", "임신 시도 또는 마지막 임신 경과 연수",
     "시술 유형", "특정 시술 유형", "배란 자극 여부", "배란 유도 유형", "단일 배아 이식 여부",
     "착상 전 유전 검사 사용 여부", "착상 전 유전 진단 사용 여부"] + CAUSES
    + ["배아 생성 주요 이유"] + COUNT_COLUMNS
    + ["총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수", "이식된 배아 수",
       "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수",
       "해동 난자 수", "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
       "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수", "난자 출처", "정자 출처",
       "난자 기증자 나이", "정자 기증자 나이", "동결 배아 사용 여부", "신선 배아 사용 여부",
       "기증 배아 사용 여부", "대리모 여부", "PGD 시술 여부", "PGS 시술 여부",
       "난자 채취 경과일", "난자 해동 경과일", "난자 혼합 경과일", "배아 이식 경과일",
       "배아 해동 경과일", TARGET])


def make_frame(rows, seed=0, with_target=True, prefix="TRAIN"):
    rng = np.random.default_rng(seed)
    pick = lambda values, p=None: rng.choice(values, rows, p=p)
    di = rng.random(rows) < 0.05
    ivf = ~di
    age = pick(AGES, [0.39, 0.23, 0.15, 0.15, 0.05, 0.02, 0.01])
    d = {"ID": [f"{prefix}_{i:05d}" for i in range(rows)],
         "시술 시기 코드": pick(["TRDQAZ", "TRCMWS", "TRYBLT", "TRVNRY", "TRJXFG", "TRZKPL", "TRXQMD"]),
         "시술 당시 나이": age,
         "시술 유형": np.where(di, "DI", "IVF"),
         "배란 자극 여부": rng.integers(0, 2, rows),
         "배란 유도 유형": pick(["기록되지 않은 시행", "알 수 없음", "세트로타이드 (억제제)",
                           "생식선 자극 호르몬"]),
         "난자 출처": np.where(di, "알 수 없음", pick(["본인 제공", "기증 제공"], [0.94, 0.06])),
         "정자 출처": pick(["배우자 제공", "기증 제공", "미할당", "배우자 및 기증 제공"],
                       [0.9, 0.08, 0.01, 0.01]),
         "난자 기증자 나이": pick(["알 수 없음", "만31-35세", "만26-30세", "만21-25세", "만20세 이하"],
                            [0.9, 0.04, 0.03, 0.02, 0.01]),
         "정자 기증자 나이": pick(["알 수 없음", "만21-25세", "만36-40세", "만26-30세", "만31-35세",
                            "만41-45세", "만20세 이하"], [0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])}
    d["특정 시술 유형"] = np.where(di, pick(["IUI", "Generic DI"]),
                             pick(["ICSI", "IVF", "Unknown", "ICSI / BLASTOCYST ", "IVF:ICSI"]))
    years = rng.integers(0, 21, rows).astype(float)
    years[rng.random(rows) < 0.96] = np.nan
    d["임신 시도 또는 마지막 임신 경과 연수"] = years
    for column in CAUSES:
        d[column] = (rng.random(rows) < 0.15).astype(int)
    d["불임 원인 - 여성 요인"] = np.zeros(rows, dtype=int)
    for column in COUNT_COLUMNS:
        d[column] = pick(COUNTS, [0.4, 0.2, 0.15, 0.1, 0.07, 0.04, 0.04])

    def ivf_only(values):
        values = np.asarray(values, dtype=float)
        values[di] = np.nan
        return values

    for column in EMBRYO_COUNTS:
        d[column] = ivf_only(rng.poisson(3, rows))
    transferred = rng.choice([0, 1, 2, 3], rows, p=[0.15, 0.37, 0.44, 0.04])
    d["이식된 배아 수"] = ivf_only(transferred)
    d["미세주입 배아 이식 수"] = ivf_only(np.minimum(transferred, rng.integers(0, 4, rows)))
    frozen = rng.random(rows) < 0.16
    for column, values in [("단일 배아 이식 여부", transferred == 1),
                           ("착상 전 유전 진단 사용 여부", rng.random(rows) < 0.01),
                           ("동결 배아 사용 여부", frozen), ("신선 배아 사용 여부", ~frozen),
                           ("기증 배아 사용 여부", rng.random(rows) < 0.01),
                           ("대리모 여부", rng.random(rows) < 0.005)]:
        d[column] = ivf_only(values.astype(int))
    # Flags whose only recorded value is 1 (type 4).
    for column in ["착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부"]:
        d[column] = np.where(rng.random(rows) < 0.02, 1.0, np.nan)
    reasons = pick(["현재 시술용", "배아 저장용", "기증용, 현재 시술용", "난자 저장용"],
                   [0.85, 0.08, 0.04, 0.03]).astype(object)
    reasons[di] = np.nan
    d["배아 생성 주요 이유"] = reasons

    # Elapsed days: blank when the step did not happen (type 2), sometimes
    # blank anyway (type 3), always blank for DI (type 1).
    def days(done, high):
        values = rng.integers(0, high + 1, rows).astype(float)
        values[~done | di | (rng.random(rows) < 0.01)] = np.nan
        return values

    d["난자 채취 경과일"] = days(d["수집된 신선 난자 수"] > 0, 0)
    d["난자 해동 경과일"] = days(d["해동 난자 수"] > 0, 1)
    d["난자 혼합 경과일"] = days(d["혼합된 난자 수"] > 0, 7)
    d["배아 이식 경과일"] = days(transferred > 0, 7)
    d["배아 해동 경과일"] = days(frozen, 7)

    frame = pd.DataFrame(d)
    if with_target:
        young = np.isin(age, ["만18-34세", "만35-37세"])
        logit = -2.0 + 1.2 * (transferred > 0) + 0.8 * young - 0.4 * frozen + 0.3 * di
        logit = logit - 4.0 * (ivf & (transferred == 0))
        frame[TARGET] = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)
        return frame[COLUMN_ORDER]
    return frame[[c for c in COLUMN_ORDER if c != TARGET]]


def write_competition_files(directory, train_rows=1500, test_rows=300, seed=0):
    """Write train.csv and test.csv like the competition's and return their paths."""
    train = make_frame(train_rows, seed=seed)
    test = make_frame(test_rows, seed=seed + 1, with_target=False, prefix="TEST")
    train_path, test_path = directory / "train.csv", directory / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path
