#!/usr/bin/env python3
"""
honest_cv.py — IVF 임신 성공 예측: 누수 없는 교차검증 평가

두 가지 프로토콜을 나란히 실행한다.

  [A] LEAKY   원본 main.py 재현: 전체 학습셋에 SMOTE를 먼저 적용한 뒤
              그 데이터를 통째로 CV에 넣는다. configs/*.json의 best_value가
              어떻게 나왔는지를 재현하기 위한 것이지, 보고용 수치가 아니다.

  [B] HONEST  StratifiedKFold(5). 각 폴드의 "학습 부분"에만 SMOTE를 적용하고
              검증 부분은 원본 분포 그대로 둔다. out-of-fold 확률을 모아
              OOF ROC-AUC를 계산한다.
              F1 임계값은 학습 부분을 다시 나눈 내부 보정셋에서만 고르고,
              그 임계값을 한 번도 보지 않은 검증 부분에 적용한다.

사용:
  python honest_cv.py --data <train_df_imputed.pkl | train.csv> --out results.json
"""

import argparse
import json
import os
import time
import warnings
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")

TARGET = "임신 성공 여부"
SEED = 42


# ---------------------------------------------------------------- utilities

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """전처리가 끝난 pkl 또는 원본 csv를 읽어 (X, y)로 돌려준다."""
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        df = df.fillna(-1)
        # 원본 csv 경로는 최소 인코딩만 적용 (pkl이 없을 때의 대비책)
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            df = pd.get_dummies(df, columns=obj_cols, dummy_na=False)

    if TARGET not in df.columns:
        raise SystemExit(f"타깃 컬럼 '{TARGET}' 을 찾을 수 없습니다. 컬럼: {list(df.columns)[:20]}")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)
    return X, y


def build_model(name: str, params: Dict[str, Any]):
    p = {k: v for k, v in params.items() if k != "best_value"}
    if name == "xgb":
        return xgb.XGBClassifier(random_state=SEED, eval_metric="logloss",
                                 n_jobs=-1, tree_method="hist", **p)
    if name == "lgb":
        return lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **p)
    if name == "cat":
        return cb.CatBoostClassifier(random_state=SEED, verbose=0,
                                     thread_count=-1, allow_writing_files=False, **p)
    if name == "rf":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **p)
    raise ValueError(name)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """F1을 최대화하는 임계값과 그때의 F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[idx]), float(f1[idx])


# ------------------------------------------------------- protocol A (leaky)

def run_leaky(X: pd.DataFrame, y: pd.Series, name: str, params: Dict[str, Any],
              n_splits: int = 5) -> Dict[str, Any]:
    """원본 파이프라인 재현: SMOTE를 CV 바깥에서 먼저 적용."""
    log(f"  [A/leaky] {name}: 전체 학습셋에 SMOTE 적용 후 CV")
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    log(f"            SMOTE 후 {X_sm.shape[0]:,}행 (원본 학습 {X_tr.shape[0]:,}행)")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    t0 = time.time()
    scores = cross_val_score(build_model(name, params), X_sm, y_sm,
                             cv=cv, scoring="roc_auc", n_jobs=1)
    log(f"            AUC {scores.mean():.4f} ± {scores.std():.4f}  ({time.time()-t0:.0f}s)")
    return {"auc_mean": float(scores.mean()), "auc_std": float(scores.std()),
            "folds": [float(s) for s in scores], "seconds": time.time() - t0}


# ------------------------------------------------------ protocol B (honest)

def run_honest(X: pd.DataFrame, y: pd.Series, name: str, params: Dict[str, Any],
               n_splits: int = 5) -> Dict[str, Any]:
    """SMOTE를 폴드 내부로. 임계값은 내부 보정셋에서만 결정."""
    log(f"  [B/honest] {name}: SMOTE를 각 폴드 학습부분 안에서만 적용")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    Xv, yv = X.values, y.values

    oof = np.zeros(len(y), dtype=np.float64)
    fold_auc, fold_f1, fold_thr = [], [], []
    t0 = time.time()

    for k, (tr_idx, va_idx) in enumerate(cv.split(Xv, yv), 1):
        ft = time.time()
        X_tr, y_tr = Xv[tr_idx], yv[tr_idx]
        X_va, y_va = Xv[va_idx], yv[va_idx]

        # 학습 부분을 다시 나눠 임계값 보정셋을 확보 (검증 부분은 손대지 않는다)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)

        # SMOTE는 fit 부분에만
        X_fit_sm, y_fit_sm = SMOTE(random_state=SEED).fit_resample(X_fit, y_fit)

        model = build_model(name, params)
        model.fit(X_fit_sm, y_fit_sm)

        # 임계값: 보정셋에서 결정 (SMOTE 미적용, 원본 분포)
        cal_proba = model.predict_proba(X_cal)[:, 1]
        thr, _ = best_f1_threshold(y_cal, cal_proba)

        # 평가: 한 번도 보지 않은 검증 폴드
        va_proba = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_proba
        auc = roc_auc_score(y_va, va_proba)
        f1 = f1_score(y_va, (va_proba >= thr).astype(int))

        fold_auc.append(float(auc)); fold_f1.append(float(f1)); fold_thr.append(float(thr))
        log(f"            fold {k}/{n_splits}  AUC {auc:.4f}  F1 {f1:.4f}  thr {thr:.3f}  ({time.time()-ft:.0f}s)")

    oof_auc = float(roc_auc_score(yv, oof))
    oof_ap = float(average_precision_score(yv, oof))
    log(f"            → OOF AUC {oof_auc:.4f} | AUC {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f} "
        f"| F1 {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}  ({time.time()-t0:.0f}s)")

    return {
        "oof_auc": oof_auc,
        "oof_average_precision": oof_ap,
        "auc_mean": float(np.mean(fold_auc)), "auc_std": float(np.std(fold_auc)),
        "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
        "threshold_mean": float(np.mean(fold_thr)),
        "fold_auc": fold_auc, "fold_f1": fold_f1, "fold_threshold": fold_thr,
        "seconds": time.time() - t0,
        "oof_proba": oof.tolist(),
    }


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--params-dir", default="configs")
    ap.add_argument("--out", default="cv_results.json")
    ap.add_argument("--models", default="xgb,lgb,cat,rf")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-leaky", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="행 샘플링 (타이밍 측정용)")
    args = ap.parse_args()

    log(f"데이터 로드: {args.data}")
    X, y = load_dataset(args.data)
    if args.sample:
        X, _, y, _ = train_test_split(X, y, train_size=args.sample, random_state=SEED, stratify=y)
        log(f"샘플링: {args.sample:,}행")
    pos = int(y.sum()); n = len(y)
    log(f"형태 {X.shape}, 양성 {pos:,}/{n:,} ({pos/n:.1%})")

    param_files = {"xgb": "xgb_params.json", "lgb": "lgb_params.json",
                   "cat": "cat_params.json", "rf": "rf_params.json"}
    results: Dict[str, Any] = {
        "dataset": {"rows": n, "features": int(X.shape[1]),
                    "positive_rate": pos / n, "source": os.path.basename(args.data)},
        "protocol": {
            "honest": f"StratifiedKFold({args.folds}), SMOTE inside training fold only, "
                      f"threshold tuned on inner 20% calibration split",
            "leaky": "SMOTE on full training set, then StratifiedKFold — reproduces original configs/*.json",
        },
        "models": {},
    }

    for name in args.models.split(","):
        name = name.strip()
        pf = os.path.join(args.params_dir, param_files[name])
        if not os.path.exists(pf):
            log(f"  ! {name}: 파라미터 파일 없음 ({pf}) — 건너뜀")
            continue
        params = json.load(open(pf))
        log(f"── {name} ── params={ {k:v for k,v in params.items() if k!='best_value'} }")
        entry: Dict[str, Any] = {"params": params,
                                 "recorded_best_value": params.get("best_value")}
        entry["honest"] = run_honest(X, y, name, params, args.folds)
        if not args.skip_leaky:
            entry["leaky"] = run_leaky(X, y, name, params, args.folds)
        results["models"][name] = entry
        # 중간 저장 — 오래 걸리는 작업이라 중단되어도 남기기
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- 가중 앙상블: OOF 확률을 그리드로 블렌딩 (동일 폴드 분할이라 정렬 일치)
    oofs = {k: np.array(v["honest"]["oof_proba"]) for k, v in results["models"].items()
            if "honest" in v}
    if len(oofs) >= 2:
        log("── 가중 앙상블 (OOF 확률 블렌딩) ──")
        names = list(oofs)
        best = (-1.0, None)
        rng = np.random.default_rng(SEED)
        for _ in range(2000):
            w = rng.dirichlet(np.ones(len(names)))
            blend = sum(w[i] * oofs[nm] for i, nm in enumerate(names))
            a = roc_auc_score(y.values, blend)
            if a > best[0]:
                best = (a, w)
        w = best[1]
        blend = sum(w[i] * oofs[nm] for i, nm in enumerate(names))
        thr, f1 = best_f1_threshold(y.values, blend)
        log(f"            weights={dict(zip(names, np.round(w,3)))}")
        log(f"            OOF AUC {best[0]:.4f}  F1 {f1:.4f} @ thr {thr:.3f}")
        results["ensemble_weighted"] = {
            "weights": {nm: float(w[i]) for i, nm in enumerate(names)},
            "oof_auc": float(best[0]), "oof_f1": float(f1), "threshold": float(thr),
            "note": "가중치와 임계값 모두 OOF 확률에서 골랐으므로 개별 모델 수치보다 낙관적임",
        }

    for v in results["models"].values():
        v["honest"].pop("oof_proba", None)
    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"저장: {args.out}")


if __name__ == "__main__":
    main()
