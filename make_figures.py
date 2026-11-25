#!/usr/bin/env python3
"""
make_figures.py — draw the README figures from saved result files.

Nothing is retrained. Each figure reads one JSON record already in results/:

    smote_leakage    results/smote_placement_20261114.json            (README 5-2)
    improvement      results/improve_20261115T064857_957864Z/...      (README 5-6)
    subgroups        results/error_analysis_20261115T162321Z/...      (README 5-8)

Usage:
    python make_figures.py              # English and Korean figures
    python make_figures.py --lang ko    # one language only

Output: docs/images/<name>.png (English) and docs/images/<name>.ko.png (Korean).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "images"

SMOTE_JSON = ROOT / "results" / "smote_placement_20261114.json"
IMPROVE_JSON = ROOT / "results" / "improve_20261115T064857_957864Z" / "comparison.json"
ERROR_JSON = ROOT / "results" / "error_analysis_20261115T162321Z" / "summary.json"

# Colours: one accent for the "honest" / gain series, one muted for the leaky one.
INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#d8dee4"
ACCENT = "#0969da"
WARN = "#cf222e"
NEUTRAL = "#8c959f"

KO_FONTS = ["AppleGothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans CJK JP",
            "NanumGothic", "Malgun Gothic"]

TEXT = {
    "en": {
        "smote_title": "Applying SMOTE before the split inflated CV AUC by about 0.18",
        "smote_x": "Cross-validation ROC-AUC (5-fold)",
        "smote_leaky": "SMOTE before split (leaky)",
        "smote_honest": "SMOTE inside each training fold",
        "smote_recorded": "Best score Optuna recorded in the competition",
        "smote_note": "Re-run of README 5-2. Both protocols reuse a GAIN cache fitted before the split.",
        "models": {"xgb": "XGBoost", "lgb": "LightGBM", "cat": "CatBoost", "rf": "Random Forest"},
        "imp_title": "Only tuning and blending raised AUC, and they did so in every fold",
        "imp_x": "Change in OOF ROC-AUC vs the previous step",
        "imp_pooled": "Pooled OOF",
        "imp_fold": "Each of 5 folds",
        "imp_folds_up": "{n}/5 folds up",
        "imp_note": "Steps are cumulative, top to bottom. Baseline 0.73922 → final blend 0.74034. One split seed (42).",
        "steps": {
            "native_nan": "Leave numeric gaps as NaN",
            "di_zero": "DI \"not applicable\" gaps as 0",
            "ordinal": "Count and age strings as ordered numbers",
            "features": "Eleven domain ratios",
            "lgb_tuned": "Nested Optuna tuning of LightGBM",
            "catboost": "CatBoost with native categories",
            "blend": "Rank average of LightGBM and CatBoost",
        },
        "sub_title": "The blend gains a little in almost every group and does not clearly lose in any",
        "sub_x": "AUC gain of the blend over the baseline (95% bootstrap interval)",
        "sub_overall": "All rows +{d:.4f}",
        "sub_note": "Same out-of-fold rows, no retraining. Groups with too few rows or positives are left out.",
        "axes": {
            "시술 유형": "Treatment",
            "시술 당시 나이": "Age",
            "난자 출처": "Egg source",
            "동결 배아 사용 여부": "Frozen embryo",
            "이식된 배아 수": "Embryos transferred",
        },
        "groups": {
            "DI": "DI", "IVF": "IVF",
            "만18-34세": "18–34", "만35-37세": "35–37", "만38-39세": "38–39",
            "만40-42세": "40–42", "만43-44세": "43–44", "만45-50세": "45–50",
            "본인 제공": "Own eggs", "기증 제공": "Donor eggs",
            "0": "No", "1": "Yes",
            "1개": "1", "2개": "2", "3개 이상": "3 or more",
        },
        "auc_label": "baseline AUC {a:.2f}",
    },
    "ko": {
        "smote_title": "분할 전 SMOTE 적용으로 교차검증 AUC가 0.18 가량 과대 추정됨",
        "smote_x": "교차검증 ROC-AUC (5-fold)",
        "smote_leaky": "분할 전 SMOTE (누수)",
        "smote_honest": "학습 fold 내부 SMOTE",
        "smote_recorded": "대회 당시 Optuna 최고 기록",
        "smote_note": "README 5-2 재측정 기준. 두 방식 모두 분할 전에 학습한 GAIN 캐시를 사용함",
        "models": {"xgb": "XGBoost", "lgb": "LightGBM", "cat": "CatBoost", "rf": "Random Forest"},
        "imp_title": "튜닝과 모델 결합만 5개 fold 모두에서 성능 향상",
        "imp_x": "직전 단계 대비 OOF ROC-AUC 변화",
        "imp_pooled": "전체 OOF",
        "imp_fold": "fold 5개 각각",
        "imp_folds_up": "{n}/5 fold 상승",
        "imp_note": "위에서부터 누적 적용. 베이스라인 0.73922 → 최종 결합 0.74034 (분할 seed 42 기준)",
        "steps": {
            "native_nan": "수치형 결측 NaN 유지",
            "di_zero": "DI '해당 없음' 결측 0 대체",
            "ordinal": "횟수·나이 문자열 순서형 변환",
            "features": "도메인 비율 변수 11개 추가",
            "lgb_tuned": "LightGBM 중첩 Optuna 튜닝",
            "catboost": "CatBoost (범주형 직접 처리)",
            "blend": "LightGBM·CatBoost 순위 평균",
        },
        "sub_title": "대부분의 집단에서 결합 모델 소폭 개선, 뚜렷한 성능 저하 집단 없음",
        "sub_x": "베이스라인 대비 결합 모델의 AUC 개선 (95% 부트스트랩 구간)",
        "sub_overall": "전체 +{d:.4f}",
        "sub_note": "재학습 없이 동일한 OOF 예측을 집단별로 비교함. 행 수나 양성 수가 적은 집단은 제외함",
        "axes": {
            "시술 유형": "시술 유형",
            "시술 당시 나이": "나이",
            "난자 출처": "난자 출처",
            "동결 배아 사용 여부": "동결 배아",
            "이식된 배아 수": "이식 배아 수",
        },
        "groups": {
            "DI": "DI", "IVF": "IVF",
            "만18-34세": "18–34세", "만35-37세": "35–37세", "만38-39세": "38–39세",
            "만40-42세": "40–42세", "만43-44세": "43–44세", "만45-50세": "45–50세",
            "본인 제공": "본인 난자", "기증 제공": "기증 난자",
            "0": "아니오", "1": "예",
            "1개": "1개", "2개": "2개", "3개 이상": "3개 이상",
        },
        "auc_label": "베이스라인 AUC {a:.2f}",
    },
}

# Subgroups shown in the forest plot, in order. "Unknown / no record" rows are the
# 6,291 DI rows again, so they are not repeated; small groups are dropped below.
SUBGROUPS = {
    "시술 유형": ["DI", "IVF"],
    "시술 당시 나이": ["만18-34세", "만35-37세", "만38-39세", "만40-42세", "만43-44세", "만45-50세"],
    "난자 출처": ["본인 제공", "기증 제공"],
    "동결 배아 사용 여부": ["0", "1"],
    "이식된 배아 수": ["1개", "2개", "3개 이상"],
}


def setup_style(lang: str) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "text.color": INK,
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.unicode_minus": False,
    })
    if lang == "ko":
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in KO_FONTS:
            if name in available:
                plt.rcParams["font.family"] = name
                break
        else:
            raise SystemExit("No Korean font found; install one of: " + ", ".join(KO_FONTS))
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"


def finish(fig, ax, title: str, note: str, path: Path) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def smote_leakage(t: dict, suffix: str) -> None:
    models = json.loads(SMOTE_JSON.read_text())["models"]
    order = ["lgb", "xgb", "rf", "cat"]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    for i, key in enumerate(order):
        m = models[key]
        leaky, honest = m["leaky"]["auc_mean"], m["honest"]["oof_auc"]
        recorded = m["recorded_best_value"]
        ax.plot([honest, leaky], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(honest, i, s=90, color=ACCENT, zorder=3, label=t["smote_honest"] if i == 0 else None)
        ax.scatter(leaky, i, s=90, color=WARN, zorder=3, label=t["smote_leaky"] if i == 0 else None)
        ax.scatter(recorded, i, s=70, marker="D", facecolor="white", edgecolor=WARN, lw=1.5, zorder=2,
                   label=t["smote_recorded"] if i == 0 else None)
        ax.text(honest - 0.004, i, f"{honest:.3f}", ha="right", va="center", color=ACCENT, fontsize=9.5)
        ax.text(leaky + 0.004, i, f"{leaky:.3f}", ha="left", va="center", color=WARN, fontsize=9.5)
        ax.text((honest + leaky) / 2, i + 0.28, f"−{leaky - honest:.3f}", ha="center", va="center",
                color=MUTED, fontsize=9)
    ax.set_yticks(range(len(order)), [t["models"][k] for k in order])
    ax.set_ylim(len(order) - 0.4, -0.5)
    ax.set_xlim(0.68, 0.95)
    ax.set_xlabel(t["smote_x"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    finish(fig, ax, t["smote_title"], t["smote_note"], OUT / f"smote_leakage{suffix}.png")


def improvement(t: dict, suffix: str) -> None:
    data = json.loads(IMPROVE_JSON.read_text())
    steps = [s for s in data["ladder"] if s != "baseline"]
    paired = data["paired_vs_previous"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, s in enumerate(steps):
        p = paired[s]
        pooled = p["pooled_auc_delta"]
        colour = ACCENT if pooled > 0 and p["positive_folds"] == 5 else NEUTRAL
        ax.barh(i, pooled, height=0.55, color=colour, alpha=0.35 if colour == NEUTRAL else 0.55,
                label=t["imp_pooled"] if i == 0 else None)
        ax.scatter(p["fold_auc_deltas"], [i] * 5, s=22, color=colour, zorder=3,
                   label=t["imp_fold"] if i == 0 else None)
        ax.text(0.00205, i, t["imp_folds_up"].format(n=p["positive_folds"]), va="center", ha="left",
                fontsize=9, color=INK if colour == ACCENT else MUTED)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(steps)), [t["steps"][s] for s in steps])
    ax.set_ylim(len(steps) - 0.4, -0.6)
    ax.set_xlim(-0.0010, 0.0026)
    ax.set_xlabel(t["imp_x"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.4f}" if v else "0"))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    finish(fig, ax, t["imp_title"], t["imp_note"], OUT / f"improvement_steps{suffix}.png")


def subgroups(t: dict, suffix: str) -> None:
    data = json.loads(ERROR_JSON.read_text())
    rows = []  # (label, delta, lo, hi, auc_baseline) or None for an axis header
    for axis, groups in SUBGROUPS.items():
        rows.append((t["axes"][axis], None, None, None, None))
        for g in groups:
            r = data["axes"][axis][g]
            if r["small"] or r["auc_delta"] is None:
                continue
            lo, hi = r["auc_delta_ci95"]
            label = f"{t['groups'][g]}  (n={r['rows']:,})"
            rows.append((label, r["auc_delta"], lo, hi, r["auc_baseline"]))
    overall = data["overall"]["auc_delta"]

    fig, ax = plt.subplots(figsize=(9, 0.33 * len(rows) + 1.6))
    labels = []
    for i, (label, d, lo, hi, base) in enumerate(rows):
        if d is None:
            labels.append(label)
            continue
        labels.append("    " + label)
        colour = ACCENT if lo > 0 else NEUTRAL
        ax.plot([lo, hi], [i, i], color=colour, lw=2, solid_capstyle="round")
        ax.scatter(d, i, s=36, color=colour, zorder=3)
        ax.text(0.0128, i, t["auc_label"].format(a=base), va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.axvline(0, color=INK, lw=0.8)
    ax.axvline(overall, color=ACCENT, lw=1, ls="--")
    ax.text(overall + 0.0002, -0.9, t["sub_overall"].format(d=overall), color=ACCENT, fontsize=9, ha="left")
    ax.set_yticks(range(len(rows)), labels)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row[1] is None:
            tick.set_fontweight("bold")
    ax.set_ylim(len(rows) - 0.4, -1.3)
    ax.set_xlim(-0.004, 0.0165)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.004))
    ax.set_xlabel(t["sub_x"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.3f}" if v else "0"))
    finish(fig, ax, t["sub_title"], t["sub_note"], OUT / f"subgroup_gains{suffix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["en", "ko", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in (["en", "ko"] if args.lang == "all" else [args.lang]):
        setup_style(lang)
        suffix = "" if lang == "en" else ".ko"
        t = TEXT[lang]
        smote_leakage(t, suffix)
        improvement(t, suffix)
        subgroups(t, suffix)


if __name__ == "__main__":
    main()
