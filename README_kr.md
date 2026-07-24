# 🧬 체외수정(IVF) 임신 성공 여부 예측 (LG Aimers 5기 해커톤)

## 📌 프로젝트 개요 및 배경
난임은 전 세계적으로 증가하는 중요한 의료 문제로, 많은 부부들이 오랜 기간 동안 신체적·정신적 부담을 겪고 있습니다. 난임 시술을 진행하는 환자들은 치료 과정에서 높은 비용과 심리적 스트레스를 경험하기 때문에, 최소한의 시술로 임신 성공 가능성을 높이는 것이 매우 중요합니다.

이러한 요구에 부응하기 위해 의료기관들은 인공지능(AI)을 활용한 임신 성공 여부 예측 모델에 큰 관심을 보이고 있습니다. AI 기반 솔루션은 방대한 난임 치료 데이터를 분석하여 최적의 의사결정을 지원하고, 환자 맞춤형 치료 계획을 수립하는 데 기여할 수 있습니다. 이는 환자의 시술 부담을 줄이는 동시에, 의료기관이 차별화된 서비스를 제공할 수 있도록 돕는 중요한 경쟁 요소가 될 것입니다.

이번 해커톤은 난임 환자 데이터를 활용하여 '임신 성공 여부'를 예측하고, 임신을 결정짓는 최적의 특성을 탐색하는 AI 모델 개발에 초점을 맞추고 있습니다. 이를 통해 실제 의료 데이터를 분석하고 예측 모델을 구축함으로써, 난임 치료의 효율성을 높이는 혁신적인 방안을 모색합니다.

## 📁 프로젝트 구조
```text
├── data/               # 원본 데이터 및 전처리된 데이터 (GitHub 업로드 제외)
├── notebooks/          # EDA 및 초기 실험용 주피터 노트북
├── src/                # 핵심 로직을 모듈화한 파이썬 스크립트
│   ├── preprocessing.py# 데이터 로드 및 결측치 처리
│   ├── features.py     # 도메인 지식 기반 파생 변수 생성
│   └── train.py        # 모델 학습 및 Optuna 하이퍼파라미터 튜닝
├── models/             # 학습된 모델 가중치 저장 폴더
├── requirements.txt    # 필요 라이브러리 목록
└── README.md           # 프로젝트 소개서
```

## 🛠️ 기술 스택
- **Data Processing & EDA**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning (Ensemble)**: `XGBoost`, `CatBoost`, `LightGBM`
- **Deep Learning**: `PyTorch` (Mac MPS 가속 활용)
- **Optimization**: `Optuna`

## 💡 핵심 엔지니어링 전략
1. **도메인 기반 피처 엔지니어링**
   - **임신 시도 대비 성공률**: 총 시술 횟수와 임신 횟수를 조합하여 환자별 실질적인 성공률 파생 변수 도출.
   - **연령대별 평균 매핑**: 연령대별 평균 임신 성공률을 변수로 추가하여 인구통계학적 패턴 반영.
2. **결측치 및 데이터 불균형 처리**
   - **결측치 대체**: 의미 없는 결측치를 `-1`로 일괄 처리하여 모델의 노이즈 감소.
   - **클래스 불균형 해결**: 타겟 변수의 불균형 문제를 해결하기 위해 양성 클래스에 가중치(`scale_pos_weight: 2.0`) 부여.
3. **정교한 모델 최적화**
   - **Optuna**를 활용하여 트리 깊이(`max_depth`), 반복 횟수(`iterations`) 등 앙상블 모델의 핵심 하이퍼파라미터 최적화.

## 🚀 실행 방법
```bash
git clone https://github.com/soonjae-dev/ivf-pregnancy-prediction.git
cd ivf-pregnancy-prediction
pip install -r requirements.txt
python src/train.py
```

## 👤 작성자
- **이순재** ([@soonjae-dev](https://github.com/soonjae-dev))
