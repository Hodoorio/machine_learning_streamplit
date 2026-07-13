"""
FIFA World Cup 예측 Streamlit 대시보드
======================================
1) CSV 업로드 → 2) reproduce_orange_logreg_auc.py 로직으로 예측
→ 3) 결과 CSV 저장 → 4) 저장된 CSV 읽어서 시각화
"""

import warnings, os

warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

# ── 설정 ──────────────────────────────────────────────────────────
st.set_page_config(page_title="FIFA WC Predictor", page_icon="⚽", layout="wide")

TRAIN_CSV = "./FIFAallMatchBoxData_with_result.csv"
OUTPUT_DIR = "/home/user1/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "result"
FEATURE_COLS = [
    "hPossesion",
    "aPossesion",
    "hshotsOnTarget",
    "ashotsOnTarget",
    "hshots",
    "ashots",
    "hredCards",
    "aredCards",
    "hyellowCards",
    "ayellowCards",
    "hfouls",
    "afouls",
]
ROW_FILTER_COL = "aPossesion"
ROW_FILTER_EXCLUDE_VALUE = 0.0
N_REPEATS = 10
TRAIN_SIZE = 0.9
RANDOM_STATE = 105


# ── 학습 데이터 로드 & 모델 학습 ─────────────────────────────────
@st.cache_data(show_spinner="학습 데이터 로드 및 모델 학습 중...")
def train_model():
    df = pd.read_csv(TRAIN_CSV)
    df = df[df[ROW_FILTER_COL] != ROW_FILTER_EXCLUDE_VALUE].reset_index(drop=True)
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values

    splitter = StratifiedShuffleSplit(
        n_splits=N_REPEATS, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    metric_keys = ["AUC", "CA", "F1", "Precision", "Recall", "MCC"]
    fold_metrics = {k: [] for k in metric_keys}
    fold_confusions = []

    for tr_idx, te_idx in splitter.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X[tr_idx], y[tr_idx])
        y_true = y[te_idx]
        preds = clf.predict(X[te_idx])
        proba = clf.predict_proba(X[te_idx])

        fold_metrics["AUC"].append(
            roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
        )
        fold_metrics["CA"].append(accuracy_score(y_true, preds))
        fold_metrics["F1"].append(
            f1_score(y_true, preds, average="weighted", zero_division=0)
        )
        fold_metrics["Precision"].append(
            precision_score(y_true, preds, average="weighted", zero_division=0)
        )
        fold_metrics["Recall"].append(
            recall_score(y_true, preds, average="weighted", zero_division=0)
        )
        fold_metrics["MCC"].append(matthews_corrcoef(y_true, preds))

        fold_confusions.append(confusion_matrix(y_true, preds, labels=clf.classes_))

    clf_final = LogisticRegression(C=1.0, max_iter=2000)
    clf_final.fit(X, y)

    # fold별 + 평균 메트릭 DataFrame
    metrics_df = pd.DataFrame(
        {k: fold_metrics[k] for k in metric_keys},
        index=[f"Fold {i}" for i in range(1, N_REPEATS + 1)],
    )
    metrics_df.loc["Mean"] = metrics_df.mean()

    return {
        "train_rows": len(df),
        "fold_metrics": fold_metrics,
        "metrics_df": metrics_df,
        "fold_confusions": fold_confusions,
        "classes": list(clf_final.classes_),
        "clf": clf_final,
    }


# ── 예측 → CSV 저장 ───────────────────────────────────────────────
def predict_and_save(clf, df_input, filename="predictions.csv"):
    X_pred = df_input[FEATURE_COLS].values.astype(float)
    proba = clf.predict_proba(X_pred)
    preds = clf.predict(X_pred)

    rows = []
    for i in range(len(df_input)):
        prob_dict = {c: float(p) for c, p in zip(clf.classes_, proba[i])}
        rows.append(
            {
                "Home": df_input.iloc[i]["hname"],
                "Away": df_input.iloc[i]["aname"],
                "Prediction": preds[i],
                "Confidence": float(proba[i].max()),
                **prob_dict,
            }
        )
    df_results = pd.DataFrame(rows)

    output_path = os.path.join(OUTPUT_DIR, filename)
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df_results, output_path


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
st.title("⚽ FIFA World Cup 예측 대시보드")

# ── 사이드바 ──────────────────────────────────────────────────────
st.sidebar.title("설정")
page = st.sidebar.radio("메뉴", ["1단계: 예측 실행"])

model_data = train_model()
clf = model_data["clf"]

# ── 사이드바: 기존 저장 파일 목록 ──────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("**저장된 예측 파일**")
saved_files = sorted(
    [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")], reverse=True
)
if saved_files:
    for f in saved_files:
        st.sidebar.caption(f)
else:
    st.sidebar.caption("(없음)")

# ══════════════════════════════════════════════════════════════════
# 1단계: 예측 실행
# ══════════════════════════════════════════════════════════════════
if page == "1단계: 예측 실행":
    st.header("1단계: CSV 업로드 → 예측 → 저장")

    # 학습 정보
    c1, c2 = st.columns(2)
    c1.metric("학습 데이터", f"{model_data['train_rows']:,}행")
    c2.metric("평균 AUC", f"{model_data['metrics_df'].loc['Mean', 'AUC']:.4f}")

    # 모델 메트릭 로그
    st.divider()
    st.subheader("모델 평가 메트릭 (Fold별)")
    st.dataframe(
        model_data["metrics_df"]
        .style.format("{:.4f}")
        .background_gradient(
            subset=["AUC", "CA", "F1", "Precision", "Recall", "MCC"], cmap="YlGnBu"
        ),
        use_container_width=True,
    )

    # 메트릭 평균 요약
    mean_m = model_data["metrics_df"].loc["Mean"]
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("AUC", f"{mean_m['AUC']:.4f}")
    mc2.metric("CA", f"{mean_m['CA']:.4f}")
    mc3.metric("F1", f"{mean_m['F1']:.4f}")
    mc4.metric("Precision", f"{mean_m['Precision']:.4f}")
    mc5.metric("Recall", f"{mean_m['Recall']:.4f}")
    mc6.metric("MCC", f"{mean_m['MCC']:.4f}")

    st.divider()

    # CSV 업로드
    uploaded = st.file_uploader(
        "예측 대상 CSV 파일 업로드",
        type=["csv"],
        help="hname, aname, 및 Feature 12개 컬럼이 포함된 CSV",
    )

    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
    else:
        st.info("CSV 파일을 업로드하지 않으면 기본 파일(2026 WC)이 사용됩니다.")
        default_path = "/home/user1/Downloads/2026 fifa world cup (without result).csv"
        df_input = pd.read_csv(default_path)

    # 필수 컬럼 체크
    required = FEATURE_COLS + ["hname", "aname"]
    missing = [c for c in required if c not in df_input.columns]
    if missing:
        st.error(f"CSV에 필수 컬럼이 없습니다: {missing}")
        st.stop()

    # 업로드된 CSV 미리보기
    with st.expander("업로드된 CSV 미리보기", expanded=False):
        st.dataframe(df_input, use_container_width=True)

    # 예측 실행 버튼
    if st.button("예측 실행 및 CSV 저장", type="primary", use_container_width=True):
        with st.spinner("예측 실행 중..."):
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{ts}.csv"
            df_results, output_path = predict_and_save(clf, df_input, filename)

        st.success(f"저장 완료: `{output_path}`")

        # 결과 미리보기
        st.subheader("예측 결과 미리보기")
        counts = df_results["Prediction"].value_counts()
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("총 경기", len(df_results))
        mc2.metric("홈 승리", counts.get("home_win", 0))
        mc3.metric("무승부", counts.get("draw", 0))
        mc4.metric("원정 승리", counts.get("away_win", 0))

        st.dataframe(
            df_results.style.format(
                {
                    "Confidence": "{:.1%}",
                    "home_win": "{:.3f}",
                    "draw": "{:.3f}",
                    "away_win": "{:.3f}",
                }
            ),
            use_container_width=True,
            height=400,
        )

        st.divider()
