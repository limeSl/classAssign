import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="반편성 도우미", page_icon="🏫", layout="wide")
st.title("🏫 반편성 도우미 (메인)")
st.caption("엑셀 업로드 → 시트(반)별 학생 목록 표시. 열 위치 고정(A~L), D열 성명은 한글만 남기도록 정제합니다.")

# -------------------------
# 유틸 함수
# -------------------------
KOREAN_ONLY_RE = re.compile(r"[^가-힣]")

def clean_korean_name(x) -> str:
    if pd.isna(x):
        return ""
    return KOREAN_ONLY_RE.sub("", str(x))

def normalize_gender(x) -> str:
    """
    다양한 성별 표기를 최대한 남/여로 통일.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()

    # 흔한 케이스들
    if s in ["남", "남자", "m", "male", "man", "남성", "1"]:
        return "남"
    if s in ["여", "여자", "f", "female", "woman", "여성", "2"]:
        return "여"

    # 예: "남/여", "남 " 등 애매한 입력은 한글만 남긴 후 부분 매칭
    s_ko = re.sub(r"[^가-힣]", "", str(x))
    if "남" in s_ko and "여" not in s_ko:
        return "남"
    if "여" in s_ko and "남" not in s_ko:
        return "여"

    return str(x).strip()

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    # Timestamp/날짜도 문자열로
    return str(x).strip()

def validate_min_columns(df: pd.DataFrame, min_cols: int = 12) -> bool:
    # A~L = 12열
    return df is not None and df.shape[1] >= min_cols

# -------------------------
# 업로드
# -------------------------
uploaded = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("엑셀 파일을 업로드하세요.")
    st.stop()

# -------------------------
# 엑셀 읽기
# -------------------------
try:
    sheets = pd.read_excel(uploaded, sheet_name=None, engine="openpyxl")
except Exception as e:
    st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

if not sheets:
    st.warning("엑셀에 시트가 없습니다.")
    st.stop()

# -------------------------
# 시트 처리
# -------------------------
processed = {}
errors = []

for sheet_name, raw_df in sheets.items():
    df = raw_df.copy()

    # 완전 빈 행 제거(전체가 NaN인 행)
    df = df.dropna(how="all")

    if df.empty:
        errors.append(f"'{sheet_name}': 데이터가 비어 있습니다.")
        continue

    if not validate_min_columns(df, 12):
        errors.append(f"'{sheet_name}': 열이 부족합니다. A~L(12열) 필요, 현재 {df.shape[1]}열.")
        continue

    # 열 인덱스 기반으로 안전하게 접근 (헤더명이 달라도 OK)
    # A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, J=9, K=10, L=11
    out = pd.DataFrame({
        "학년(A)": df.iloc[:, 0],
        "반(B)": df.iloc[:, 1],
        "번호(C)": df.iloc[:, 2],
        "성명_원본(D)": df.iloc[:, 3].map(safe_str),
        "성명_정제(한글만)": df.iloc[:, 3].map(clean_korean_name),
        "생년월일(E)": df.iloc[:, 4].map(safe_str),
        "생년월일(F)": df.iloc[:, 5].map(safe_str),
        "성별(G)": df.iloc[:, 6].map(normalize_gender),
        "성별(H)": df.iloc[:, 7].map(normalize_gender),
        "기준성적(I)": df.iloc[:, 8],
        "기준성적(J)": df.iloc[:, 9],
        "기준성적(K)": df.iloc[:, 10],
        "기준성적(L)": df.iloc[:, 11],
    })

    # 숫자열 정리(선택): 학년/반/번호는 가능하면 Int로 보이게
    for col in ["학년(A)", "반(B)", "번호(C)"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    processed[sheet_name] = out

# -------------------------
# 결과 표시
# -------------------------
if errors:
    st.warning("일부 시트를 처리하지 못했습니다. 아래 내용을 확인하세요.")
    for msg in errors:
        st.write(f"- {msg}")

if not processed:
    st.error("처리 가능한 시트가 없습니다. 엑셀 형식을 확인해주세요.")
    st.stop()

st.subheader("📋 반(시트)별 학생 목록")

sheet_names = list(processed.keys())
tabs = st.tabs(sheet_names)

for tab, sheet_name in zip(tabs, sheet_names):
    with tab:
        view_df = processed[sheet_name].copy()

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            hide_empty_name = st.checkbox("정제 성명이 빈 학생 숨기기", value=True, key=f"hide_empty_{sheet_name}")
        with c2:
            show_original_cols = st.checkbox("원본 성명/성별 열도 보기", value=False, key=f"show_orig_{sheet_name}")

        if hide_empty_name:
            view_df = view_df[view_df["성명_정제(한글만)"].astype(str).str.len() > 0]

        if not show_original_cols:
            drop_cols = ["성명_원본(D)"]
            # 필요하면 성별 원본 구분도 가능하지만 지금은 G/H 둘다 정규화된 값이므로 drop 없음
            view_df = view_df.drop(columns=drop_cols, errors="ignore")

        st.write(f"**시트명:** {sheet_name}  |  **학생 수:** {len(view_df)}")
        st.dataframe(view_df, use_container_width=True)
