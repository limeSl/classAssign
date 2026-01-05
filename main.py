import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="반편성 도우미", page_icon="🏫", layout="wide")
st.title("🏫 반편성 도우미")
st.caption("엑셀 업로드 → 반(B열) 기준 표 분리 / 번호순·점수순 정렬 / 이름 한글만 보기")

# -----------------------------
# 유틸
# -----------------------------
def clean_name_korean_only(x) -> str:
    """이름에서 한글만 남기기 (영문/특수문자/공백/숫자 제거)"""
    if pd.isna(x):
        return ""
    s = str(x)
    return re.sub(r"[^가-힣]", "", s)

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def read_excel_all_sheets(uploaded_file) -> dict[str, pd.DataFrame]:
    """Streamlit 업로드 파일을 가장 안정적으로 읽기(BytesIO + engine 지정)"""
    data = uploaded_file.getvalue()
    bio = io.BytesIO(data)
    bio.seek(0)
    return pd.read_excel(bio, sheet_name=None, engine="openpyxl")

def normalize_df_from_spec(df: pd.DataFrame) -> pd.DataFrame:
    """
    새 형식 스펙 기반으로 컬럼을 인덱스로 뽑아서 표준화.
    A:학년(표시 제외), B:반, C:번호, D:이름, E:생년월일, F:성별, G:점수, I:이전반
    """
    # 최소 9열(A~I)이 있어야 I(8)을 읽을 수 있음
    if df.shape[1] < 9:
        raise ValueError(f"열이 부족합니다. I열(이전 반)까지 필요합니다. 현재 열 수: {df.shape[1]}")

    out = pd.DataFrame({
        "반": df.iloc[:, 1].map(safe_str),
        "번호": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
        "이름(원본)": df.iloc[:, 3].map(safe_str),
        "이름(한글만)": df.iloc[:, 3].map(clean_name_korean_only),
        "생년월일": df.iloc[:, 4].map(safe_str),
        "성별": df.iloc[:, 5].map(safe_str),
        "점수": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
        "이전 반": df.iloc[:, 8].map(safe_str),
    })

    # 완전 빈 학생 행 제거(반/번호/이름 다 비면 제거)
    out = out.dropna(how="all")
    out = out[~((out["반"] == "") & (out["번호"].isna()) & (out["이름(원본)"] == ""))]

    return out

# -----------------------------
# 업로드
# -----------------------------
uploaded = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("엑셀 파일을 업로드하세요.")
    st.stop()

# -----------------------------
# 읽기 + 정규화(모든 시트 합치기)
# -----------------------------
try:
    sheets = read_excel_all_sheets(uploaded)
except Exception as e:
    st.error("엑셀 파일을 읽는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

normalized_frames = []
bad_sheets = []

for sheet_name, raw_df in sheets.items():
    # 완전 빈 시트 방지
    if raw_df is None or raw_df.dropna(how="all").empty:
        continue
    try:
        normalized = normalize_df_from_spec(raw_df.dropna(how="all").copy())
        normalized["시트"] = sheet_name  # 추적용
        normalized_frames.append(normalized)
    except Exception as e:
        bad_sheets.append((sheet_name, str(e)))

if not normalized_frames:
    st.error("처리 가능한 시트가 없습니다. (열 구조가 스펙과 맞는지 확인해주세요.)")
    if bad_sheets:
        with st.expander("시트별 오류 보기"):
            for name, msg in bad_sheets:
                st.write(f"- {name}: {msg}")
    st.stop()

all_df = pd.concat(normalized_frames, ignore_index=True)

# -----------------------------
# 옵션 UI
# -----------------------------
st.subheader("설정")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    name_mode = st.radio("이름 표시", ["원본", "한글만"], horizontal=True)
with c2:
    sort_mode = st.selectbox("정렬 기준", ["번호순", "점수순(내림차순)"])
with c3:
    available_classes = sorted([x for x in all_df["반"].unique() if x != ""])
    selected_classes = st.multiselect("표시할 반 선택(미선택 시 전체)", options=available_classes, default=[])

# 표시 이름 컬럼 결정
display_name_col = "이름(한글만)" if name_mode == "한글만" else "이름(원본)"

# 선택 반 필터
view_base = all_df.copy()
if selected_classes:
    view_base = view_base[view_base["반"].isin(selected_classes)]

# 정렬
if sort_mode == "번호순":
    view_base = view_base.sort_values(by=["반", "번호"], ascending=[True, True], na_position="last")
else:
    view_base = view_base.sort_values(by=["반", "점수", "번호"], ascending=[True, False, True], na_position="last")

# 표에서 보여줄 컬럼(학년 제외)
view_cols = ["반", "번호", display_name_col, "생년월일", "성별", "점수", "이전 반", "시트"]
view_base = view_base.rename(columns={display_name_col: "이름"})[["반", "번호", "이름", "생년월일", "성별", "점수", "이전 반", "시트"]]

# -----------------------------
# 반별 출력 (탭)
# -----------------------------
st.subheader("📋 반별 학생 목록")

classes = sorted([x for x in view_base["반"].unique() if x != ""])
if not classes:
    st.warning("선택 조건에 해당하는 반이 없습니다.")
    st.stop()

tabs = st.tabs(classes)
for tab, cls in zip(tabs, classes):
    with tab:
        df_cls = view_base[view_base["반"] == cls].copy()
        st.write(f"**{cls}반** | 인원: **{len(df_cls)}**")
        st.dataframe(df_cls, use_container_width=True)

# -----------------------------
# 참고: 시트 처리 오류 안내
# -----------------------------
if bad_sheets:
    with st.expander("⚠️ 일부 시트를 처리하지 못했어요(열 구조 불일치/열 부족 등)"):
        for name, msg in bad_sheets:
            st.write(f"- **{name}**: {msg}")
