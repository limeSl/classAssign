import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="반편성 도우미", page_icon="🏫", layout="wide")
st.title("🏫 반편성 도우미")
st.caption("엑셀 파일을 업로드 하시면 제가 반편성을 도와드립니다.")

# -----------------------------
# 유틸
# -----------------------------
def clean_name_korean_only(x) -> str:
    """이름에서 한글만 남기기 (영문/특수문자/공백/숫자 제거)"""
    if pd.isna(x):
        return ""
    return re.sub(r"[^가-힣]", "", str(x))

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def format_prev_class(x) -> str:
    """
    이전 반: 5 -> 1-5 형식으로 보이게.
    빈 값이면 빈 문자열.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    # 숫자만 있는 경우 깔끔하게
    nums = re.findall(r"\d+", s)
    if nums:
        return f"1-{nums[0]}"
    return f"1-{s}"

def read_excel_all_sheets(uploaded_file) -> dict[str, pd.DataFrame]:
    """Streamlit 업로드 파일을 안정적으로 읽기(BytesIO + engine 지정)"""
    data = uploaded_file.getvalue()
    bio = io.BytesIO(data)
    bio.seek(0)
    return pd.read_excel(bio, sheet_name=None, engine="openpyxl")

def normalize_df_from_spec(df: pd.DataFrame) -> pd.DataFrame:
    """
    새 형식 스펙(열 위치 고정) 기반 표준화.
    A:학년(표시 제외), B:반, C:번호, D:이름, E:생년월일, F:성별, G:점수, I:이전반
    """
    # 최소 9열(A~I)이 있어야 I(8)을 읽을 수 있음
    if df.shape[1] < 9:
        raise ValueError(f"열이 부족합니다. I열(이전 반)까지 필요합니다. 현재 열 수: {df.shape[1]}")

    # 엑셀 행번호: pandas index(0부터) + 2 (헤더 1행 + 데이터 시작 2행 가정)
    excel_row = (df.index.to_series() + 2).astype(int)

    out = pd.DataFrame({
        "엑셀행번호": excel_row,
        "반": df.iloc[:, 1].map(safe_str),
        "번호": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
        "이름(원본)": df.iloc[:, 3].map(safe_str),
        "이름(한글만)": df.iloc[:, 3].map(clean_name_korean_only),
        "생년월일": df.iloc[:, 4].map(safe_str),
        "성별": df.iloc[:, 5].map(safe_str),
        "점수": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
        "이전반": df.iloc[:, 8].map(format_prev_class),
    })

    # 완전 빈 학생 행 제거(반/번호/이름이 전부 비어 있으면 제거)
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
    if raw_df is None:
        continue
    raw_df = raw_df.dropna(how="all")
    if raw_df.empty:
        continue
    try:
        normalized_frames.append(normalize_df_from_spec(raw_df.copy()))
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

c1, c2 = st.columns([1, 2])
with c1:
    name_mode = st.radio("이름 표시", ["원본", "한글만"], horizontal=True)
with c2:
    sort_mode = st.radio("정렬 기준", ["번호순", "성적순"], horizontal=True)

display_name_col = "이름(한글만)" if name_mode == "한글만" else "이름(원본)"

# 표시용 컬럼 구성(시트 컬럼 없음)
view_df = all_df.copy()
view_df = view_df.rename(columns={display_name_col: "이름"})[
    ["엑셀행번호", "반", "번호", "이름", "생년월일", "성별", "점수", "이전반"]
]

# 정렬
if sort_mode == "번호순":
    view_df = view_df.sort_values(by=["반", "번호"], ascending=[True, True], na_position="last")
else:
    # 성적순: 점수 내림차순, 동점이면 번호 오름차순
    view_df = view_df.sort_values(by=["반", "점수", "번호"], ascending=[True, False, True], na_position="last")

# -----------------------------
# 반별 평균점수 (전체 요약)
# -----------------------------
st.subheader("📊 반별 평균점수")

avg_df = (
    view_df.groupby("반", dropna=False)["점수"]
    .mean()
    .reset_index()
    .rename(columns={"점수": "평균점수"})
)

# 반 이름이 빈 경우 제외
avg_df = avg_df[avg_df["반"].astype(str).str.strip() != ""]
avg_df["평균점수"] = avg_df["평균점수"].round(2)

st.dataframe(avg_df.sort_values("반"), use_container_width=True)

# -----------------------------
# 반별 출력 (탭)
# -----------------------------
st.subheader("📋 반별 학생 목록")

classes = sorted([x for x in view_df["반"].unique() if str(x).strip() != ""])
if not classes:
    st.warning("반(B열) 값이 비어 있어 반별로 나눌 수 없습니다.")
    st.stop()

tabs = st.tabs([f"{c}반" for c in classes])

for tab, cls in zip(tabs, classes):
    with tab:
        df_cls = view_df[view_df["반"] == cls].copy()

        # 평균 점수(탭 상단)
        mean_score = df_cls["점수"].mean()
        mean_text = "—" if pd.isna(mean_score) else f"{mean_score:.2f}"

        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric(label="평균점수", value=mean_text)
        with c2:
            st.write(f"**인원:** {len(df_cls)}")

        st.dataframe(df_cls, use_container_width=True)

# -----------------------------
# 참고: 시트 처리 오류 안내
# -----------------------------
if bad_sheets:
    with st.expander("⚠️ 일부 시트를 처리하지 못했어요(열 부족/형식 불일치 등)"):
        for name, msg in bad_sheets:
            st.write(f"- **{name}**: {msg}")
