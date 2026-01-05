import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="반편성 도우미", page_icon="🏫", layout="wide")
st.title("🏫 반편성 도우미 (메인)")
st.caption("엑셀 업로드 → 시트(반)별 학생 목록 표시")

# -------------------------
# 정제 함수
# -------------------------
def clean_korean_name(x) -> str:
    """성명에서 한글(가-힣)만 남겨서 이어붙임. 예: 'GAO ... (고운정)' -> '고운정'"""
    if pd.isna(x):
        return ""
    s = str(x)
    return "".join(re.findall(r"[가-힣]+", s))

def normalize_gender(x) -> str:
    """성별을 최대한 남/여로 정규화"""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in ["남", "남자", "m", "male", "man", "남성", "1"]:
        return "남"
    if s in ["여", "여자", "f", "female", "woman", "여성", "2"]:
        return "여"
    # 한글만 남겨 부분 매칭
    ko = re.sub(r"[^가-힣]", "", str(x))
    if "남" in ko and "여" not in ko:
        return "남"
    if "여" in ko and "남" not in ko:
        return "여"
    return str(x).strip()

def normalize_birth(x) -> str:
    """생년월일을 보이는 형태로 정리(가능하면 YYYY-MM-DD)"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # 예: 2012.04.10. / 2012-04-10 / 20120410 등 대응
    nums = re.findall(r"\d+", s)
    joined = "".join(nums)
    if len(joined) >= 8:
        y, m, d = joined[:4], joined[4:6], joined[6:8]
        return f"{y}-{m}-{d}"
    return s

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

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
    st.error("엑셀에 시트가 없습니다.")
    st.stop()

# -------------------------
# 파일 구조 보정 + 시트별 처리
# -------------------------
processed = {}
errors = []

for sheet_name, raw_df in sheets.items():
    df = raw_df.copy().dropna(how="all")  # 완전 빈 행 제거

    if df.empty:
        errors.append(f"'{sheet_name}': 데이터가 비어 있습니다.")
        continue

    # 필수 컬럼(이 파일 기준)
    required = ["학년", "반", "번호", "성명", "생년월일", "성별", "기준성적"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # 그래도 최대한 살려보기: 이름/학년 등이 컬럼명으로 없으면 인덱스 기반 시도 가능
        errors.append(f"'{sheet_name}': 필수 컬럼 누락: {', '.join(missing)} (엑셀 헤더 확인 필요)")
        # 일단 그대로 보여주고 다음 시트 진행
        processed[sheet_name] = df
        continue

    # --- (중요) '이전학적' 관련: 첫 행이 '학년/반/번호' 같은 헤더 역할을 하는 경우가 있음 ---
    prev_grade_col = "이전학적" if "이전학적" in df.columns else None

    # Unnamed 컬럼 중에 첫 행이 '반', '번호' 등인 것을 찾아서 이전학적 열로 사용
    prev_class_col = None
    prev_no_col = None
    for c in df.columns:
        if str(c).startswith("Unnamed"):
            v0 = safe_str(df.iloc[0][c])
            if v0 == "반":
                prev_class_col = c
            elif v0 == "번호":
                prev_no_col = c

    # 이전학적(학년/반/번호) 컬럼을 새로 만들어 담고, 첫 행이 헤더라면 제거
    df["이전학년"] = ""
    df["이전반"] = ""
    df["이전번호"] = ""

    first_row_looks_like_header = False
    if prev_grade_col and safe_str(df.iloc[0][prev_grade_col]) == "학년":
        first_row_looks_like_header = True

    if prev_grade_col:
        # 이전학년은 '이전학적' 컬럼 값이 실제로 '1학년' 등으로 들어감
        df["이전학년"] = df[prev_grade_col].map(safe_str)
    if prev_class_col:
        df["이전반"] = df[prev_class_col].map(safe_str)
    if prev_no_col:
        df["이전번호"] = df[prev_no_col].map(safe_str)

    # 첫 행이 '학년/반/번호' 헤더 역할이면 제거
    if first_row_looks_like_header:
        df = df.iloc[1:].copy()

    # 보기용 테이블 구성
    out = pd.DataFrame({
        "학년": df["학년"].map(safe_str),
        "반": df["반"],
        "번호": df["번호"],
        "성명(원본)": df["성명"].map(safe_str),
        "성명(한글만)": df["성명"].map(clean_korean_name),
        "생년월일": df["생년월일"].map(normalize_birth),
        "성별": df["성별"].map(normalize_gender),
        "기준성적": df["기준성적"],
        "이전학년": df["이전학년"].map(safe_str),
        "이전반": df["이전반"].map(safe_str),
        "이전번호": df["이전번호"].map(safe_str),
        "특이사항": df["특이사항"].map(safe_str) if "특이사항" in df.columns else "",
    })

    # 숫자 정리(가능하면 숫자로)
    for c in ["반", "번호", "기준성적"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    processed[sheet_name] = out

# -------------------------
# UI 출력
# -------------------------
with st.expander("📌 엑셀 형식 안내(이 파일 기준)", expanded=False):
    st.markdown(
        """
- 각 시트 = 한 반(예: Sheet2~Sheet8)
- 주요 열: **학년 / 반 / 번호 / 성명 / 생년월일 / 성별 / 기준성적 / (이전학적) / 특이사항**
- 이 파일은 병합/빈셀 때문에 `Unnamed:*` 열이 포함될 수 있으며,
  첫 데이터 행이 `이전학적(학년/반/번호)`의 헤더처럼 들어가 있어 자동으로 보정합니다.
- 성명은 **한글만 추출**합니다. 예: `GAO YUNQING (고운정)` → `고운정`
        """
    )

if errors:
    st.warning("일부 시트에서 형식 문제가 감지되었습니다.")
    for msg in errors:
        st.write(f"- {msg}")

st.subheader("📋 반(시트)별 학생 목록")
tabs = st.tabs(list(processed.keys()))

for tab, sheet_name in zip(tabs, processed.keys()):
    with tab:
        df_view = processed[sheet_name]

        # 만약 형식 오류로 원본 df가 들어간 경우도 있으니 분기
        if isinstance(df_view, pd.DataFrame) and "성명(한글만)" in df_view.columns:
            c1, c2 = st.columns([1, 2])
            with c1:
                hide_empty = st.checkbox("한글 성명이 빈 학생 숨기기", value=True, key=f"hide_{sheet_name}")
            with c2:
                show_original = st.checkbox("원본 성명도 보기", value=False, key=f"orig_{sheet_name}")

            if hide_empty:
                df_view = df_view[df_view["성명(한글만)"].astype(str).str.len() > 0]

            if not show_original:
                df_view = df_view.drop(columns=["성명(원본)"], errors="ignore")

            st.write(f"**시트명:** {sheet_name}  |  **학생 수:** {len(df_view)}")
            st.dataframe(df_view, use_container_width=True)
        else:
            st.warning("이 시트는 형식이 예상과 달라 원본 데이터 그대로 표시합니다.")
            st.dataframe(df_view, use_container_width=True)
