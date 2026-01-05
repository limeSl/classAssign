import io
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import pandas as pd
import streamlit as st

# =============================
# UI 기본
# =============================
st.set_page_config(page_title="반편성 도우미", page_icon="🏫", layout="wide")
st.title("🏫 반편성 도우미")
st.caption("엑셀 파일을 업로드 하시면 제가 반편성을 도와드립니다.")

# =============================
# 유틸
# =============================
def clean_name_korean_only(x) -> str:
    """이름에서 한글만 남기기 (영문/특수문자/공백/숫자 제거)"""
    if pd.isna(x):
        return ""
    return re.sub(r"[^가-힣]", "", str(x))

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def normalize_gender(x) -> str:
    """성별을 남/여로 최대한 정규화"""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in ["남", "남자", "m", "male", "man", "남성", "1"]:
        return "남"
    if s in ["여", "여자", "f", "female", "woman", "여성", "2"]:
        return "여"
    ko = re.sub(r"[^가-힣]", "", str(x))
    if "남" in ko and "여" not in ko:
        return "남"
    if "여" in ko and "남" not in ko:
        return "여"
    return str(x).strip()

def read_excel_all_sheets(uploaded_file) -> Dict[str, pd.DataFrame]:
    """Streamlit 업로드 파일 안정적으로 읽기(BytesIO + engine 지정)"""
    data = uploaded_file.getvalue()
    bio = io.BytesIO(data)
    bio.seek(0)
    return pd.read_excel(bio, sheet_name=None, engine="openpyxl")

def normalize_df_from_spec(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    새 형식 스펙(열 위치 고정) 기반 표준화.
    A:학년, B:반, C:번호, D:이름, E:생년월일, F:성별, G:점수, I:이전반
    (H는 무시)
    """
    # 최소 9열(A~I)
    if df.shape[1] < 9:
        raise ValueError(f"[{sheet_name}] 열이 부족합니다. I열(이전 반)까지 필요합니다. 현재 열 수: {df.shape[1]}")

    # 엑셀 행번호: (헤더 1행) + 데이터 시작 2행 가정
    excel_row = (df.index.to_series() + 2).astype(int)

    out = pd.DataFrame({
        "_sheet": sheet_name,
        "_excel_row": excel_row,
        "학년": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
        "반": df.iloc[:, 1].map(safe_str),
        "번호": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
        "이름(원본)": df.iloc[:, 3].map(safe_str),
        "이름(한글만)": df.iloc[:, 3].map(clean_name_korean_only),
        "생년월일": df.iloc[:, 4].map(safe_str),
        "성별": df.iloc[:, 5].map(normalize_gender),
        "점수": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
        "이전반_raw": df.iloc[:, 8].map(safe_str),
    })

    # 완전 빈 행 제거(반/번호/이름 모두 비어있으면 제거)
    out = out.dropna(how="all")
    out = out[~((out["반"] == "") & (out["번호"].isna()) & (out["이름(원본)"] == ""))]

    # UID(전역 유니크): sheet + excel_row
    out["_uid"] = out["_sheet"].astype(str) + ":" + out["_excel_row"].astype(str)

    return out

def format_prev_class_display(prev_raw: str) -> str:
    """표에서 보이는 이전반 표기: '5' -> '1-5'"""
    if prev_raw is None:
        return ""
    s = str(prev_raw).strip()
    if s == "":
        return ""
    nums = re.findall(r"\d+", s)
    if nums:
        return f"1-{nums[0]}"
    return f"1-{s}"

# =============================
# 조건 데이터 구조
# =============================
@dataclass
class Constraint:
    kind: str  # "묶기" or "떨어뜨리기"
    uids: List[str]  # 학생 uid 리스트

# Union-Find for 묶기 그룹
class UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

# =============================
# 조정 로직(휴리스틱)
# =============================
def build_blocks(df: pd.DataFrame, constraints: List[Constraint]):
    """묶기 조건을 union-find로 묶어서 '블록' 단위로 다룬다."""
    uids = df["_uid"].tolist()
    uf = UnionFind(uids)

    # 묶기: 선택된 uids 모두 같은 그룹
    for c in constraints:
        if c.kind != "묶기":
            continue
        if len(c.uids) < 2:
            continue
        first = c.uids[0]
        for other in c.uids[1:]:
            uf.union(first, other)

    # block_id -> members
    blocks: Dict[str, List[str]] = {}
    for uid in uids:
        root = uf.find(uid)
        blocks.setdefault(root, []).append(uid)

    # uid -> block_id
    uid_to_block = {uid: uf.find(uid) for uid in uids}

    # 떨어뜨리기 제약은 block 단위로 변환 (같은 block이면 애초에 불가능)
    not_same_edges: Set[Tuple[str, str]] = set()
    impossible = []
    for c in constraints:
        if c.kind != "떨어뜨리기":
            continue
        # 선택된 학생들끼리 pairwise로 같은 반 금지
        us = [uid_to_block[u] for u in c.uids]
        # 같은 블록 포함이면 모순
        if len(set(us)) < len(us):
            impossible.append(c)
            continue
        for i in range(len(us)):
            for j in range(i+1, len(us)):
                a, b = us[i], us[j]
                if a == b:
                    continue
                if a < b:
                    not_same_edges.add((a, b))
                else:
                    not_same_edges.add((b, a))

    return blocks, uid_to_block, not_same_edges, impossible

def compute_penalty(
    assignment: Dict[str, str],  # block_id -> class
    blocks: Dict[str, List[str]],
    df_index: Dict[str, Dict],  # uid -> info dict
    original_class: Dict[str, str],  # uid -> original class
    classes: List[str],
    not_same_edges: Set[Tuple[str, str]],
    size_min=19, size_max=21,
):
    """
    페널티 함수(작을수록 좋음)
    - 하드: 인원(19~21) 위반, 떨어뜨리기 위반
    - 소프트: 이동 최소, 평균 점수 분산 최소, 성별 불균형 최소
    """
    # 반별 집계
    cls_uids: Dict[str, List[str]] = {c: [] for c in classes}
    for bid, members in blocks.items():
        cls = assignment[bid]
        for uid in members:
            cls_uids[cls].append(uid)

    # 하드: 인원 위반
    hard = 0.0
    sizes = {}
    for c in classes:
        sz = len(cls_uids[c])
        sizes[c] = sz
        if sz < size_min:
            hard += (size_min - sz) * 1_000_000
        if sz > size_max:
            hard += (sz - size_max) * 1_000_000

    # 하드: 떨어뜨리기 위반 (같은 반이면 큰 페널티)
    # block 레벨이므로 block의 assigned class 비교
    for a, b in not_same_edges:
        if assignment.get(a) == assignment.get(b):
            hard += 2_000_000

    # 이동 수(소프트, 큰 가중치)
    moved = 0
    for bid, members in blocks.items():
        new_cls = assignment[bid]
        for uid in members:
            if original_class[uid] != new_cls:
                moved += 1
    move_pen = moved * 3000  # 이동 최소화 우선

    # 평균 점수 분산(소프트)
    means = []
    mean_pen = 0.0
    for c in classes:
        scores = [df_index[uid]["점수"] for uid in cls_uids[c] if df_index[uid]["점수"] is not None]
        scores = [s for s in scores if pd.notna(s)]
        if len(scores) == 0:
            continue
        means.append(sum(scores) / len(scores))
    if len(means) >= 2:
        mu = sum(means) / len(means)
        var = sum((m - mu) ** 2 for m in means) / len(means)
        mean_pen = var * 20000  # 평균 고르게

    # 성별 균형(소프트): |남-여| 합
    gender_pen = 0.0
    for c in classes:
        males = 0
        females = 0
        for uid in cls_uids[c]:
            g = df_index[uid]["성별"]
            if g == "남":
                males += 1
            elif g == "여":
                females += 1
        gender_pen += abs(males - females) * 5000

    return hard + move_pen + mean_pen + gender_pen

def adjust_classes(df: pd.DataFrame, constraints: List[Constraint], seed=7, steps=25000):
    """
    메인 조정 함수.
    - 묶기 블록화
    - 떨어뜨리기(같은반 금지) 반영
    - 휴리스틱 랜덤 탐색으로 페널티 최소화
    """
    random.seed(seed)

    # 기본 정보 인덱스
    df_index = {}
    original_class = {}
    for _, row in df.iterrows():
        uid = row["_uid"]
        df_index[uid] = {
            "점수": row["점수"],
            "성별": row["성별"],
        }
        original_class[uid] = row["반"]

    classes = sorted([c for c in df["반"].unique() if str(c).strip() != ""])
    if not classes:
        raise ValueError("반(B열) 값이 비어 있어 조정을 진행할 수 없습니다.")

    blocks, uid_to_block, not_same_edges, impossible = build_blocks(df, constraints)
    if impossible:
        # 떨어뜨리기 조건이 묶기 그룹 내부를 가리키는 경우
        raise ValueError("조건이 서로 모순입니다. '떨어뜨리기'가 '묶기'로 묶인 학생들을 포함하고 있습니다.")

    block_ids = list(blocks.keys())

    # 초기 배정: 블록의 '원본 반'에 최대한 유지(블록 내 최빈값)
    assignment: Dict[str, str] = {}
    for bid, members in blocks.items():
        # 원본 반 최빈값
        counts = {}
        for uid in members:
            oc = original_class[uid]
            counts[oc] = counts.get(oc, 0) + 1
        # 빈 반 값이 있으면 제외
        counts = {k: v for k, v in counts.items() if str(k).strip() != ""}
        if counts:
            best = max(counts.items(), key=lambda x: x[1])[0]
        else:
            best = classes[0]
        assignment[bid] = best

    # 페널티 계산
    best_assign = dict(assignment)
    best_pen = compute_penalty(best_assign, blocks, df_index, original_class, classes, not_same_edges)

    # 탐색: 블록을 다른 반으로 이동 / 블록간 swap
    # (인원 19~21은 하드로 걸어두었으니 결국 만족하는 방향으로 수렴)
    for t in range(steps):
        cur_pen = compute_penalty(assignment, blocks, df_index, original_class, classes, not_same_edges)

        # 랜덤 선택: move or swap
        if random.random() < 0.7:
            # move
            bid = random.choice(block_ids)
            cur_cls = assignment[bid]
            target = random.choice(classes)
            if target == cur_cls:
                continue
            assignment[bid] = target
            new_pen = compute_penalty(assignment, blocks, df_index, original_class, classes, not_same_edges)

            # accept if better or with small probability (탐색)
            if new_pen <= cur_pen or random.random() < 0.001:
                if new_pen < best_pen:
                    best_pen = new_pen
                    best_assign = dict(assignment)
            else:
                assignment[bid] = cur_cls
        else:
            # swap
            a, b = random.sample(block_ids, 2)
            ca, cb = assignment[a], assignment[b]
            if ca == cb:
                continue
            assignment[a], assignment[b] = cb, ca
            new_pen = compute_penalty(assignment, blocks, df_index, original_class, classes, not_same_edges)
            if new_pen <= cur_pen or random.random() < 0.001:
                if new_pen < best_pen:
                    best_pen = new_pen
                    best_assign = dict(assignment)
            else:
                assignment[a], assignment[b] = ca, cb

    # 최종 배정 uid -> new class
    uid_new_class = {}
    for bid, members in blocks.items():
        new_cls = best_assign[bid]
        for uid in members:
            uid_new_class[uid] = new_cls

    return uid_new_class

# =============================
# 세션 상태 초기화
# =============================
if "did_adjust" not in st.session_state:
    st.session_state.did_adjust = False
if "constraints" not in st.session_state:
    st.session_state.constraints: List[Constraint] = []
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "picker_reset" not in st.session_state:
    st.session_state.picker_reset = 0


# =============================
# 업로드 & 데이터 구성
# =============================
uploaded = st.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("엑셀 파일을 업로드하세요.")
    st.stop()

try:
    sheets = read_excel_all_sheets(uploaded)
except Exception as e:
    st.error("엑셀 파일을 읽는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

frames = []
bad_sheets = []
for sheet_name, raw in sheets.items():
    if raw is None:
        continue
    raw = raw.dropna(how="all")
    if raw.empty:
        continue
    try:
        frames.append(normalize_df_from_spec(raw.copy(), sheet_name))
    except Exception as e:
        bad_sheets.append((sheet_name, str(e)))

if not frames:
    st.error("처리 가능한 시트가 없습니다. (열 구조가 스펙과 맞는지 확인해주세요.)")
    if bad_sheets:
        with st.expander("시트별 오류 보기"):
            for n, m in bad_sheets:
                st.write(f"- {n}: {m}")
    st.stop()

df_all = pd.concat(frames, ignore_index=True)

# 웹 표시용: 학년 제외. (내부에는 학년 유지)
# 이름 토글/정렬 라디오
st.subheader("설정")
c1, c2 = st.columns([1, 2])
with c1:
    name_mode = st.radio("이름 표시", ["원본", "한글만"], horizontal=True)
with c2:
    sort_mode = st.radio("정렬 기준", ["번호순", "성적순"], horizontal=True)

display_name_col = "이름(한글만)" if name_mode == "한글만" else "이름(원본)"

view_base = df_all.copy()
view_base["이전반(표시)"] = view_base["이전반_raw"].map(format_prev_class_display)
view_base = view_base.rename(columns={display_name_col: "이름"})

# 정렬 반영
if sort_mode == "번호순":
    view_base = view_base.sort_values(by=["반", "번호"], ascending=[True, True], na_position="last")
else:
    view_base = view_base.sort_values(by=["반", "점수", "번호"], ascending=[True, False, True], na_position="last")

# 반 목록
classes = sorted([c for c in view_base["반"].unique() if str(c).strip() != ""])
if not classes:
    st.error("반(B열) 값이 비어 있습니다.")
    st.stop()

# ---- 업로드 직후(조정 전): 반 테이블만 표시 ----
st.subheader("📋 반별 학생 목록")

# view_base(정렬/이름표시 반영된 DF) 만들고 classes 구한 뒤
tabs = st.tabs([f"{c}반" for c in classes])
for tab, cls in zip(tabs, classes):
    with tab:
        df_cls = view_base[view_base["반"] == cls].copy()
        st.write(f"**인원:** {len(df_cls)}")
        # (평균점수는 여기서 표시하지 않음)
        st.dataframe(
            df_cls[["_excel_row", "반", "번호", "이름", "생년월일", "성별", "점수", "이전반(표시)"]]
            .rename(columns={"_excel_row": "엑셀행번호", "이전반(표시)": "이전반"}),
            use_container_width=True
        )

# =============================
# 조건 추가 UI (전체 학생 기준, 별도 검색창 없음)
# =============================
st.subheader("🧩 조건 추가(묶기 / 떨어뜨리기)")

with st.container(border=True):
    kind = st.radio("조건 종류", ["묶기", "떨어뜨리기"], horizontal=True, key="constraint_kind")

    # ✅ 전체 학생 기준 옵션 목록 만들기 (탭/반 무관)
    # - 표시 이름은 현재 name_mode(원본/한글만) 설정을 따름
    # - multiselect는 기본적으로 타이핑 검색 지원 (따로 검색창 불필요)
    base = view_base.copy()  # view_base는 df_all 기반으로 만들어져 있고, 이름/정렬 반영된 DF

    # 선택 라벨에 필요한 컬럼 준비
    # 이전반(표시), 이름, 성별, 점수, 현재반/번호를 함께 보여주기
    # uid -> label 매핑
    options = base["_uid"].tolist()
    uid_to_label = {}

    for _, r in base.iterrows():
        uid = r["_uid"]
        prev_disp = r.get("이전반(표시)", "")
        nm = r.get("이름", "")
        gender = r.get("성별", "")
        score = r.get("점수", None)

        # 점수 표시 포맷
        if pd.isna(score):
            score_txt = ""
        else:
            try:
                score_f = float(score)
                score_txt = str(int(score_f)) if score_f.is_integer() else str(score_f)
            except Exception:
                score_txt = str(score)

        cur_class = r.get("반", "")
        num = r.get("번호", "")
        num_txt = "" if pd.isna(num) else str(int(num)) if float(num).is_integer() else str(num)

        # ✅ 멀티셀렉트에서 검색하기 좋게: (이름) 앞쪽에 두고 정보 붙이기
        uid_to_label[uid] = f"{nm} | {prev_disp} | {gender} | 점수:{score_txt} | 현재 {cur_class}반 {num_txt}번"

    selected_uids = st.multiselect(
        "학생 선택 (여기서 바로 검색해서 선택하세요. 예: 이름 타이핑)",
        options=options,
        format_func=lambda x: uid_to_label.get(x, x),
        key=f"selected_uids_for_constraint_{st.session_state.picker_reset}"
    )

    add_btn = st.button("➕ 조건 추가", use_container_width=True)
    if add_btn:
        if len(selected_uids) < 2:
            st.warning("조건은 최소 2명 이상 선택해야 합니다.")
        else:
            st.session_state.constraints.append(Constraint(kind=kind, uids=list(selected_uids)))
            st.session_state.picker_reset += 1
            st.success(f"{kind} 조건 1개가 추가되었습니다. (대상 {len(selected_uids)}명)")
            st.rerun()

# 조건 목록 표시/삭제
st.subheader("📌 추가된 조건 목록")
if not st.session_state.constraints:
    st.info("아직 추가된 조건이 없습니다.")
else:
    for i, c in enumerate(st.session_state.constraints):
        with st.container(border=True):
            st.write(f"**#{i+1} {c.kind}** (대상 {len(c.uids)}명)")
            # 라벨로 표시
            lines = []
            for uid in c.uids:
                row = view_base[view_base["_uid"] == uid].head(1)
                if row.empty:
                    lines.append(uid)
                else:
                    r = row.iloc[0]
                    lines.append(f"- {r['이전반(표시)']} | {r['이름']} | {r['성별']} | 점수:{r['점수']}")
            st.write("\n".join(lines))
            if st.button("🗑️ 이 조건 삭제", key=f"del_{i}"):
                st.session_state.constraints.pop(i)
                st.rerun()

# =============================
# 조정 실행
# =============================
st.subheader("🛠️ 반편성 조정")

run = st.button("✅ 조정 누르기", type="primary", use_container_width=True)

if run:
    try:
        uid_new_class = adjust_classes(df_all, st.session_state.constraints, seed=7, steps=25000)

        result = df_all.copy()
        result["반_원본"] = result["반"]
        result["반"] = result["_uid"].map(uid_new_class)

        # 변경 여부
        result["변경"] = result["반"] != result["반_원본"]

        # 표시용 컬럼 구성(학년 제외, 시트 제외)
        result["이전반(표시)"] = result["이전반_raw"].map(format_prev_class_display)
        result_display = result.copy()
        # 이름 모드 반영
        if name_mode == "한글만":
            result_display["이름"] = result_display["이름(한글만)"]
        else:
            result_display["이름"] = result_display["이름(원본)"]

        # 정렬 반영
        if sort_mode == "번호순":
            result_display = result_display.sort_values(by=["반", "번호"], ascending=[True, True], na_position="last")
        else:
            result_display = result_display.sort_values(by=["반", "점수", "번호"], ascending=[True, False, True], na_position="last")

        st.session_state.result_df = result_display

        st.success("조정 완료! 아래에서 조정된 테이블을 확인하세요.")

    except Exception as e:
        st.error("조정 중 오류가 발생했습니다.")
        st.exception(e)

# =============================
# 결과 표시 + 다운로드
# =============================
if st.session_state.result_df is not None:
    res = st.session_state.result_df.copy()

    # 1) 설정
    st.subheader("설정(조정 결과 보기)")
    c1, c2 = st.columns([1, 2])
    with c1:
        name_mode_after = st.radio("이름 표시", ["원본", "한글만"], horizontal=True, key="name_mode_after")
    with c2:
        sort_mode_after = st.radio("정렬 기준", ["번호순", "성적순"], horizontal=True, key="sort_mode_after")

    # 설정 반영 (res는 내부적으로 원본/한글 이름 컬럼을 갖고 있어야 함)
    # 만약 res에 '이름(원본)', '이름(한글만)'이 없다면, 조정 시 result_df에 같이 포함시키도록 해야 함.
    if "이름(원본)" in res.columns and "이름(한글만)" in res.columns:
        res["이름"] = res["이름(한글만)"] if name_mode_after == "한글만" else res["이름(원본)"]

    if sort_mode_after == "번호순":
        res = res.sort_values(by=["반", "번호"], ascending=[True, True], na_position="last")
    else:
        res = res.sort_values(by=["반", "점수", "번호"], ascending=[True, False, True], na_position="last")

    # 2) 반별 테이블
    st.subheader("📋 반별 학생 목록(조정 결과)")
    def highlight_rows(row):
        # 은은한 반투명 오버레이 (다크/라이트 모두 무난)
        moved_bg = "background-color: rgba(255, 255, 255, 0.08);"      # 변경됨(살짝 밝게)
        constraint_bg = "background-color: rgba(0, 180, 255, 0.12);"   # 조건대상(차분한 청록)
        both_bg = "background-color: rgba(0, 180, 255, 0.12); box-shadow: inset 0 0 0 9999px rgba(255, 255, 255, 0.06);"  
        # ↑ 둘 다면 '조건색' 위에 아주 약한 밝기 오버레이를 한 겹 더

        changed = bool(row.get("변경", False))
        constrained = bool(row.get("조건대상", False))

        if changed and constrained:
            style = both_bg
        elif constrained:
            style = constraint_bg
        elif changed:
            style = moved_bg
        else:
            style = ""

        return [style] * len(row)

    classes2 = sorted([c for c in res["반"].unique() if str(c).strip() != ""])
    tabs = st.tabs([f"{c}반" for c in classes2])

    show_cols = ["_excel_row", "반_원본", "반", "번호", "이름", "생년월일", "성별", "점수", "이전반(표시)", "변경"]
    rename_map = {"_excel_row": "엑셀행번호", "반_원본": "원본반", "반": "조정반", "이전반(표시)": "이전반"}

    for tab, cls in zip(tabs, classes2):
        with tab:
            d = res[res["반"] == cls].copy()
            st.write(f"**인원:** {len(d)}")
            st.dataframe(dd.style.apply(highlight_rows, axis=1), use_container_width=True)

    # 3) 반별 평균점수(테이블 아래에서만 표시)
    st.subheader("📊 반별 평균점수(조정 후)")
    avg2 = (
        res.groupby("반")["점수"]
        .mean()
        .reset_index()
        .rename(columns={"점수": "평균점수"})
    )
    avg2["평균점수"] = avg2["평균점수"].round(2)
    st.dataframe(avg2.sort_values("반"), use_container_width=True)

    # =============================
    # 엑셀 다운로드 생성
    # =============================
    st.subheader("⬇️ 엑셀 다운로드")

    # 다운로드용 원본 데이터(학년 포함)로 재구성
    # 요구 형식: A학년, B반, C번호, D이름, E생년월일, F성별, G점수, H(빈칸), I이전반
    download_df = df_all.copy()
    # 조정된 반 반영
    # (result_df의 반이 조정 반이므로 uid 매핑)
    # st.session_state.result_df에는 반이 조정반으로 들어있음.
    uid_to_new = {uid: cls for uid, cls in zip(res["_uid"], res["반"])}
    download_df["반"] = download_df["_uid"].map(uid_to_new)

    # 출력 이름은 "원본 이름"으로 저장(파일은 사람이 읽는 원본이 보통 더 좋음)
    # 원하시면 한글만으로 바꾸는 옵션도 추가 가능
    download_df["H_빈칸"] = ""

    # 이전반은 파일에서는 원래 값(숫자/텍스트)을 유지
    out_cols = pd.DataFrame({
        "학년": download_df["학년"],
        "반": download_df["반"],
        "번호": download_df["번호"],
        "이름": download_df["이름(원본)"],
        "생년월일": download_df["생년월일"],
        "성별": download_df["성별"],
        "점수": download_df["점수"],
        "": download_df["H_빈칸"],         # H열 빈칸
        "이전 반": download_df["이전반_raw"],  # I열
    })

    # 반별 시트로 저장
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # 시트명: '2-5'처럼 만들기 (학년은 2로 가정하되, 학년 값이 있으면 첫 값 사용)
        # 학년 값이 시트마다 다르면 첫 유효값 사용
        default_grade = int(pd.to_numeric(download_df["학년"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(download_df["학년"], errors="coerce").dropna().shape[0] else 2

        for cls in sorted([c for c in out_cols["반"].unique() if str(c).strip() != ""]):
            sheet_df = out_cols[out_cols["반"] == cls].copy()
            # 정렬: 번호 오름차순
            sheet_df = sheet_df.sort_values(by="번호", ascending=True, na_position="last")
            sheet_name = f"{default_grade}-{cls}"
            # 엑셀 시트명 길이 제한(31)
            sheet_name = sheet_name[:31]
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)
    st.download_button(
        label="📥 조정된 반편성 엑셀 다운로드",
        data=buffer.getvalue(),
        file_name="반편성_조정결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
