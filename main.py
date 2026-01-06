import io
import re
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

import pandas as pd
import streamlit as st
# =============================
# UI 기본
# =============================
st.set_page_config(page_title="반편성 도우미", page_icon="🏫", layout="wide")

if "picker_reset" not in st.session_state:
    st.session_state.picker_reset = 0

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

def render_class_tabs(
    df: pd.DataFrame,
    title: str,
    class_col: str = "반",
    show_changed: bool = False,
    highlight_func=None,
    table_cols=None,
    rename_map=None,
    hide_cols=None,
):
    """
    반별 탭 + (상단) 인원/성비/반평균 + 표 출력 공통 렌더러
    - df: 표시할 DF (반/성별/점수 컬럼 포함 권장)
    - highlight_func: pandas Styler 행 스타일 함수 (axis=1)
    - table_cols: 표에 포함할 컬럼(스타일 판단용 컬럼도 포함 가능)
    - rename_map: 표 표시용 컬럼명 매핑
    - hide_cols: 표에서는 숨기되(style 판단에는 남기고 싶은) 컬럼명 리스트 (rename 이후 이름 기준)
    """
    st.subheader(title)

    classes = sorted([c for c in df[class_col].unique() if str(c).strip() != ""])
    if not classes:
        st.warning("반 정보가 없어 반별로 표시할 수 없습니다.")
        return

    tabs = st.tabs([f"{c}반" for c in classes])

    for tab, cls in zip(tabs, classes):
        with tab:
            d = df[df[class_col] == cls].copy()

            # ---- 상단 요약: 인원 / 성비 / 평균점수 ----
            n = len(d)
            m = int((d["성별"] == "남").sum()) if "성별" in d.columns else 0
            f = int((d["성별"] == "여").sum()) if "성별" in d.columns else 0
            diff = abs(m - f)
            mean_score = d["점수"].mean() if "점수" in d.columns else None
            mean_text = "—" if mean_score is None or pd.isna(mean_score) else f"{mean_score:.2f}"

            c1, c2, c3 = st.columns(3)
            c1.metric("인원", n)
            c2.metric("성비(남/여)", f"{m}/{f}")
            c3.metric("반 평균점수", mean_text)

            # ---- 표 준비 ----
            if table_cols is None:
                table_cols = [col for col in d.columns if not col.startswith("_")]

            existing_cols = [c for c in table_cols if c in d.columns]
            missing_cols = [c for c in table_cols if c not in d.columns]
            if missing_cols:
                st.warning(f"표시용 컬럼 누락: {missing_cols}")

            out = d[existing_cols].copy()

            if rename_map:
                out = out.rename(columns=rename_map)

            hide_cols = hide_cols or []
            hide_cols_present = [c for c in hide_cols if c in out.columns]

            # ---- 표 출력 (스타일 적용 + 숨김 지원) ----
            if show_changed and highlight_func is not None:
                styled = out.style.apply(highlight_func, axis=1)
                # pandas/streamlit 환경에 따라 hide 지원 여부가 달라서 try 처리
                try:
                    styled = styled.hide(columns=hide_cols_present)
                    st.dataframe(styled, use_container_width=True)
                except Exception:
                    # hide가 안 되면 표시에서만 drop (이 경우 숨긴 컬럼을 스타일에서 못 쓰게 됨)
                    st.dataframe(out.drop(columns=hide_cols_present), use_container_width=True)
            else:
                st.dataframe(out.drop(columns=hide_cols_present), use_container_width=True)

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
import math
from typing import Dict, List, Tuple, Set, Optional

def block_stats(block_id: str, blocks: Dict[str, List[str]], df_index: Dict[str, Dict]):
    """블록의 (남수, 여수, 평균점수) 계산"""
    m = f = 0
    scores = []
    for uid in blocks[block_id]:
        g = df_index[uid]["성별"]
        if g == "남":
            m += 1
        elif g == "여":
            f += 1
        s = df_index[uid]["점수"]
        if s is not None and not (isinstance(s, float) and math.isnan(s)):
            scores.append(float(s))
    avg = sum(scores) / len(scores) if scores else None
    return m, f, avg

def class_counts_from_assignment(
    assignment: Dict[str, str],
    blocks: Dict[str, List[str]],
    df_index: Dict[str, Dict],
    classes: List[str],
):
    """반별 (인원, 남, 여) 집계"""
    cnt = {c: {"n": 0, "m": 0, "f": 0} for c in classes}
    for bid, cls in assignment.items():
        for uid in blocks[bid]:
            cnt[cls]["n"] += 1
            g = df_index[uid]["성별"]
            if g == "남":
                cnt[cls]["m"] += 1
            elif g == "여":
                cnt[cls]["f"] += 1
    return cnt

def check_hard_rules(cnt, size_min=19, size_max=21, gender_diff_max=2) -> bool:
    for _, v in cnt.items():
        if not (size_min <= v["n"] <= size_max):
            return False
        if abs(v["m"] - v["f"]) > gender_diff_max:
            return False
    return True

def violates_not_same(assignment: Dict[str, str], not_same_edges: Set[Tuple[str, str]]) -> List[Tuple[str, str]]:
    bad = []
    for a, b in not_same_edges:
        if assignment.get(a) == assignment.get(b):
            bad.append((a, b))
    return bad

def score_distance(stats_cache, bid1: str, bid2: str) -> float:
    """성적 유사도(작을수록 좋음). 점수 없으면 큰 값."""
    _, _, a1 = stats_cache[bid1]
    _, _, a2 = stats_cache[bid2]
    if a1 is None or a2 is None:
        return 10_000_000.0
    return abs(a1 - a2)

def _swap_and_check(
    assignment: Dict[str, str],
    bid1: str,
    bid2: str,
    blocks: Dict[str, List[str]],
    df_index: Dict[str, Dict],
    classes: List[str],
    not_same_edges: Set[Tuple[str, str]],
    size_min: int,
    size_max: int,
    gender_diff_max: int,
) -> Tuple[bool, bool, bool]:
    """
    swap 후:
    - hard_ok: 인원/성비 하드 규칙 만족?
    - edges_ok: not_same 위반이 '줄어드는' 방향?
    - any_edges_violation: 현재 위반 존재 여부(디버그)
    """
    before_bad = violates_not_same(assignment, not_same_edges)
    ca, cb = assignment[bid1], assignment[bid2]
    assignment[bid1], assignment[bid2] = cb, ca

    cnt = class_counts_from_assignment(assignment, blocks, df_index, classes)
    hard_ok = check_hard_rules(cnt, size_min, size_max, gender_diff_max)

    after_bad = violates_not_same(assignment, not_same_edges)
    edges_ok = (len(after_bad) < len(before_bad))

    assignment[bid1], assignment[bid2] = ca, cb
    return hard_ok, edges_ok, (len(after_bad) > 0)
def expand_movable_candidates(
    df,
    blocks,
    assignment,
    df_index,
    movable_blocks,
    move_bid,
    classes,
    k_per_class=2,
    score_window=30.0,
):
    """
    swap/move+fix가 막힐 때, '보정 후보군'을 임시 movable로 확장
    - 같은 성별 우선
    - 점수 차가 score_window 이내인 학생들 우선
    - 각 반에서 k_per_class명씩만 추가
    """
    m1, f1, avg1 = block_stats(move_bid, blocks, df_index)
    # move_bid가 1명 블록일 때 성별을 기준으로 잡기
    target_gender = None
    for uid in blocks[move_bid]:
        g = df_index[uid]["성별"]
        if g in ("남", "여"):
            target_gender = g
            break

    if avg1 is None:
        avg1 = 0.0

    extra = set()

    for cls in classes:
        # 현재 cls 반에 있는 블록들
        bids_in_cls = [bid for bid, c in assignment.items() if c == cls and bid not in movable_blocks and bid != move_bid]
        scored = []
        for bid in bids_in_cls:
            m2, f2, avg2 = block_stats(bid, blocks, df_index)
            # 성별 우선(블록 단위라 완벽치 않지만, 1명 블록이면 거의 맞음)
            if target_gender == "남" and f2 > 0 and m2 == 0:
                continue
            if target_gender == "여" and m2 > 0 and f2 == 0:
                continue

            if avg2 is None:
                continue

            dist = abs(avg2 - avg1)
            if dist <= score_window:
                scored.append((dist, bid))

        scored.sort(key=lambda x: x[0])
        for _, bid in scored[:k_per_class]:
            extra.add(bid)

    return movable_blocks | extra

def adjust_classes_min_change_swap_only_v2(
    df: pd.DataFrame,
    constraints: List[Constraint],
    blocks: Dict[str, List[str]],
    uid_to_block: Dict[str, str],
    not_same_edges: Set[Tuple[str, str]],
    size_min=19,  # UNUSED (kept for backward compatibility)
    size_max=21,  # UNUSED
    gender_diff_max=2,  # UNUSED
    relax_gender_swap=True,  # UNUSED
    max_iters=5000,
    seed=7,
    max_cycle_len: int = 4,
    candidates_per_iter: int = 60,
):
    """
    v3 엔진(단순/안정):
    - 인원/성비/평균점수 균등화 같은 '하드 규칙'은 고려하지 않습니다.
    - 오직 사용자가 추가한 조건(묶기/떨어뜨리기)만 만족하도록 조정합니다.
    - 반 인원수/성비가 "계속 유지"되도록, '블록(묶기 그룹)'의 성별구성(남/여 인원)이 동일한 블록끼리만 교환합니다.
      -> 1:1 교환(swap) + 필요 시 3~4개 블록 사이클 교환(cycle)을 지원합니다.
    - 절대 None을 반환하지 않습니다. (성공 return / 실패 raise)
    """

    random.seed(seed)
    diag_lines: List[str] = []

    # 불가능 조건(묶기 내부에 떨어뜨리기)이면 바로 실패
    # build_blocks에서 impossible을 이미 체크했을 수도 있으나, 안전을 위해 재검증
    for a, b in list(not_same_edges):
        if a == b:
            raise ValueError("조건 모순: 같은 블록을 떨어뜨리기로 지정했습니다.")

    # df_index / 원본반
    df_index = {}
    original_class_uid = {}
    for _, r in df.iterrows():
        uid = r["_uid"]
        df_index[uid] = {"성별": r.get("성별", ""), "점수": r.get("점수", None)}
        original_class_uid[uid] = r.get("반", "")

    classes = sorted([c for c in df["반"].unique() if str(c).strip() != ""])
    if not classes:
        raise ValueError("반(B열) 값이 비어 있어 조정을 진행할 수 없습니다.")

    # 초기 배정: 블록을 원본 반 최빈값으로
    assignment: Dict[str, str] = {}
    for bid, members in blocks.items():
        counts = {}
        for uid in members:
            oc = original_class_uid.get(uid, "")
            counts[oc] = counts.get(oc, 0) + 1
        counts = {k: v for k, v in counts.items() if str(k).strip() != ""}
        assignment[bid] = max(counts.items(), key=lambda x: x[1])[0] if counts else classes[0]

    # ----- helpers -----
    def violates_edges(assign: Dict[str, str]) -> List[Tuple[str, str, str]]:
        bad = []
        for a, b in not_same_edges:
            if assign.get(a) == assign.get(b):
                bad.append((a, b, assign.get(a)))
        return bad

    def block_gender_signature(bid: str) -> Tuple[int, int]:
        m = f = 0
        for uid in blocks[bid]:
            g = df_index[uid]["성별"]
            if g == "남":
                m += 1
            elif g == "여":
                f += 1
        return (m, f)

    def block_avg_score(bid: str) -> Optional[float]:
        scores = []
        for uid in blocks[bid]:
            s = df_index[uid]["점수"]
            if s is None or pd.isna(s):
                continue
            scores.append(float(s))
        if not scores:
            return None
        return sum(scores) / len(scores)

    gender_sig = {bid: block_gender_signature(bid) for bid in blocks.keys()}
    avg_score = {bid: block_avg_score(bid) for bid in blocks.keys()}

    def score_dist(a: str, b: str) -> float:
        av = avg_score.get(a, None)
        bv = avg_score.get(b, None)
        if av is None or bv is None:
            return 1e9
        return abs(av - bv)

    def apply_swap(assign: Dict[str, str], x: str, y: str):
        assign[x], assign[y] = assign[y], assign[x]

    def apply_cycle(assign: Dict[str, str], bids: List[str]):
        # b1<-b2, b2<-b3, ..., last<-b1
        old = [assign[b] for b in bids]
        for i in range(len(bids) - 1):
            assign[bids[i]] = old[i + 1]
        assign[bids[-1]] = old[0]

    # ----- main -----
    bad = violates_edges(assignment)
    if not bad:
        return assignment, "이미 모든 '떨어뜨리기' 조건을 만족합니다."

    diag_lines.append(f"- initial violations: {len(bad)}")
    block_ids = list(blocks.keys())

    for it in range(max_iters):
        bad = violates_edges(assignment)
        if not bad:
            diag_lines.append(f"- OK: all constraints satisfied at iter={it}")
            return assignment, "\n".join(diag_lines)

        a, b, cls = bad[0]

        solved = False

        # 1) 1:1 swap 시도 (a 또는 b를 다른 반으로 보내는 스왑)
        for pivot in (a, b):
            cur_cls = assignment[pivot]

            candidates = [
                bid for bid in block_ids
                if bid != pivot
                and assignment[bid] != cur_cls
                and gender_sig[bid] == gender_sig[pivot]
            ]
            # 점수 유사 우선
            candidates.sort(key=lambda bid: score_dist(pivot, bid))

            for cand in candidates[:candidates_per_iter]:
                apply_swap(assignment, pivot, cand)
                # 위반 해결됐는지
                if assignment[a] != assignment[b]:
                    solved = True
                    diag_lines.append(f"[iter {it}] swap: {pivot} <-> {cand} (dist={score_dist(pivot,cand):.2f})")
                    break
                apply_swap(assignment, pivot, cand)

            if solved:
                break

        if solved:
            continue

        # 2) cycle 교환(3~max_cycle_len) 시도
        if max_cycle_len >= 3:
            # pivot을 고정하고, 같은 성별구성 블록들로 cycle 구성
            pivot = a
            sig = gender_sig[pivot]
            pool = [bid for bid in block_ids if bid != pivot and gender_sig[bid] == sig]

            # pivot과 점수 가까운 애들 위주로 pool 줄이기
            pool.sort(key=lambda bid: score_dist(pivot, bid))
            pool = pool[:max(30, candidates_per_iter)]

            found = False
            for L in range(3, max_cycle_len + 1):
                # 랜덤 샘플 기반 얕은 탐색
                tries = 250
                if len(pool) < L - 1:
                    continue
                for _ in range(tries):
                    cand = random.sample(pool, k=L - 1)
                    cycle = [pivot] + cand

                    # cycle이 전부 같은 반이면 의미 없음
                    if len({assignment[x] for x in cycle}) < 2:
                        continue

                    backup = {x: assignment[x] for x in cycle}
                    apply_cycle(assignment, cycle)

                    if assignment[a] != assignment[b]:
                        diag_lines.append(f"[iter {it}] cycle({L}): " + " -> ".join(cycle))
                        found = True
                        break

                    # revert
                    for x in cycle:
                        assignment[x] = backup[x]

                if found:
                    solved = True
                    break

        if solved:
            continue

        # 3) 막혔으면, '가장 점수 가까운 스왑'을 한 번 강제로 수행해서 상태를 흔들어줌
        #    (그래도 교환은 동일 성별구성끼리만 함)
        pivot = a
        sig = gender_sig[pivot]
        cur_cls = assignment[pivot]
        pool2 = [
            bid for bid in block_ids
            if bid != pivot
            and assignment[bid] != cur_cls
            and gender_sig[bid] == sig
        ]
        pool2.sort(key=lambda bid: score_dist(pivot, bid))

        if pool2:
            cand = pool2[0]
            apply_swap(assignment, pivot, cand)
            diag_lines.append(f"[iter {it}] shake swap: {pivot} <-> {cand} (dist={score_dist(pivot,cand):.2f})")
            continue

        # 4) 동일 성별구성의 다른 반 블록이 하나도 없으면, 이 edge는 swap-only로는 불가
        diag_lines.append(f"[iter {it}] FAIL edge ({a},{b}) in class={cls}: no compatible blocks to swap/cycle")
        raise ValueError("\n".join(diag_lines))

    bad = violates_edges(assignment)
    diag_lines.append(f"- FAIL: max_iters reached, remaining violations={len(bad)}")
    if bad:
        a, b, cls = bad[0]
        diag_lines.append(f"- sample remaining edge: ({a},{b}) in class={cls}")
    raise ValueError("\n".join(diag_lines))
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
view_base = df_all.copy()

# 이름 표시 모드 반영(업로드 화면용)
# name_mode_before 라디오를 쓰고 있다면 그걸, 아니면 기본은 원본
if "name_mode_before" in st.session_state:
    nm = st.session_state.name_mode_before
else:
    nm = "원본"

if "이름(원본)" in view_base.columns and "이름(한글만)" in view_base.columns:
    view_base["이름"] = view_base["이름(한글만)"] if nm == "한글만" else view_base["이름(원본)"]
elif "이름" not in view_base.columns:
    # 최후 fallback
    view_base["이름"] = view_base.get("이름(원본)", "")

# 이전반 표시 컬럼 보장
if "이전반(표시)" not in view_base.columns:
    if "이전반_raw" in view_base.columns:
        view_base["이전반(표시)"] = view_base["이전반_raw"].map(format_prev_class_display)
    else:
        view_base["이전반(표시)"] = ""

# 업로드 화면 정렬(기본: 번호순)
view_base = view_base.sort_values(by=["반", "번호"], ascending=[True, True], na_position="last")


# ---- 업로드 직후(조정 전): 반 테이블만 표시 ----
render_class_tabs(
    df=view_base,
    title="📋 반별 학생 목록",
    show_changed=False,
    table_cols=["_excel_row", "반", "번호", "이름", "생년월일", "성별", "점수", "이전반(표시)"],
    rename_map={"_excel_row": "엑셀행번호", "이전반(표시)": "이전반"},
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
        # 1) 블록 만들기 (기존 build_blocks 그대로 사용)
        blocks, uid_to_block, not_same_edges, impossible = build_blocks(df_all, st.session_state.constraints)
        if impossible:
            raise ValueError("조건이 서로 모순입니다. '떨어뜨리기'가 '묶기'로 묶인 학생들을 포함하고 있습니다.")

        # 2) 새 조정 함수 호출 (block->class + diagnostics 반환)
        assignment_block_to_class, diagnostics = adjust_classes_min_change_swap_only_v2(
            df_all,
            st.session_state.constraints,
            blocks,
            uid_to_block,
            not_same_edges,
            size_min=19,
            size_max=21,
            gender_diff_max=2,
            relax_gender_swap=True,
            max_iters=5000,
        )

        # 3) block->class 를 uid->class로 풀기
        uid_new_class = {}
        for bid, members in blocks.items():
            new_cls = assignment_block_to_class[bid]
            for uid in members:
                uid_new_class[uid] = new_cls
        
        # (선택) 실패/성공 진단 로그 UI
        with st.expander("조정 진단 로그", expanded=False):
            st.text(diagnostics)

        result = df_all.copy()
        result["반_원본"] = result["반"]
        result["반"] = result["_uid"].map(uid_new_class)

        # 변경 여부
        result["변경"] = result["반"] != result["반_원본"]

        # 표시용 컬럼 구성(학년 제외, 시트 제외)
        result["이전반(표시)"] = result["이전반_raw"].map(format_prev_class_display)
        result_display = result.copy()

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
    nm = st.session_state.get("name_mode_after", "원본")  # 라디오와 동일 key
    if "이름" not in res.columns:
        if "이름(원본)" in res.columns and "이름(한글만)" in res.columns:
            res["이름"] = res["이름(한글만)"] if nm == "한글만" else res["이름(원본)"]
        elif "이름(원본)" in res.columns:
            res["이름"] = res["이름(원본)"]
        elif "이름(한글만)" in res.columns:
            res["이름"] = res["이름(한글만)"]
        else:
            raise ValueError("조정 결과(res)에 이름 관련 컬럼이 없습니다. result_df 생성 시 df_all의 이름 컬럼을 포함하세요.")

    if "반_원본" not in res.columns and "원본반" in res.columns:
        res["반_원본"] = res["원본반"]
    if "반_원본" not in res.columns:
        # 최소한 현재 반을 원본으로 가정(임시)
        res["반_원본"] = res.get("반", "")
    
    if "변경" not in res.columns:
        res["변경"] = (res.get("반", "") != res.get("반_원본", ""))
    
    if "이전반(표시)" not in res.columns:
        # 이전반_raw 또는 이전반이 있으면 표시 변환
        if "이전반_raw" in res.columns:
            res["이전반(표시)"] = res["이전반_raw"].map(format_prev_class_display)
        elif "이전반" in res.columns:
            res["이전반(표시)"] = res["이전반"].map(format_prev_class_display)
        else:
            res["이전반(표시)"] = ""
    
    if "조건대상" not in res.columns:
        # 조건 리스트가 있다면 uid 기반으로 계산 가능
        constrained_uids = {u for c in st.session_state.constraints for u in c.uids} if "constraints" in st.session_state else set()
        if "_uid" in res.columns:
            res["조건대상"] = res["_uid"].isin(constrained_uids)
        else:
            res["조건대상"] = False

    def highlight_rows(row):
        # 다크/라이트 모두 무난한 반투명 오버레이
        moved_bg = "background-color: rgba(255, 255, 255, 0.18);"         # 변경됨
        constraint_bg = "background-color: rgba(0, 180, 255, 0.14);"      # 조건대상
        both_bg = (
            "background-color: rgba(0, 180, 255, 0.14);"
            "box-shadow: inset 0 0 0 9999px rgba(255, 255, 255, 0.06);"
        )

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

    render_class_tabs(
        df=res,
        title="📋 반별 학생 목록(조정 결과)",
        show_changed=True,
        highlight_func=highlight_rows,
        table_cols=["_excel_row", "반_원본", "반", "번호", "이름", "생년월일", "성별", "점수", "이전반(표시)", "조건대상", "변경"],
        rename_map={
            "_excel_row": "엑셀행번호",
            "반_원본": "원본반",
            "반": "조정반",
            "이전반(표시)": "이전반",
        },
        hide_cols=["조건대상", "변경"],
    )

    # ✅ 반별 테이블(조정 결과) 아래: 조정된 학생 목록(원본 대비 반 변경)
    st.subheader("🔁 조정된 학생 목록(원본 대비 반 변경)")
    
    # 변경 컬럼이 없으면 생성(안전)
    if "변경" not in res.columns:
        res["변경"] = (res["반"].astype(str) != res["반_원본"].astype(str))
    
    moved = res[res["변경"] == True].copy()
    
    # 보기 좋게 정렬: 원본반 → 조정반 → 번호
    sort_cols = [c for c in ["반_원본", "반", "번호", "점수"] if c in moved.columns]
    if sort_cols:
        moved = moved.sort_values(by=sort_cols, ascending=[True, True, True, False][:len(sort_cols)], na_position="last")
    
    # 표시 컬럼(있는 것만)
    moved_cols = [c for c in ["_excel_row", "반_원본", "반", "번호", "이름", "성별", "점수", "이전반(표시)"] if c in moved.columns]
    moved_rename = {
        "_excel_row": "엑셀행번호",
        "반_원본": "원본반",
        "반": "조정반",
        "이전반(표시)": "이전반",
    }
    
    if moved.empty:
        st.info("조정된 학생이 없습니다. (원본 배정을 그대로 유지했습니다.)")
    else:
        st.dataframe(
            moved[moved_cols].rename(columns=moved_rename),
            use_container_width=True
        )

    
                
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
