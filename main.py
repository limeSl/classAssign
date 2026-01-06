import io
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

import pandas as pd
import streamlit as st

# =============================
# UI 기본 (유지)
# =============================
st.set_page_config(page_title="반편성 도우미", page_icon="🏫", layout="wide")
st.title("🏫 반편성 도우미")
st.caption("엑셀 파일을 업로드 하시면 제가 반편성을 도와드립니다.")

# =============================
# 유틸 (유지/안정화)
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
    if df.shape[1] < 9:
        raise ValueError(f"[{sheet_name}] 열이 부족합니다. I열(이전 반)까지 필요합니다. 현재 열 수: {df.shape[1]}")

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

    out = out.dropna(how="all")
    out = out[~((out["반"] == "") & (out["번호"].isna()) & (out["이름(원본)"] == ""))]

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
    반별 탭 + (상단) 인원/성비/반평균 + 표 출력 공통 렌더러 (UI 유지)
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

            n = len(d)
            m = int((d["성별"] == "남").sum()) if "성별" in d.columns else 0
            f = int((d["성별"] == "여").sum()) if "성별" in d.columns else 0
            mean_score = d["점수"].mean() if "점수" in d.columns else None
            mean_text = "—" if mean_score is None or pd.isna(mean_score) else f"{mean_score:.2f}"

            c1, c2, c3 = st.columns(3)
            c1.metric("인원", n)
            c2.metric("성비(남/여)", f"{m}/{f}")
            c3.metric("반 평균점수", mean_text)

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

            if show_changed and highlight_func is not None:
                styled = out.style.apply(highlight_func, axis=1)
                # 환경별 hide 지원 차이 대비
                try:
                    styled = styled.hide(columns=hide_cols_present)
                    st.dataframe(styled, use_container_width=True)
                except Exception:
                    st.dataframe(out.drop(columns=hide_cols_present), use_container_width=True)
            else:
                st.dataframe(out.drop(columns=hide_cols_present), use_container_width=True)

# =============================
# 조건 데이터 구조 (유지)
# =============================
@dataclass
class Constraint:
    kind: str  # "묶기" or "떨어뜨리기"
    uids: List[str]  # 학생 uid 리스트

# Union-Find for 묶기 그룹 (유지)
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

    for c in constraints:
        if c.kind != "묶기":
            continue
        if len(c.uids) < 2:
            continue
        first = c.uids[0]
        for other in c.uids[1:]:
            uf.union(first, other)

    blocks: Dict[str, List[str]] = {}
    for uid in uids:
        root = uf.find(uid)
        blocks.setdefault(root, []).append(uid)

    uid_to_block = {uid: uf.find(uid) for uid in uids}

    not_same_edges: Set[Tuple[str, str]] = set()
    impossible = []
    for c in constraints:
        if c.kind != "떨어뜨리기":
            continue
        us = [uid_to_block[u] for u in c.uids]
        if len(set(us)) < len(us):
            impossible.append(c)
            continue
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a, b = us[i], us[j]
                if a == b:
                    continue
                not_same_edges.add((a, b) if a < b else (b, a))

    return blocks, uid_to_block, not_same_edges, impossible

# =============================
# 조정 로직 (새로 정리: 안정적인 "반복-개선" 스왑 엔진)
# =============================
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

def block_stats(block_id: str, blocks: Dict[str, List[str]], df_index: Dict[str, Dict]):
    """블록의 (인원, 남수, 여수, 평균점수)"""
    n = 0
    m = f = 0
    scores = []
    for uid in blocks[block_id]:
        n += 1
        g = df_index[uid]["성별"]
        if g == "남":
            m += 1
        elif g == "여":
            f += 1
        s = _safe_float(df_index[uid]["점수"])
        if s is not None:
            scores.append(s)
    avg = sum(scores) / len(scores) if scores else None
    return n, m, f, avg

def class_counts_from_assignment(
    assignment: Dict[str, str],
    blocks: Dict[str, List[str]],
    df_index: Dict[str, Dict],
    classes: List[str],
):
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
    for v in cnt.values():
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

def _swap_assign(assignment: Dict[str, str], a: str, b: str):
    ca, cb = assignment[a], assignment[b]
    assignment[a], assignment[b] = cb, ca

def _score_distance(stats_cache, bid1: str, bid2: str) -> float:
    _, _, _, a1 = stats_cache[bid1]
    _, _, _, a2 = stats_cache[bid2]
    if a1 is None or a2 is None:
        return 1e9
    return abs(a1 - a2)

def _same_gender_comp(stats_cache, bid1: str, bid2: str) -> bool:
    _, m1, f1, _ = stats_cache[bid1]
    _, m2, f2, _ = stats_cache[bid2]
    return (m1 == m2) and (f1 == f2)

def _try_swap(
    assignment: Dict[str, str],
    bid1: str,
    bid2: str,
    blocks,
    df_index,
    classes,
    not_same_edges,
    size_min,
    size_max,
    gender_diff_max,
) -> Tuple[bool, int]:
    """swap 적용 가능하면 적용 후 bad_edges 개수를 반환(적용된 상태)."""
    _swap_assign(assignment, bid1, bid2)
    cnt = class_counts_from_assignment(assignment, blocks, df_index, classes)
    if not check_hard_rules(cnt, size_min, size_max, gender_diff_max):
        _swap_assign(assignment, bid1, bid2)
        return False, 10**9
    bad = len(violates_not_same(assignment, not_same_edges))
    return True, bad

def _choose_helper_blocks(
    assignment: Dict[str, str],
    blocks: Dict[str, List[str]],
    df_index: Dict[str, Dict],
    stats_cache: Dict[str, Tuple[int,int,int,Optional[float]]],
    classes: List[str],
    target_bid: str,
    base_pool: Set[str],
    k_per_class: int = 2,
    score_window: float = 30.0,
) -> Set[str]:
    """
    후보가 막히면, 각 반에서 '성별/점수 유사' 블록을 조금씩 풀(pool 확장).
    - base_pool: 현재 허용 풀(보통 조건대상 블록)
    """
    _, tm, tf, tavg = stats_cache[target_bid]
    tavg = 0.0 if tavg is None else float(tavg)

    extra: Set[str] = set()

    for cls in classes:
        # cls 반에 있는 블록 중 base_pool에 없는 블록
        in_cls = [bid for bid, c in assignment.items() if c == cls and bid not in base_pool and bid != target_bid]

        scored = []
        for bid in in_cls:
            _, m, f, avg = stats_cache[bid]
            # 성별 구성 완전 반대면 제외(대체로 교환 후보로 부적절)
            if (tm > 0 and tf == 0) and (f > 0 and m == 0):
                continue
            if (tf > 0 and tm == 0) and (m > 0 and f == 0):
                continue

            if avg is None:
                continue
            dist = abs(float(avg) - tavg)
            if dist <= score_window:
                scored.append((dist, bid))

        scored.sort(key=lambda x: x[0])
        for _, bid in scored[:k_per_class]:
            extra.add(bid)

    return base_pool | extra

def adjust_classes_engine(
    df: pd.DataFrame,
    constraints: List[Constraint],
    blocks: Dict[str, List[str]],
    uid_to_block: Dict[str, str],
    not_same_edges: Set[Tuple[str, str]],
    size_min=19,
    size_max=21,
    gender_diff_max=2,
    max_iters=5000,
) -> Tuple[Dict[str, str], str]:
    """
    안정형 엔진:
    - 기본은 '원본반 유지' assignment
    - 떨어뜨리기 위반을 하나씩 잡아서, 스왑 후보를 찾고 bad_edges를 줄이는 방향으로 반복 개선
    - 후보가 막히면 '보정 후보군(헬퍼)'를 소량 확장(성별/점수 유사)
    - 하드 규칙(인원/성비)은 항상 유지
    """
    diag: List[str] = []

    # uid->info
    df_index: Dict[str, Dict] = {}
    original_class_uid: Dict[str, str] = {}
    for _, r in df.iterrows():
        uid = r["_uid"]
        df_index[uid] = {"성별": r.get("성별", ""), "점수": r.get("점수", None)}
        original_class_uid[uid] = r.get("반", "")

    classes = sorted([c for c in df["반"].unique() if str(c).strip() != ""])
    if not classes:
        raise ValueError("반(B열)이 비어 있어 조정을 진행할 수 없습니다.")

    # 블록의 원본반(최빈)
    original_class_block: Dict[str, str] = {}
    for bid, members in blocks.items():
        counts = {}
        for uid in members:
            oc = original_class_uid.get(uid, "")
            if str(oc).strip() == "":
                continue
            counts[oc] = counts.get(oc, 0) + 1
        original_class_block[bid] = max(counts.items(), key=lambda x: x[1])[0] if counts else classes[0]

    # 초기 assignment: 원본반 유지
    assignment: Dict[str, str] = {bid: original_class_block[bid] for bid in blocks.keys()}

    stats_cache = {bid: block_stats(bid, blocks, df_index) for bid in blocks.keys()}

    # movable 풀: 조건에 포함된 uid의 블록
    constrained_uids = {u for c in constraints for u in c.uids}
    base_movable = {uid_to_block[u] for u in constrained_uids if u in uid_to_block}
    diag.append(f"- movable blocks(base): {len(base_movable)}개 (조건대상 포함 블록)")

    # 원본이 하드규칙을 만족하지 않으면: 최소한의 전체 swap으로 하드규칙을 먼저 맞춰야 함
    cnt0 = class_counts_from_assignment(assignment, blocks, df_index, classes)
    if not check_hard_rules(cnt0, size_min, size_max, gender_diff_max):
        diag.append("- WARN: 원본 배정이 하드 규칙(인원/성비)을 만족하지 않아, 하드규칙 맞추기용 전체 swap(최소)을 허용합니다.")
        # 하드규칙 보정 단계: 풀을 전체로 확장
        movable_pool = set(blocks.keys())
    else:
        movable_pool = set(base_movable)

    # 하드규칙 보정(필요 시): 간단 스왑 반복 (bad_edges는 신경 X)
    def _fix_hard_rules(max_fix_iters=3000):
        nonlocal movable_pool
        for i in range(max_fix_iters):
            cnt = class_counts_from_assignment(assignment, blocks, df_index, classes)
            if check_hard_rules(cnt, size_min, size_max, gender_diff_max):
                diag.append(f"- hard rules fix: OK (iters={i})")
                return True
            # 가장 심한 반/성비를 가진 class를 찾아 스왑
            # 목표: 성비차/인원초과/인원미달 완화
            worst_cls = None
            worst_score = -1
            for c in classes:
                v = cnt[c]
                size_pen = 0
                if v["n"] < size_min:
                    size_pen = (size_min - v["n"]) * 10
                elif v["n"] > size_max:
                    size_pen = (v["n"] - size_max) * 10
                gender_pen = max(0, abs(v["m"] - v["f"]) - gender_diff_max) * 5
                s = size_pen + gender_pen
                if s > worst_score:
                    worst_score = s
                    worst_cls = c
            if worst_cls is None or worst_score <= 0:
                return False

            # worst_cls에 있는 블록 하나와 다른 반 블록 하나를 교환해 규칙 개선 시도
            in_worst = [bid for bid, c in assignment.items() if c == worst_cls]
            candidates = [bid for bid in movable_pool if assignment[bid] != worst_cls]
            if not in_worst or not candidates:
                return False

            # 간단히 점수 유사 + 성별구성 동일 우선
            best_pair = None
            best_metric = 1e18
            for a in in_worst[:80]:
                for b in candidates[:80]:
                    # 스왑 후 규칙 체크
                    _swap_assign(assignment, a, b)
                    cnt2 = class_counts_from_assignment(assignment, blocks, df_index, classes)
                    ok = check_hard_rules(cnt2, size_min, size_max, gender_diff_max)
                    # 완벽 ok면 채택
                    if ok:
                        diag.append(f"- hard-fix swap: {a} <-> {b}")
                        return True
                    # 아니면 개선 정도(최악 점수 감소)로 평가
                    # 원복하고 후보 점수
                    _swap_assign(assignment, a, b)

                    # 휴리스틱 metric: 성별구성 동일/점수차 우선
                    metric = (_score_distance(stats_cache, a, b) +
                              (0 if _same_gender_comp(stats_cache, a, b) else 500.0))
                    if metric < best_metric:
                        best_metric = metric
                        best_pair = (a, b)

            if best_pair:
                a, b = best_pair
                _swap_assign(assignment, a, b)
        diag.append("- hard rules fix: FAIL")
        return False

    if not check_hard_rules(cnt0, size_min, size_max, gender_diff_max):
        if not _fix_hard_rules():
            raise ValueError("\n".join(diag + ["- FAIL: 하드 규칙(인원/성비)을 만족시키지 못했습니다. 엑셀 원본 배정부터 확인이 필요합니다."]))

    # not_same 해결 단계
    helper_k = 2
    helper_win = 30.0

    for it in range(max_iters):
        bad = violates_not_same(assignment, not_same_edges)
        if not bad:
            diag.append(f"- SUCCESS: 떨어뜨리기 위반 0개. iters={it}")
            return assignment, "\n".join(diag)

        a, b = bad[0]
        cls = assignment[a]
        diag.append(f"\n[ITER {it}] violation: ({a},{b}) in class={cls}")

        # 이동시킬 대상 선택: 우선 base_movable에 있는 쪽, 아니면 movable_pool에 있는 쪽
        if a in base_movable:
            move_bid = a
        elif b in base_movable:
            move_bid = b
        elif a in movable_pool:
            move_bid = a
        elif b in movable_pool:
            move_bid = b
        else:
            raise ValueError("\n".join(diag + ["- FAIL: 위반 두 블록 모두 이동 불가(정책상)."]))

        # 후보 풀: 현재 movable_pool 기준
        candidates = [cand for cand in movable_pool if cand != move_bid and assignment[cand] != assignment[move_bid]]
        if not candidates:
            # 후보가 아예 없다면 helper 확장
            movable_pool = _choose_helper_blocks(assignment, blocks, df_index, stats_cache, classes, move_bid, movable_pool, helper_k, helper_win)
            helper_k = min(helper_k + 1, 6)
            helper_win = min(helper_win + 20.0, 120.0)
            diag.append(f"- expand helper pool -> {len(movable_pool)} blocks (k={helper_k}, win={helper_win})")
            continue

        # 현재 bad 개수
        cur_bad = len(bad)

        # 후보 평가: bad_edges 최소, 성별구성 동일 우선, 점수유사 우선, 변경 최소(원본반 유지)
        best = None
        best_key = None

        for cand in candidates:
            # 성별구성 동일을 강하게 선호
            same_comp = _same_gender_comp(stats_cache, move_bid, cand)
            ok, new_bad = _try_swap(
                assignment, move_bid, cand,
                blocks, df_index, classes, not_same_edges,
                size_min, size_max, gender_diff_max
            )
            if not ok:
                continue

            # 스왑은 적용된 상태이므로 이동/변경 비용 계산 후 원복(평가만)
            # moved_count: 원본반과 다르면 1로 카운트(블록 단위 근사)
            moved_after = 0
            for bid in (move_bid, cand):
                if assignment[bid] != original_class_block[bid]:
                    moved_after += 1

            # 점수 차
            sd = _score_distance(stats_cache, move_bid, cand)

            # key: (bad_edges, moved_after, not same_comp, score_dist)
            key = (new_bad, moved_after, 0 if same_comp else 1, sd)

            # 원복
            _swap_assign(assignment, move_bid, cand)

            # 개선이 없으면(위반 개수 동일 이상) 기본적으로 스킵.
            # 다만 stuck 방지를 위해 동일이면 moved/comp가 좋아지면 허용.
            if new_bad > cur_bad:
                continue

            if best is None or key < best_key:
                best = cand
                best_key = key

        if best is None:
            # 막힘: helper 확장
            movable_pool = _choose_helper_blocks(assignment, blocks, df_index, stats_cache, classes, move_bid, movable_pool, helper_k, helper_win)
            helper_k = min(helper_k + 1, 6)
            helper_win = min(helper_win + 20.0, 120.0)
            diag.append(f"- NO swap candidate improving/hard-ok. expand pool -> {len(movable_pool)} blocks (k={helper_k}, win={helper_win})")
            continue

        # best swap 적용
        _swap_assign(assignment, move_bid, best)
        diag.append(f"- SWAP: {move_bid}({original_class_block[move_bid]}→{assignment[move_bid]}) <-> {best}({original_class_block[best]}→{assignment[best]}) | key={best_key}")

    raise ValueError("\n".join(diag + [f"- FAIL: max_iters({max_iters}) 도달(해결 못함)"]))

# =============================
# 세션 상태 초기화 (유지)
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
# 업로드 & 데이터 구성 (UI 유지)
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

# 이름 표시 모드(업로드 화면용)
nm_before = st.session_state.get("name_mode_before", "원본")
if "이름(원본)" in view_base.columns and "이름(한글만)" in view_base.columns:
    view_base["이름"] = view_base["이름(한글만)"] if nm_before == "한글만" else view_base["이름(원본)"]
elif "이름" not in view_base.columns:
    view_base["이름"] = view_base.get("이름(원본)", "")

# 이전반 표시
if "이전반(표시)" not in view_base.columns:
    if "이전반_raw" in view_base.columns:
        view_base["이전반(표시)"] = view_base["이전반_raw"].map(format_prev_class_display)
    else:
        view_base["이전반(표시)"] = ""

# 정렬(기본 번호순)
view_base = view_base.sort_values(by=["반", "번호"], ascending=[True, True], na_position="last")

# ---- 업로드 직후(조정 전) ----
render_class_tabs(
    df=view_base,
    title="📋 반별 학생 목록",
    show_changed=False,
    table_cols=["_excel_row", "반", "번호", "이름", "생년월일", "성별", "점수", "이전반(표시)"],
    rename_map={"_excel_row": "엑셀행번호", "이전반(표시)": "이전반"},
)

# =============================
# 조건 추가 UI (유지)
# =============================
st.subheader("🧩 조건 추가(묶기 / 떨어뜨리기)")

with st.container(border=True):
    kind = st.radio("조건 종류", ["묶기", "떨어뜨리기"], horizontal=True, key="constraint_kind")

    base = view_base.copy()

    options = base["_uid"].tolist()
    uid_to_label = {}

    for _, r in base.iterrows():
        uid = r["_uid"]
        prev_disp = r.get("이전반(표시)", "")
        nm = r.get("이름", "")
        gender = r.get("성별", "")
        score = r.get("점수", None)

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

st.subheader("📌 추가된 조건 목록")
if not st.session_state.constraints:
    st.info("아직 추가된 조건이 없습니다.")
else:
    for i, c in enumerate(st.session_state.constraints):
        with st.container(border=True):
            st.write(f"**#{i+1} {c.kind}** (대상 {len(c.uids)}명)")
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
# 조정 실행 (UI 유지)
# =============================
st.subheader("🛠️ 반편성 조정")

run = st.button("✅ 조정 누르기", type="primary", use_container_width=True)

if run:
    try:
        blocks, uid_to_block, not_same_edges, impossible = build_blocks(df_all, st.session_state.constraints)
        if impossible:
            raise ValueError("조건이 서로 모순입니다. '떨어뜨리기'가 '묶기'로 묶인 학생들을 포함하고 있습니다.")

        assignment_block_to_class, diagnostics = adjust_classes_engine(
            df=df_all,
            constraints=st.session_state.constraints,
            blocks=blocks,
            uid_to_block=uid_to_block,
            not_same_edges=not_same_edges,
            size_min=19,
            size_max=21,
            gender_diff_max=2,
            max_iters=5000,
        )

        uid_new_class = {}
        for bid, members in blocks.items():
            new_cls = assignment_block_to_class[bid]
            for uid in members:
                uid_new_class[uid] = new_cls

        with st.expander("조정 진단 로그", expanded=False):
            st.text(diagnostics)

        result = df_all.copy()
        result["반_원본"] = result["반"]
        result["반"] = result["_uid"].map(uid_new_class)

        result["변경"] = result["반"].astype(str) != result["반_원본"].astype(str)
        result["이전반(표시)"] = result["이전반_raw"].map(format_prev_class_display)

        st.session_state.result_df = result.copy()
        st.success("조정 완료! 아래에서 조정된 테이블을 확인하세요.")

    except Exception as e:
        st.error("조정 중 오류가 발생했습니다.")
        st.exception(e)

# =============================
# 결과 표시 + 다운로드 (UI 유지)
# =============================
if st.session_state.result_df is not None:
    res = st.session_state.result_df.copy()

    # 이름 컬럼 보장
    nm_after = st.session_state.get("name_mode_after", "원본")
    if "이름" not in res.columns:
        if "이름(원본)" in res.columns and "이름(한글만)" in res.columns:
            res["이름"] = res["이름(한글만)"] if nm_after == "한글만" else res["이름(원본)"]
        elif "이름(원본)" in res.columns:
            res["이름"] = res["이름(원본)"]
        elif "이름(한글만)" in res.columns:
            res["이름"] = res["이름(한글만)"]
        else:
            raise ValueError("조정 결과(res)에 이름 관련 컬럼이 없습니다. result_df 생성 시 df_all의 이름 컬럼을 포함하세요.")

    if "반_원본" not in res.columns:
        res["반_원본"] = res.get("반", "")

    if "변경" not in res.columns:
        res["변경"] = (res["반"].astype(str) != res["반_원본"].astype(str))

    if "이전반(표시)" not in res.columns:
        res["이전반(표시)"] = res["이전반_raw"].map(format_prev_class_display) if "이전반_raw" in res.columns else ""

    if "조건대상" not in res.columns:
        constrained_uids = {u for c in st.session_state.constraints for u in c.uids} if "constraints" in st.session_state else set()
        res["조건대상"] = res["_uid"].isin(constrained_uids) if "_uid" in res.columns else False

    # ✅ 스타일(다크/라이트 무난)
    def highlight_rows(row):
        moved_bg = "background-color: rgba(255, 255, 255, 0.18);"
        constraint_bg = "background-color: rgba(0, 180, 255, 0.14);"
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

    # 조정된 학생 목록
    st.subheader("🔁 조정된 학생 목록(원본 대비 반 변경)")
    moved = res[res["변경"] == True].copy()

    sort_cols = [c for c in ["반_원본", "반", "번호", "점수"] if c in moved.columns]
    if sort_cols:
        asc = [True] * len(sort_cols)
        if "점수" in sort_cols:
            # 점수는 내림차순
            asc[sort_cols.index("점수")] = False
        moved = moved.sort_values(by=sort_cols, ascending=asc, na_position="last")

    moved_cols = [c for c in ["_excel_row", "반_원본", "반", "번호", "이름", "성별", "점수", "이전반(표시)"] if c in moved.columns]
    moved_rename = {"_excel_row": "엑셀행번호", "반_원본": "원본반", "반": "조정반", "이전반(표시)": "이전반"}

    if moved.empty:
        st.info("조정된 학생이 없습니다. (원본 배정을 그대로 유지했습니다.)")
    else:
        st.dataframe(moved[moved_cols].rename(columns=moved_rename), use_container_width=True)

    # =============================
    # 엑셀 다운로드 (UI 유지)
    # =============================
    st.subheader("⬇️ 엑셀 다운로드")

    download_df = df_all.copy()
    uid_to_new = {uid: cls for uid, cls in zip(res["_uid"], res["반"])}
    download_df["반"] = download_df["_uid"].map(uid_to_new)

    download_df["H_빈칸"] = ""

    out_cols = pd.DataFrame({
        "학년": download_df["학년"],
        "반": download_df["반"],
        "번호": download_df["번호"],
        "이름": download_df["이름(원본)"],
        "생년월일": download_df["생년월일"],
        "성별": download_df["성별"],
        "점수": download_df["점수"],
        "": download_df["H_빈칸"],              # H열 빈칸
        "이전 반": download_df["이전반_raw"],    # I열
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        grade_series = pd.to_numeric(download_df["학년"], errors="coerce").dropna()
        default_grade = int(grade_series.iloc[0]) if len(grade_series) else 2

        for cls in sorted([c for c in out_cols["반"].unique() if str(c).strip() != ""]):
            sheet_df = out_cols[out_cols["반"] == cls].copy()
            sheet_df = sheet_df.sort_values(by="번호", ascending=True, na_position="last")
            sheet_name = f"{default_grade}-{cls}"[:31]
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)
    st.download_button(
        label="📥 조정된 반편성 엑셀 다운로드",
        data=buffer.getvalue(),
        file_name="반편성_조정결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
