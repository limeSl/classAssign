import io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st


# =========================
# 설정
# =========================
st.set_page_config(page_title="반편성 합시다.", layout="wide")
st.title("반편성 합시다.")

REQUIRED_COLS = {
    "학년": "A",
    "반": "B",
    "번호": "C",
    "이름": "D",
    "생년월일": "E",
    "성별": "F",
    "점수": "G",
    "이전 반": "I",
}

WEB_COL_ORDER = ["반", "번호", "이름", "생년월일", "성별", "점수", "이전 반"]


# =========================
# 유틸
# =========================
def _norm_gender(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # 흔한 표기 정리
    if s in ["남", "남자", "M", "m", "male", "Male"]:
        return "남"
    if s in ["여", "여자", "F", "f", "female", "Female"]:
        return "여"
    return s


def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def build_uid(df: pd.DataFrame) -> pd.Series:
    return (
        df["이전 학년"].astype(str).str.strip() + "-"
        + df["이전 반"].astype(str).str.strip() + "-"
        + df["이전 번호"].astype(str).str.strip()
    )

def display_name(df: pd.DataFrame) -> pd.Series:
    # 선택창에서 검색 편하도록 "이름 (반-번호, 생년월일)" 형태
    return (
        df["이름"].astype(str)
        + " ("
        + df["반"].astype(str)
        + "-"
        + df["번호"].astype(str)
        + ", "
        + df["생년월일"].astype(str)
        + ")"
    )


def to_web_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 웹에서는 학년 제외, 이전 반은 "1-n" 형태로 표기 (요청)
    # 입력 I열(이전 반)에 뭐가 오든 문자열로 보여줌.
    out["이전 반"] = out["이전 반"].apply(_safe_str)
    return out[WEB_COL_ORDER].sort_values(["반", "번호"]).reset_index(drop=True)


def highlight_moved(original_uid_set: set):
    """
    st.dataframe의 Styler에 적용: moved인 학생 하이라이트 (다크모드에서도 잘 보이게)
    """
    def _apply(row):
        uid = row.get("UID", "")
        if uid in original_uid_set:
            # 이동된 학생 행 스타일: 밝은 배경 + 진한 테두리(다크모드 대비)
            return [
                "background-color: #FFD54F; color: #111; font-weight: 700; border: 1px solid #FFB300;"
            ] * len(row)
        return [""] * len(row)
    return _apply


# =========================
# 제약조건 구조
# =========================
@dataclass
class Constraint:
    kind: str  # "묶기" or "떨어뜨리기"
    uids: List[str]


# =========================
# 교환 로직
# =========================
def score_distance(a: float, b: float) -> float:
    try:
        return abs(float(a) - float(b))
    except Exception:
        return 10**9


def pick_best_swap_candidate(
    df: pd.DataFrame,
    from_uid: str,
    target_class: str,
    forbidden_uids: set,
) -> Optional[str]:
    """
    from_uid 학생을 target_class로 보내기 위해,
    target_class 안에서 같은 성별 + 점수 가장 유사한 학생 UID를 선택.
    forbidden_uids는 교환에 쓰면 안 되는 UID들(같은 묶음 등).
    """
    if from_uid not in df["UID"].values:
        return None
    s = df.loc[df["UID"] == from_uid].iloc[0]
    gender = s["성별"]
    score = s["점수"]

    pool = df[(df["반"] == target_class) & (df["성별"] == gender)]
    if forbidden_uids:
        pool = pool[~pool["UID"].isin(forbidden_uids)]

    if pool.empty:
        return None

    pool = pool.copy()
    pool["_dist"] = pool["점수"].apply(lambda x: score_distance(x, score))
    pool = pool.sort_values(["_dist"])
    return pool.iloc[0]["UID"]


def swap_uids(df: pd.DataFrame, uid_a: str, uid_b: str) -> None:
    """
    uid_a와 uid_b의 반을 맞교환.
    (성별 동일 전제: 호출부에서 보장)
    """
    ia = df.index[df["UID"] == uid_a][0]
    ib = df.index[df["UID"] == uid_b][0]
    class_a = df.at[ia, "반"]
    class_b = df.at[ib, "반"]
    df.at[ia, "반"] = class_b
    df.at[ib, "반"] = class_a


def violated_group(df: pd.DataFrame, uids: List[str]) -> bool:
    classes = set(df[df["UID"].isin(uids)]["반"].tolist())
    return len(classes) > 1


def violated_separate(df: pd.DataFrame, uids: List[str]) -> bool:
    sub = df[df["UID"].isin(uids)][["UID", "반"]]
    # 같은 반 중복 있으면 위반
    return sub["반"].duplicated().any()


def satisfy_group_constraint_by_swaps(
    df: pd.DataFrame,
    uids: List[str],
    max_steps: int = 300,
) -> List[Tuple[str, str]]:
    """
    묶기: uids를 한 반으로 모으되,
    학생 이동은 "같은 성별 + 점수유사" 학생과 교환만 허용.
    반환: 수행한 (uid_moved, uid_swapped_with) 기록
    """
    actions = []
    if not uids:
        return actions

    # 목표 반: 현재 uids 중 가장 많은 반(최소 이동)
    sub = df[df["UID"].isin(uids)]
    if sub.empty:
        return actions
    target_class = sub["반"].value_counts().idxmax()

    forbidden = set(uids)  # 같은 묶음끼리는 교환상대에서 제외(원치않는 꼬임 줄이기)

    steps = 0
    while violated_group(df, uids) and steps < max_steps:
        steps += 1

        sub = df[df["UID"].isin(uids)]
        not_in_target = sub[sub["반"] != target_class]
        if not_in_target.empty:
            break

        # 하나씩 target_class로 보내기
        mover = not_in_target.iloc[0]
        mover_uid = mover["UID"]
        mover_gender = mover["성별"]

        # target_class 안에서 같은 성별 & 점수 가장 유사한 교환상대 찾기
        cand_uid = pick_best_swap_candidate(
            df=df,
            from_uid=mover_uid,
            target_class=target_class,
            forbidden_uids=forbidden,
        )

        # target_class에 같은 성별 교환상대가 없다면: 다른 반에서 먼저 "자리 만들기"
        # (실무상 성비 고정 때문에 이 상황이 종종 생김)
        if cand_uid is None:
            # 같은 성별 학생이 target_class로 들어올 수 있게,
            # target_class에 있는 같은 성별 학생을 밖으로 빼는 대신,
            # 밖에서 같은 성별을 target_class로 넣는 교환을 시도(즉, 2단계 교환으로 slot 확보)
            # -> 간단히: mover와 같은 성별인 학생이 있는 다른 반 중,
            # target_class에 같은 성별이 존재하는 반을 우선적으로 선택해서 1회 스왑만으로 해결을 시도
            # 그래도 없으면 이 묶기는 더 진행 불가(성비/교환 제약상).
            # 여기서는 "최대한"만 시도.
            other_pool = df[(df["성별"] == mover_gender) & (~df["UID"].isin(forbidden))]
            if other_pool.empty:
                break

            # mover가 target_class로 들어가려면 target_class 안에 같은 성별이 있어야 교환 가능
            # 없으면 더 진행 불가
            tc_same_gender = df[(df["반"] == target_class) & (df["성별"] == mover_gender) & (~df["UID"].isin(forbidden))]
            if tc_same_gender.empty:
                break

            # 그중 점수 가장 비슷한 target_class 학생을 임의로 선택
            tc_same_gender = tc_same_gender.copy()
            tc_same_gender["_dist"] = tc_same_gender["점수"].apply(lambda x: score_distance(x, mover["점수"]))
            cand_uid = tc_same_gender.sort_values("_dist").iloc[0]["UID"]

        # 실제 스왑
        # (같은 성별인지 확인)
        a = df[df["UID"] == mover_uid].iloc[0]
        b = df[df["UID"] == cand_uid].iloc[0]
        if a["성별"] != b["성별"]:
            break

        swap_uids(df, mover_uid, cand_uid)
        actions.append((mover_uid, cand_uid))

    return actions


def satisfy_separate_constraint_by_swaps(
    df: pd.DataFrame,
    uids: List[str],
    max_steps: int = 300,
) -> List[Tuple[str, str]]:
    """
    떨어뜨리기: 같은 반에 2명 이상 있으면 한 명을 다른 반으로 보내되,
    같은 성별 + 점수유사 학생과 교환만 허용.
    """
    actions = []
    if not uids:
        return actions

    forbidden = set(uids)  # 서로 교환상대로 쓰지 않기

    steps = 0
    while violated_separate(df, uids) and steps < max_steps:
        steps += 1

        sub = df[df["UID"].isin(uids)][["UID", "반", "성별", "점수"]].copy()
        # 중복 반 찾기
        dup_class = sub[sub["반"].duplicated(keep=False)]["반"].iloc[0]
        in_same = sub[sub["반"] == dup_class]

        # 그중 한 명을 이동 대상으로(임의로 첫 번째)
        mover = in_same.iloc[0]
        mover_uid = mover["UID"]
        mover_gender = mover["성별"]

        # 목적 반 후보: 이 제약 uids가 아직 없는 반 우선
        classes_all = sorted(df["반"].unique().tolist())
        occupied = set(sub["반"].tolist())
        target_candidates = [c for c in classes_all if c not in occupied]
        if not target_candidates:
            # 모든 반에 이미 누군가 있다면, 그래도 현재 반(dup_class)만 피해서 시도
            target_candidates = [c for c in classes_all if c != dup_class]
        if not target_candidates:
            break

        # 후보 반들 중 "교환상대가 존재"하면서 점수 가장 유사한 곳을 찾기
        best = None  # (dist, target_class, cand_uid)
        for tc in target_candidates:
            cand_uid = pick_best_swap_candidate(
                df=df,
                from_uid=mover_uid,
                target_class=tc,
                forbidden_uids=forbidden,
            )
            if cand_uid is None:
                continue
            cand = df[df["UID"] == cand_uid].iloc[0]
            dist = score_distance(cand["점수"], mover["점수"])
            if best is None or dist < best[0]:
                best = (dist, tc, cand_uid)

        if best is None:
            # 가능한 교환상대가 없으면 진행 불가
            break

        _, _, cand_uid = best

        # 성별 확인 후 스왑
        a = df[df["UID"] == mover_uid].iloc[0]
        b = df[df["UID"] == cand_uid].iloc[0]
        if a["성별"] != b["성별"]:
            break

        swap_uids(df, mover_uid, cand_uid)
        actions.append((mover_uid, cand_uid))

    return actions


def apply_constraints(
    df_original: pd.DataFrame,
    constraints: List[Constraint],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    제약들을 순차적으로 최대한 만족시키도록 스왑 수행.
    반환: 조정된 df, 이동한 UID 리스트
    """
    df = df_original.copy()

    # 반복 적용(앞 제약 해결이 뒤 제약을 깨뜨릴 수 있어 라운드로 조금 돌림)
    moved_pairs = []
    MAX_ROUNDS = 6

    for _ in range(MAX_ROUNDS):
        changed = False

        for c in constraints:
            if c.kind == "묶기":
                if violated_group(df, c.uids):
                    acts = satisfy_group_constraint_by_swaps(df, c.uids)
                    if acts:
                        moved_pairs.extend(acts)
                        changed = True
            elif c.kind == "떨어뜨리기":
                if violated_separate(df, c.uids):
                    acts = satisfy_separate_constraint_by_swaps(df, c.uids)
                    if acts:
                        moved_pairs.extend(acts)
                        changed = True

        if not changed:
            break

    moved_uids = sorted(set([a for a, _ in moved_pairs] + [b for _, b in moved_pairs]))
    return df, moved_uids


def make_download_excel(df: pd.DataFrame) -> bytes:
    """
    다운로드용 엑셀 생성.
    입력 형식에 맞춰 A~I 중 웹 제외 항목도 어느 정도 맞춰줌.
    - A 학년: 비워두거나 원본 유지 가능(원본이 있으면 유지)
    - B 반, C 번호, D 이름, E 생년월일, F 성별, G 점수, I 이전 반
    """
    out = df.copy()

    # 출력 컬럼 구성(엑셀 칼럼명은 사람이 읽기 쉽게 한글 유지)
    # 학년은 원본에 있으면 유지, 없으면 빈칸
    if "학년" not in out.columns:
        out["학년"] = ""

    cols = ["학년", "반", "번호", "이름", "생년월일", "성별", "점수", "이전 반"]
    out = out[cols]

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="반편성결과")
    return bio.getvalue()


# =========================
# 1. 엑셀 파일 업로드
# =========================
uploaded = st.file_uploader("1. 엑셀 파일 업로드", type=["xlsx"])

if uploaded is not None:
    # 엑셀 읽기
    try:
        raw = pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"엑셀을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    # 열 매핑: 사용자가 준 형식이 "A~I 위치"라서,
    # 실제 헤더가 없거나 제각각일 수 있음 → 우선 컬럼명 그대로 사용하되,
    # 만약 사용자가 이미 한글 헤더로 저장해둔 경우도 지원.
    # 가장 안전: 컬럼 개수 기준으로 A/B/C... 위치로 재명명.
    if raw.shape[1] < 9:
        st.error("엑셀 컬럼이 부족합니다. 최소 A~I(9개)까지 있어야 합니다.")
        st.stop()

    # 위치 기반 재명명(A=0,...,I=8)
    col_map_by_pos = {
        0: "학년",
        1: "반",
        2: "번호",
        3: "이름",
        4: "생년월일",
        5: "성별",
        6: "점수",
        7: "이전 학년",
        8: "이전 반",
        9: "이전 번호",
    }

    df = raw.copy()
    # 먼저 전체를 문자열/기본형으로 두고 필요한 칼럼만 rename
    rename = {}
    for idx, newname in col_map_by_pos.items():
        old = df.columns[idx]
        rename[old] = newname
    df = df.rename(columns=rename)

    # 필요한 컬럼 존재 확인
    needed = ["반", "번호", "이름", "성별", "점수", "이전 학년", "이전 반", "이전 번호"]
    for c in needed:
        if c not in df.columns:
            st.error(f"필수 컬럼 '{c}'를 찾지 못했습니다. 엑셀 형식을 확인해주세요.")
            st.stop()

    # 정리
    df = df[["학년", "반", "번호", "이름", "생년월일", "성별", "점수", "이전 반"]].copy()
    df["성별"] = df["성별"].apply(_norm_gender)
    df["UID"] = build_uid(df)
    df["표시명"] = display_name(df)

    # 세션 저장 (원본 고정)
    st.session_state["df_original"] = df.copy()

    # =========================
    # 2. 업로드 하면 -> 반별 학생 테이블
    # =========================
    st.subheader("2. 업로드 결과(원본)")
    st.dataframe(to_web_df(df), use_container_width=True)

    # =========================
    # 3. 조건 추가 (학생 테이블 아래에 표시)
    # =========================
    st.subheader("3. 조건 추가")

    if "constraints" not in st.session_state:
        st.session_state["constraints"] = []

    col1, col2 = st.columns([1, 3])

    with col1:
        kind = st.radio("조건 종류", ["학생 묶기", "학생 떨어뜨리기"], horizontal=False)

    with col2:
        # 이름 검색 가능한 multiselect
        options = dict(zip(df["표시명"].tolist(), df["UID"].tolist()))
        selected_display = st.multiselect(
            "학생 선택(이름 검색 가능)",
            options=list(options.keys()),
        )
        selected_uids = [options[x] for x in selected_display]

    add_btn = st.button("조건 추가", type="primary")

    if add_btn:
        if len(selected_uids) < 2:
            st.warning("조건은 최소 2명 이상 선택해야 합니다.")
        else:
            st.session_state["constraints"].append(
                Constraint(
                    kind="묶기" if kind == "학생 묶기" else "떨어뜨리기",
                    uids=selected_uids,
                )
            )

    # 현재 조건 표시
    if st.session_state["constraints"]:
        st.markdown("**현재 추가된 조건**")
        for i, c in enumerate(st.session_state["constraints"], start=1):
            names = df[df["UID"].isin(c.uids)]["표시명"].tolist()
            st.write(f"{i}. [{c.kind}] " + ", ".join(names))
        clear_btn = st.button("조건 전체 삭제")
        if clear_btn:
            st.session_state["constraints"] = []
            st.rerun()
    else:
        st.info("아직 조건이 없습니다. 조건을 추가하세요.")

    # =========================
    # 4. 조정하기 버튼 -> 조정된 테이블/목록
    # =========================
    st.subheader("4. 조정")
    adjust_btn = st.button("조정하기")

    if adjust_btn:
        df_original = st.session_state["df_original"].copy()
        constraints = st.session_state["constraints"]

        df_adjusted, moved_uids = apply_constraints(df_original, constraints)

        st.session_state["df_adjusted"] = df_adjusted
        st.session_state["moved_uids"] = moved_uids

    if "df_adjusted" in st.session_state:
        df_adjusted = st.session_state["df_adjusted"]
        moved_uids = set(st.session_state.get("moved_uids", []))

        st.markdown("**조정된 반별 학생 테이블**")
        show_df = df_adjusted.copy()
        web_df = to_web_df(show_df)

        # moved 표시용으로 UID 포함한 테이블에 스타일 적용 후 UID는 숨기기 어려워서:
        # 웹표시는 st.dataframe(Styler)로, 내부적으로 UID열을 잠깐 붙여서 하이라이트.
        temp = show_df[WEB_COL_ORDER + ["UID"]].copy()
        temp = temp.sort_values(["반", "번호"]).reset_index(drop=True)
        styled = temp.style.apply(highlight_moved(moved_uids), axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("**조정된 학생 목록(원본과 위치가 변한 학생만)**")
        if moved_uids:
            moved_rows = df_adjusted[df_adjusted["UID"].isin(list(moved_uids))].copy()

            # 원본 반/번호 정보도 같이 보여주기 위해 merge
            orig = st.session_state["df_original"][["UID", "반", "번호"]].copy()
            orig = orig.rename(columns={"반": "원본 반", "번호": "원본 번호"})
            moved_rows = moved_rows.merge(orig, on="UID", how="left")

            moved_rows = moved_rows[["원본 반", "원본 번호", "반", "번호", "이름", "성별", "점수"]].copy()
            moved_rows = moved_rows.sort_values(["원본 반", "원본 번호"]).reset_index(drop=True)
            st.dataframe(moved_rows, use_container_width=True)
        else:
            st.success("이동한 학생이 없습니다(조건이 이미 만족되었거나, 교환 제약상 더 조정할 수 없는 경우).")

        # =========================
        # 5. 엑셀 다운로드 버튼
        # =========================
        st.subheader("5. 엑셀 다운로드")
        excel_bytes = make_download_excel(df_adjusted)
        st.download_button(
            label="엑셀 다운로드",
            data=excel_bytes,
            file_name="반편성_결과.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
