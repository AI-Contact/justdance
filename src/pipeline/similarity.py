"""
유사도 계산 및 가중치 관련 함수들
"""

import numpy as np
import json, os
from .pose_utils import BONES, ANGLE_TRIPLES

# 영역별 기본 가중치(원하는 값으로 조정 가능)
REGION_WEIGHTS = {
    "arm": 2.2,     # 팔
    "leg": 2.4,     # 다리
    "torso": 0.6,   # 몸통(어깨폭, 골반폭, 좌/우 연결 등)
}
ANGLE_WEIGHTS = {
    "left elbow": 0.5,
    "right elbow": 2.0,
    "left knee": 0.5,
    "right knee": 2.4,
    "hip": 2.2,
    "shoulder": 0.5,
}
REGION_COLORS = {
    "arm":   (60, 180, 255),   # 주황톤
    "leg":   (60, 255, 60),    # 연초록
    "torso": (200, 200, 200),  # 회색
}


# BONES를 부위로 태깅(필요시 수정)
def _bone_region(i, j):
    arms = {11,12,13,14,15,16}
    legs = {23,24,25,26,27,28}
    torso_pairs = {(11,12),(23,24),(11,23),(12,24)}
    if (i,j) in torso_pairs or (j,i) in torso_pairs:
        return "torso"
    if i in arms and j in arms:
        return "arm"
    if i in legs and j in legs:
        return "leg"
    # 팔↔몸통/다리↔몸통 연결은 중간값으로: 여기선 torso로 취급
    return "torso"


# ANGLE_TRIPLES를 부위로 태깅(필요시 수정)
def _angle_region(a,b,c):
    # (25,23,27) left knee / (26,24,28) right knee
    if {a,b,c} & {25,27} and 23 in {a,b,c}: return "left knee"
    if {a,b,c} & {26,28} and 24 in {a,b,c}: return "right knee"
    # (11,13,15)/(12,14,16) -> elbow
    if {a,b,c} & {13,15}: return "left elbow"
    if {a,b,c} & {14,16}: return "right elbow"
    # (13,11,23)/(14,12,24) -> shoulder/hip 복합 -> shoulder 쪽으로
    if {a,b,c} & {11,12}: return "shoulder"
    if {a,b,c} & {23,24}: return "hip"
    return "shoulder"


def build_static_feature_weights_old(BONES, ANGLE_TRIPLES, lm_for_vis=None, vis_thresh=0.15):
    """
    임베딩 순서:
      - BONES 개수 * 2 (각 뼈대의 (dx, dy) 단위벡터)
      - ANGLE_TRIPLES 개수 * 1 (라디안)
    반환: (D,) 벡터
    """
    w = []

    # 1) Bone 방향 벡터(x,y)에 부위 가중치 적용
    for (i,j) in BONES:
        region = _bone_region(i,j)
        base = REGION_WEIGHTS.get(region, 1.0)
        # 가시성 보정(선택): 두 관절 모두 보일수록↑ (옆모습 대응)
        if lm_for_vis is not None:
            vi = 1.0 if lm_for_vis[i,3] >= vis_thresh else 0.5
            vj = 1.0 if lm_for_vis[j,3] >= vis_thresh else 0.5
            base = base * min(vi, vj)
        # (dx, dy)에 동일 가중
        w.extend([base, base])

    # 2) 각도 성분에 가중치
    for (a,b,c) in ANGLE_TRIPLES:
        region = _angle_region(a,b,c)
        base = ANGLE_WEIGHTS.get(region, 1.0)
        if lm_for_vis is not None:
            va = 1.0 if lm_for_vis[a,3] >= vis_thresh else 0.5
            vb = 1.0 if lm_for_vis[b,3] >= vis_thresh else 0.5
            vc = 1.0 if lm_for_vis[c,3] >= vis_thresh else 0.5
            base = base * min(va, vb, vc)
        w.append(base)

    return np.array(w, dtype=np.float32)

import json, os
import numpy as np

def build_static_feature_weights(
    BONES, ANGLE_TRIPLES,
    lm_for_vis=None,
    region_w=None,
    angle_w=None,
    vis_thresh=0.15  # 옆모습에서도 점수 계산 가능하도록 0.2로 낮춤
):
    """
    BONES와 ANGLE_TRIPLES 구조를 기반으로 feature weight 벡터를 생성한다.
    region_w, angle_w는 load_weights_for_video()가 반환한 dict를 받을 수 있다.
    """
    # 입력된 region/angle weight가 없으면 전역 기본값 사용
    if region_w is None:
        region_w = REGION_WEIGHTS
    if angle_w is None:
        angle_w = ANGLE_WEIGHTS

    w = []

    # 1) Bone 방향 벡터(x,y)에 부위 가중치 적용
    for (i, j) in BONES:
        region = _bone_region(i, j)
        base = region_w.get(region, 1.0)

        # 가시성 보정 (옆모습 대응: 낮은 visibility도 일부 반영)
        if lm_for_vis is not None:
            vi = 1.0 if lm_for_vis[i, 3] >= vis_thresh else 0.5
            vj = 1.0 if lm_for_vis[j, 3] >= vis_thresh else 0.5
            base *= min(vi, vj)

        w.extend([base, base])  # (dx, dy) 각각 동일 가중치

    # 2) 각도 성분 가중치
    for (a, b, c) in ANGLE_TRIPLES:
        region = _angle_region(a, b, c)
        base = angle_w.get(region, 1.0)
        if lm_for_vis is not None:
            va = 1.0 if lm_for_vis[a, 3] >= vis_thresh else 0.5
            vb = 1.0 if lm_for_vis[b, 3] >= vis_thresh else 0.5
            vc = 1.0 if lm_for_vis[c, 3] >= vis_thresh else 0.5
            base *= min(va, vb, vc)
        w.append(base)

    return np.array(w, dtype=np.float32), region_w, angle_w

def load_weights_for_video(ref_video_path: str, weight_json_path: str = "data/weights.json"):
    """
    영상 파일명(확장자 제외)과 동일한 key를 갖는 weight 세트를 불러온다.
    예: ref_video_path = "data/squat.mp4" → key = "squat"
    """
    w=[]
    # 1️⃣ 파일명(확장자 제외)
    key = os.path.splitext(os.path.basename(ref_video_path))[0].lower()  # ex: "yout_squat"

    # 2️⃣ JSON 파일 로드
    if not os.path.exists(weight_json_path):
        raise FileNotFoundError(f"{weight_json_path} not found.")

    with open(weight_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3️⃣ 정확히 동일한 key만 사용
    if key not in data:
        raise KeyError(f"'{key}' not found in {weight_json_path}.")

    weights = data[key]
    region_w = weights.get("REGION_WEIGHTS", {})
    angle_w = weights.get("ANGLE_WEIGHTS", {})

    return region_w, angle_w


def weighted_cosine(a, b, w, eps=1e-8):
    """가중 코사인 유사도: ( (w*a)·(w*b) ) / (||w*a|| ||w*b||)"""
    aw = a * w
    bw = b * w
    na = np.linalg.norm(aw)
    nb = np.linalg.norm(bw)
    return float(np.dot(aw, bw) / (na*nb + eps))


def cosine_sim(a, b, eps=1e-8):
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    return float(np.dot(a,b) / (an*bn + eps))


def exp_moving_avg(prev, new, alpha=0.2):
    return prev*(1-alpha) + new*alpha

