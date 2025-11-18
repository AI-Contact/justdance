"""
포즈 관련 유틸리티 함수들: 랜드마크 정규화, 임베딩 생성, 피드백, 스켈레톤 그리기
"""

import math
import cv2
import numpy as np

# Reuse geometry from extractor to keep embeddings consistent
BONES = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (11, 12),
    (23, 24),
    (11, 23), (12, 24)
]
ANGLE_TRIPLES = [
    (25, 23, 27), (26, 24, 28), (13, 11, 23), (14, 12, 24), (11, 13, 15), (12, 14, 16)
]
ANGLE_NAMES = ["left knee", "right knee", "left hip", "right hip", "left elbow", "right elbow"]

# Indices for MediaPipe Pose (ignore face landmarks for similarity)
FACE_IDXS = list(range(0, 11))   # 0..10 (nose, eyes, ears, mouth region)
BODY_IDXS = list(range(11, 33))  # 11..32 (shoulders, arms, torso, legs, feet)


def _safe_get_xy(landmarks, idx):
    # 옆모습에서도 점수 계산이 가능하도록 임계값을 0.2로 낮춤
    if landmarks[idx, 3] < 0.15:
        return None
    return landmarks[idx, :2]


def _angle(a, b, c):
    if a is None or b is None or c is None:
        return np.nan
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.acos(cos)


def normalize_landmarks(landmarks_xyv):
    lm = landmarks_xyv.copy()
    # Drop face landmarks: remove from visibility and position so they don't affect centering/scaling
    lm[FACE_IDXS, :2] = np.nan
    lm[FACE_IDXS,  3] = 0.0
    if lm[23,3] > 0.5 and lm[24,3] > 0.5:
        pelvis = (lm[23,:2] + lm[24,:2]) / 2.0
    elif lm[23,3] > 0.5:
        pelvis = lm[23,:2]
    elif lm[24,3] > 0.5:
        pelvis = lm[24,:2]
    else:
        pelvis = np.nanmean(lm[BODY_IDXS, :2], axis=0)
    lm[:,:2] -= pelvis
    scale_refs = []
    if lm[11,3]>0.5 and lm[12,3]>0.5:
        scale_refs.append(np.linalg.norm(lm[11,:2]-lm[12,:2]))
    if lm[23,3]>0.5 and lm[24,3]>0.5:
        scale_refs.append(np.linalg.norm(lm[23,:2]-lm[24,:2]))
    scale = np.nanmean(scale_refs) if len(scale_refs)>0 else np.nan
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.nanmax(np.linalg.norm(lm[BODY_IDXS, :2], axis=1)) + 1e-6
    lm[:,:2] /= scale
    if lm[11,3]>0.5 and lm[12,3]>0.5:
        v = lm[12,:2]-lm[11,:2]
        ang = math.atan2(v[1], v[0])
        c, s = math.cos(-ang), math.sin(-ang)
        R = np.array([[c,-s],[s,c]], dtype=np.float32)
        lm[:,:2] = (R @ lm[:,:2].T).T
    return lm


# Note: BONES and ANGLE_TRIPLES only use BODY_IDXS (11..32), so embeddings ignore face by design.
def pose_embedding(landmarks_xyv):
    lm = landmarks_xyv
    vecs = []
    for i,j in BONES:
        a = _safe_get_xy(lm, i); b = _safe_get_xy(lm, j)
        if a is None or b is None:
            vecs.append([0.0, 0.0])
        else:
            v = b - a
            n = np.linalg.norm(v)
            vecs.append((v / (n + 1e-8)).tolist())
    vecs = np.array(vecs, dtype=np.float32).reshape(-1)
    angs = []
    for a,b,c in ANGLE_TRIPLES:
        A = _safe_get_xy(lm, a); B = _safe_get_xy(lm, b); C = _safe_get_xy(lm, c)
        ang = _angle(A,B,C)
        if not np.isfinite(ang):
            ang = 0.0
        angs.append(ang)
    angs = np.array(angs, dtype=np.float32)
    return np.concatenate([vecs, angs], axis=0)


def draw_colored_skeleton(image_bgr, lm33_xyv, thickness=3):
    """가시성>=0.5 인 관절만 선분/원점으로 가시화. BONES를 region 색으로."""
    from .similarity import REGION_COLORS, _bone_region
    
    if lm33_xyv is None:
        return image_bgr
    h, w = image_bgr.shape[:2]

    # 점 찍기
    for idx in range(33):
        x, y, v = lm33_xyv[idx, 0], lm33_xyv[idx, 1], lm33_xyv[idx, 3]
        if v < 0.5 or not np.isfinite(x) or not np.isfinite(y):
            continue
        cx, cy = int(x * w), int(y * h)
        cv2.circle(image_bgr, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # 뼈대 선 그리기(부위별 색)
    for (i, j) in BONES:
        vi, vj = lm33_xyv[i, 3], lm33_xyv[j, 3]
        if vi < 0.5 or vj < 0.5:
            continue
        xi, yi = int(lm33_xyv[i, 0] * w), int(lm33_xyv[i, 1] * h)
        xj, yj = int(lm33_xyv[j, 0] * w), int(lm33_xyv[j, 1] * h)
        region = _bone_region(i, j)
        color = REGION_COLORS.get(region, (255, 255, 255))
        cv2.line(image_bgr, (xi, yi), (xj, yj), color, thickness, cv2.LINE_AA)

    return image_bgr


def angle_feedback(curr_lm, ref_lm, angle_tol_deg=10.0):
    curr_angles, ref_angles = [], []
    for (a,b,c) in ANGLE_TRIPLES:
        A = _safe_get_xy(curr_lm, a); B = _safe_get_xy(curr_lm, b); C = _safe_get_xy(curr_lm, c)
        ra = _safe_get_xy(ref_lm,  a); rb = _safe_get_xy(ref_lm,  b); rc = _safe_get_xy(ref_lm,  c)
        curr_angles.append(_angle(A,B,C))
        ref_angles.append(_angle(ra,rb,rc))
    curr_angles = np.array([0.0 if not np.isfinite(v) else v for v in curr_angles])
    ref_angles  = np.array([0.0 if not np.isfinite(v) else v for v in ref_angles])
    diff_deg = np.abs((curr_angles - ref_angles) * 180.0 / math.pi)
    msgs = []
    for i, d in enumerate(diff_deg):
        if d > angle_tol_deg:
            hint = "Stretch" if curr_angles[i] > ref_angles[i] else "Bend"
            msgs.append(f"{ANGLE_NAMES[i]} {hint} (Difference {d:.1f}°)")
    return msgs

