# pose_play.py
# -------------
# Usage:
#   python3 pose_play.py --ref data/skeleton/short_ref.npy --camera 0 --search_radius 5
# If ref_lm.npy is present (same prefix), per-joint feedback will be shown.
"""
# 예시: stride=2로 extract 했고, 같은 short_ref.mp4를 옆에 띄워보자
python3 pose_play.py \
  --ref data/skeleton/short_ref.npy \
  --ref_video data/short_ref.mp4 \
  --ref_stride 2 \
  --search_radius 3 \
  --loop_ref \
  --rtf 1.0 \
  --late_grace 3 --late_penalty 0.05
"""

import argparse
import time
import math
import cv2
import numpy as np
import os
import subprocess
import shutil
import json
import sys

def parse_bgr(color_str, default=(0, 215, 255)):
    """
    Parse 'B,G,R' into a BGR tuple of ints. Fallback to default on error.
    """
    try:
        parts = [int(x.strip()) for x in color_str.split(',')]
        if len(parts) != 3:
            return default
        return tuple(parts[:3])
    except Exception:
        return default

def play_beep(freq=1000.0, dur=0.2, sr=44100, amp=0.2):
    """
    Play a short sine beep using sounddevice if available.
    Non-blocking; failures are safely ignored.
    """
    try:
        import sounddevice as sd
        t = np.linspace(0, dur, int(sr*dur), endpoint=False, dtype=np.float32)
        wave = (amp*np.sin(2*np.pi*freq*t)).astype(np.float32)
        sd.play(wave, samplerate=sr, blocking=False)
    except Exception:
        pass

# Helpers to play/stop reference video audio using ffplay or afplay
def start_ref_audio_player(ref_video_path, start_sec=0.0):
    """
    Try to play audio from the reference video file using ffplay (preferred) or afplay (macOS).
    Returns the subprocess.Popen handle or None on failure.
    """
    try:
        # Prefer ffplay if present
        if shutil.which("ffplay"):
            # -nodisp: no window, -autoexit: exit when done, -loglevel quiet: silent
            # -ss start time
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if start_sec and start_sec > 0:
                cmd += ["-ss", f"{start_sec}"]
            cmd += [ref_video_path]
            return subprocess.Popen(cmd)
        # Fallback to afplay on macOS (cannot seek reliably before Big Sur for video; we start from 0)
        if shutil.which("afplay"):
            cmd = ["afplay", ref_video_path]
            return subprocess.Popen(cmd)
    except Exception:
        return None
    return None


def stop_ref_audio_player(proc):
    try:
        if proc and proc.poll() is None:
            proc.terminate()
    except Exception:
        pass

# --- REST interval helpers ---
def parse_mmss_to_seconds(mmss: str) -> float:
    """'M:S' or 'MM:SS' -> total seconds (float). Returns 0.0 on parse error."""
    try:
        m, s = mmss.strip().split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0.0

def load_rest_intervals_json(json_path: str, ref_video_basename: str, ref_fps: float, ref_stride: int) -> list[tuple[int, int]]:
    """
    Load rest intervals from JSON and convert to *embedding index* intervals.
    JSON format example:
    {
      "clap.mp4": [
        ["0:45","1:00"],
        ["2:30","2:45"]
      ]
    }
    We convert (MM:SS) -> video frames using ref_fps, then -> embedding indices using ref_stride.
    Returns: list of (start_emb_idx, end_emb_idx) inclusive ranges.
    """
    intervals_emb = []
    if not json_path or not os.path.exists(json_path):
        print(f"[INFO] 쉬는 구간 파일이 없음: {json_path}", file=sys.stderr, flush=True)
        return intervals_emb
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 쉬는 구간 JSON 파싱 실패: {json_path} ({e})", file=sys.stderr, flush=True)
        return intervals_emb
    if not isinstance(data, dict):
        print(f"[WARN] 쉬는 구간 JSON의 최상위 형식이 dict가 아님: {type(data)}", file=sys.stderr, flush=True)
        return intervals_emb
    items = data.get(ref_video_basename, [])
    if not items:
        print(f"[INFO] '{ref_video_basename}'에 해당하는 쉬는 구간이 없습니다.", file=sys.stderr, flush=True)
    else:
        print(f"[INFO] '{ref_video_basename}'에 대한 쉬는 구간 {len(items)}개 로드됨:", file=sys.stderr, flush=True)
        for s, e in items:
            print(f"   - {s} ~ {e}", file=sys.stderr, flush=True)
    for pair in items:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        s_sec = parse_mmss_to_seconds(pair[0])
        e_sec = parse_mmss_to_seconds(pair[1])
        if e_sec < s_sec:
            s_sec, e_sec = e_sec, s_sec
        s_frame = int(round(s_sec * max(1e-6, ref_fps)))
        e_frame = int(round(e_sec * max(1e-6, ref_fps)))
        stride = max(1, int(ref_stride))
        s_emb = s_frame // stride
        e_emb = e_frame // stride
        intervals_emb.append((s_emb, e_emb))
    if intervals_emb:
        print(f"[INFO] 변환된 쉬는 구간(임베딩 인덱스 기준): {len(intervals_emb)}개", file=sys.stderr, flush=True)
    return intervals_emb

def in_intervals(idx: int, intervals: list[tuple[int, int]]) -> bool:
    """Return True if idx is inside any (start, end) inclusive interval."""
    for s, e in intervals:
        if s <= idx <= e:
            return True
    return False

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("pip install mediapipe opencv-python numpy")

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
ANGLE_NAMES = ["left knee","right knee","left hip","right hip","left elbow","right elbow"]

# Indices for MediaPipe Pose (ignore face landmarks for similarity)
FACE_IDXS = list(range(0, 11))   # 0..10 (nose, eyes, ears, mouth region)
BODY_IDXS = list(range(11, 33))  # 11..32 (shoulders, arms, torso, legs, feet)


def _safe_get_xy(landmarks, idx):
    if landmarks[idx, 3] < 0.5:
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

# ===== Weighted similarity helpers =====

# 영역별 기본 가중치(원하는 값으로 조정 가능)
REGION_WEIGHTS = {
    "arm": 2.2,     # 팔
    "leg": 2.4,     # 다리
    "torso": 0.6,   # 몸통(어깨폭, 골반폭, 좌/우 연결 등)
}
ANGLE_WEIGHTS = {
    "elbow": 2.0,
    "knee": 2.4,
    "hip": 2.2,
    "shoulder": 0.5,
}
REGION_COLORS = {
    "arm":   (60, 180, 255),   # 주황톤
    "leg":   (60, 255, 60),    # 연초록
    "torso": (200, 200, 200),  # 회색
}

def draw_colored_skeleton(image_bgr, lm33_xyv, thickness=3):
    """가시성>=0.5 인 관절만 선분/원점으로 가시화. BONES를 region 색으로."""
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
    if {a,b,c} & {25,27} and 23 in {a,b,c}: return "knee"
    if {a,b,c} & {26,28} and 24 in {a,b,c}: return "knee"
    # (11,13,15)/(12,14,16) -> elbow
    if {a,b,c} & {13,15}: return "elbow"
    if {a,b,c} & {14,16}: return "elbow"
    # (13,11,23)/(14,12,24) -> shoulder/hip 복합 -> shoulder 쪽으로
    if {a,b,c} & {11,12}: return "shoulder"
    if {a,b,c} & {23,24}: return "hip"
    return "shoulder"

def build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None, vis_thresh=0.5):
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
        # 가시성 보정(선택): 두 관절 모두 보일수록↑
        if lm_for_vis is not None:
            vi = 1.0 if lm_for_vis[i,3] >= vis_thresh else 0.4
            vj = 1.0 if lm_for_vis[j,3] >= vis_thresh else 0.4
            base = base * min(vi, vj)
        # (dx, dy)에 동일 가중
        w.extend([base, base])

    # 2) 각도 성분에 가중치
    for (a,b,c) in ANGLE_TRIPLES:
        region = _angle_region(a,b,c)
        base = ANGLE_WEIGHTS.get(region, 1.0)
        if lm_for_vis is not None:
            va = 1.0 if lm_for_vis[a,3] >= vis_thresh else 0.4
            vb = 1.0 if lm_for_vis[b,3] >= vis_thresh else 0.4
            vc = 1.0 if lm_for_vis[c,3] >= vis_thresh else 0.4
            base = base * min(va, vb, vc)
        w.append(base)

    return np.array(w, dtype=np.float32)

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

class PoseExtractor:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_seg=False):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_seg,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    def infer(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        arr = np.zeros((33,4), dtype=np.float32)
        for i, p in enumerate(lm):
            arr[i,0] = p.x; arr[i,1] = p.y; arr[i,2] = p.z; arr[i,3] = p.visibility
        return arr


# --- New OnlineMatcher class supporting hint index ---
class OnlineMatcher:
    def __init__(self, ref_embs):
        self.ref = ref_embs
        self.T = ref_embs.shape[0]

    def step_with_hint(self, emb, hint_idx: int, search_radius: int = 0):
        """
        Compute similarity against a window centered at hint_idx.
        If search_radius==0, compare only with ref[hint_idx].
        Returns (best_sim, best_global_idx).
        """
        if self.T == 0:
            return 0.0, 0
        hint_idx = int(max(0, min(self.T - 1, hint_idx)))
        if search_radius <= 0:
            sim = cosine_sim(self.ref[hint_idx], emb)
            return float(sim), hint_idx
        lo = max(0, hint_idx - search_radius)
        hi = min(self.T, hint_idx + search_radius + 1)
        candidates = self.ref[lo:hi]
        sims = np.array([cosine_sim(c, emb) for c in candidates])
        k = int(np.argmax(sims))
        return float(sims[k]), lo + k
def read_frame_at(cap, target_idx, current_idx):
    """
    Try to advance to target_idx. If target is far behind, perform a seek.
    Returns (ok, frame, new_current_idx)
    """
    if target_idx < 0:
        target_idx = 0
    # If we are behind by more than 5 frames, seek for efficiency.
    if target_idx < current_idx or (target_idx - current_idx) > 5:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        return ok, frame, target_idx + 1
    # Otherwise, step forward incrementally.
    while current_idx < target_idx:
        ok = cap.grab()
        if not ok:
            return False, None, current_idx
        current_idx += 1
    ok, frame = cap.read()
    return ok, frame, current_idx + 1

# --- Video-to-reference comparison mode ---
def compare_videos(ref_path,
                   user_video,
                   search_radius=3,
                   ema_alpha=0.2,
                   show_feedback=True,
                   ref_video=None,
                   ref_stride=1,
                   loop_ref=False,
                   rtf=1.0,
                   model_complexity=1,
                   w_pose=0.7,
                   w_motion=0.3,
                   time_penalty=0.2,
                   late_grace=3,
                   late_penalty=0.05,
                   rest_json: str | None = None,
                   out_video: str | None = None):
    """
    로컬 사용자 동영상(user_video)과 레퍼런스(임베딩/영상)를 프레임 기준으로 동기화하여 유사도를 비교/시각화.
    - ref_path: extract.py로 만든 레퍼런스 임베딩 .npy
    - user_video: 비교 대상 로컬 동영상 경로 (웹캠 대신 사용)
    - ref_video: 레퍼런스 가이드 영상(선택). 있으면 오른쪽에 표시하고 타임라인 동기화에 사용
    - ref_stride: extract 시 사용한 stride (레퍼런스 임베딩 1스텝 = ref_stride 프레임)
    - rtf: 재생 속도 배수. 1.0이면 사용자 영상의 실제 fps 기준으로 동기화
    - out_video: 지정 시, 시각화 결과를 동영상으로 저장(mp4/h264 등, FourCC 자동 설정)
    """
    # --- Load reference embeddings & (optional) per-frame landmarks for feedback ---
    ref = np.load(ref_path)
    lm_path = ref_path.replace('.npy', '_lm.npy')
    ref_lm = None
    try:
        ref_lm = np.load(lm_path, allow_pickle=True)
    except Exception:
        ref_lm = None

    # --- Open user/local video ---
    ucap = cv2.VideoCapture(user_video)
    if not ucap.isOpened():
        raise SystemExit(f"사용자 영상 열기 실패: {user_video}")
    user_fps = ucap.get(cv2.CAP_PROP_FPS) or 30.0
    user_fps = float(user_fps if user_fps > 1e-3 else 30.0)
    user_total_frames = int(ucap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # --- Prepare reference video (optional) ---
    ref_cap = None
    ref_fps = 30.0
    ref_total_frames = None
    if ref_video is not None and os.path.exists(ref_video):
        ref_cap = cv2.VideoCapture(ref_video)
        if ref_cap.isOpened():
            fps_val = ref_cap.get(cv2.CAP_PROP_FPS)
            if fps_val and fps_val > 1e-3:
                ref_fps = float(fps_val)
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            ref_cap = None

    # --- Pacing: each embedding step corresponds to ref_stride frames of reference video ---
    step_sec = ref_stride / (ref_fps * max(1e-6, rtf))

    # --- REST intervals (ref timeline -> embedding indices) ---
    rest_intervals_emb = []
    if rest_json and ref_video is not None:
        ref_base = os.path.basename(ref_video)
        rest_intervals_emb = load_rest_intervals_json(rest_json, ref_base, ref_fps, ref_stride)

    # --- Pose extractor ---
    pe = PoseExtractor(static_image_mode=False, model_complexity=model_complexity)
    matcher = OnlineMatcher(ref)

    # --- Optional video writer ---
    writer = None
    if out_video:
        # Probe a first frame to get size
        ok_u, uframe = ucap.read()
        if not ok_u:
            ucap.release()
            if ref_cap is not None:
                ref_cap.release()
            raise SystemExit("사용자 영상에서 프레임을 읽을 수 없습니다.")
        # Prepare a matching ref frame for width/height
        display_right = None
        if ref_cap is not None:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_r, rframe = ref_cap.read()
            if ok_r:
                display_right = rframe
        h, w = uframe.shape[:2]
        if display_right is not None:
            rh, rw = display_right.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            out_w = uframe.shape[1] + display_right.shape[1]
            out_h = h
        else:
            out_w, out_h = w, h
        # Reset user cap to frame 0 for actual processing
        ucap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video, fourcc, user_fps, (out_w, out_h))

    # --- Main loop over user video frames ---
    prev_live_emb = None
    prev_ref_emb = None
    score_ema = 0.0
    last_feedback = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    ref_frame_cur = 0
    user_idx = 0

    while True:
        ok_u, uframe = ucap.read()
        if not ok_u:
            break

        # Compute synchronized hint index from *user timeline* (no wall-clock)
        elapsed_user_sec = (user_idx / user_fps)
        hint_idx = int(elapsed_user_sec / step_sec)
        hint_idx = min(hint_idx, ref.shape[0] - 1)
        if ref_lm is not None:
            hint_idx = min(hint_idx, len(ref_lm) - 1)

        # If REST interval on ref timeline, overlay and skip scoring
        if in_intervals(hint_idx, rest_intervals_emb):
            display_left = uframe
            display_right = None
            if ref_cap is not None:
                target_ref_frame = hint_idx * ref_stride
                ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
                if ok_ref:
                    display_right = ref_frame
            # compose
            h, w = display_left.shape[:2]
            if display_right is not None:
                rh, rw = display_right.shape[:2]
                if rh != h:
                    scale = h / float(rh)
                    display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display_left, display_right])
            else:
                combined = display_left
            cv2.putText(combined, "REST", (20, 40), font, 1.4, (255, 200, 200), 3, cv2.LINE_AA)
            if ref_cap is not None:
                cv2.putText(combined, "REFERENCE", (display_left.shape[1] + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            if writer is not None:
                writer.write(combined)
            else:
                cv2.imshow("Fitness Dance - Video Compare (left: USER, right: REFERENCE)", combined)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
            user_idx += 1
            continue

        # Pose on user frame
        arr = pe.infer(uframe)
        if arr is not None:

            # arr: (33,4) with x,y,z,visibility (x,y는 0~1 정규화)
            overlay = uframe.copy()
            draw_colored_skeleton(overlay, arr)  # 원본 좌표 기준
            # 혹은 normalize 후 좌표로 그릴거면 normalize_landmarks(arr.copy())의 x,y가 이미 정규화된 좌표이므로
            # 화면 합성은 별도 좌표계 주의!
            alpha = 0.5
            uframe = cv2.addWeighted(overlay, alpha, uframe, 1 - alpha, 0)

            lm = normalize_landmarks(arr.copy())
            emb = pose_embedding(lm)
            sim, ref_idx = matcher.step_with_hint(emb, hint_idx=hint_idx, search_radius=search_radius)

            # Build per-feature weights (use ref landmarks if available at this index)
            if ref_lm is not None and ref_idx < len(ref_lm):
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=ref_lm[ref_idx])
            else:
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)

            static_sim = weighted_cosine(matcher.ref[ref_idx], emb, w_feat)
            motion_sim = 0.0
            motion_mag_match = 1.0
            ref_emb = matcher.ref[ref_idx]

            if prev_live_emb is not None and prev_ref_emb is not None:
                d_live = emb - prev_live_emb
                d_ref  = ref_emb - prev_ref_emb
                motion_sim = weighted_cosine(d_live, d_ref, w_feat)
                n_live = np.linalg.norm(d_live)
                n_ref  = np.linalg.norm(d_ref)
                if n_live > 1e-6 or n_ref > 1e-6:
                    motion_mag_match = min(n_live, n_ref) / (max(n_live, n_ref) + 1e-8)
                else:
                    motion_mag_match = 1.0

            # alignment penalty (late-friendly)
            if search_radius > 0:
                delta = ref_idx - hint_idx
                if delta < 0:
                    late_dt = -delta
                    if late_dt <= late_grace:
                        align_factor = 1.0
                    else:
                        span = max(1, search_radius - late_grace)
                        align_factor = 1.0 - late_penalty * ((late_dt - late_grace) / span)
                elif delta > 0:
                    span = max(1, search_radius)
                    align_factor = 1.0 - time_penalty * (delta / span)
                else:
                    align_factor = 1.0
                align_factor = max(0.0, min(1.0, align_factor))
            else:
                align_factor = 1.0

            blended = (w_pose * static_sim) + (w_motion * motion_sim * motion_mag_match)
            score = ((blended + 1.0) * 0.5) * align_factor

            #normalize
            #score = np.clip((score - 50) / 45 * 100, 0, 100)
            score_ema = exp_moving_avg(score_ema, score, alpha=ema_alpha)

            prev_live_emb = emb
            prev_ref_emb  = ref_emb

            msgs = []
            if show_feedback and ref_lm is not None and ref_idx < len(ref_lm):
                msgs = angle_feedback(lm, ref_lm[ref_idx], angle_tol_deg=10.0)
                last_feedback = msgs if len(msgs)>0 else ["Good! Nice pose 👏"]


        else:
            static_sim = 0.0
            motion_sim = 0.0
            motion_mag_match = 1.0
            last_feedback = ["cannot extract pose"]

        # Prepare right (reference frame)
        display_right = None
        if ref_cap is not None:
            target_ref_frame = hint_idx * ref_stride
            ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
            if ok_ref:
                display_right = ref_frame

        # Compose and overlay
        display_left = uframe
        h, w = display_left.shape[:2]
        if display_right is not None:
            rh, rw = display_right.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            combined = np.hstack([display_left, display_right])
        else:
            combined = display_left

        # --- Rolling similarity buffer for stability ---
        if 'score_window' not in locals():
            score_window = []
        score_window.append(score_ema)
        if len(score_window) > int(user_fps * 3):  # 3초 평균
            score_window.pop(0)
        avg_score = float(np.nan_to_num(np.mean(score_window) if len(score_window) > 0 else score_ema, nan=0.0, posinf=1.0, neginf=0.0))

        avg_50_95 = avg_score * 100
        avg_pct = (avg_50_95 - 50) / 45.0
        avg_pct = avg_pct * 100.0

        grade_text = ""
        grade_color = (255, 255, 255)
        if avg_pct >= 70:
            grade_text = "PERFECT!"
            grade_color = (0, 255, 0)
        elif avg_pct >= 65:
            grade_text = "GOOD"
            grade_color = (0, 200, 255)
        else:
            grade_text = "BAD"
            grade_color = (255, 0, 0)

        cv2.putText(combined, f"Similarity(avg): {avg_pct:.1f}%  {avg_score*100:.1f}%", (20, 40), font, 1.0, (0,255,0), 2, cv2.LINE_AA)
        if grade_text:
            cv2.putText(combined, grade_text, (20, 90), font, 1.4, grade_color, 3, cv2.LINE_AA)
        y0 = 120
        for i, m in enumerate(last_feedback[:3]):
            cv2.putText(combined, m, (20, y0 + i*30), font, 0.8, (0,200,255), 2, cv2.LINE_AA)
        cv2.putText(combined, f"pose={static_sim:+.2f}  motion={motion_sim:+.2f}  mag={motion_mag_match:.2f}",
                    (20, y0 + 3*30), font, 0.6, (200,200,200), 1, cv2.LINE_AA)
        if ref_cap is not None:
            cv2.putText(combined, "REFERENCE", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(combined)
        else:
            cv2.imshow("Fitness Dance - Video Compare (left: USER, right: REFERENCE)", combined)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break

        user_idx += 1

        # End of embeddings handling: if hint exceeds, loop ref video timeline or stop
        if hint_idx >= ref.shape[0] - 1:
            if loop_ref and ref_cap is not None:
                if ref_total_frames:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ref_frame_cur = 0
            else:
                # No loop: keep showing last ref frame; processing continues until user video ends
                pass

    # cleanup
    ucap.release()
    if ref_cap is not None:
        ref_cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

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
            hint = "Jom deo peogi" if curr_angles[i] > ref_angles[i] else "Jeom deo gup"
            msgs.append(f"{ANGLE_NAMES[i]} {hint} (Difference {d:.1f}°)")
    return msgs



def live_play_1(ref_path, camera=0, search_radius=3, ema_alpha=0.2, show_feedback=True,
              ref_video=None, ref_stride=1, loop_ref=True, rtf=1.0, model_complexity=1,
              warmup_sec=5.0, countdown_color="0,215,255", countdown_beep=True, play_ref_audio=True,
              w_pose=0.7, w_motion=0.3, time_penalty=0.2, late_grace=3, late_penalty=0.05,
              rest_json: str | None = None):
    """
    - ref_path: npy embeddings generated by extract.py
    - ref_video: path to the same reference video file for visual guidance (optional but recommended)
    - ref_stride: stride used during extraction (must match extract.py --stride)
    - rtf: real-time factor for pacing (1.0 = exact ref speed; 1.2 = 20% faster)
    """
    ref = np.load(ref_path)
    lm_path = ref_path.replace('.npy', '_lm.npy')
    ref_lm = None
    try:
        ref_lm = np.load(lm_path, allow_pickle=True)
    except Exception:
        ref_lm = None

    # Prepare reference video for display + pacing
    ref_cap = None
    ref_fps = 30.0
    ref_total_frames = None
    if ref_video is not None and os.path.exists(ref_video):
        ref_cap = cv2.VideoCapture(ref_video)
        if ref_cap.isOpened():
            fps_val = ref_cap.get(cv2.CAP_PROP_FPS)
            if fps_val and fps_val > 1e-3:
                ref_fps = float(fps_val)
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            ref_cap = None  # fallback to no video

    # Derived pacing
    # Each embedding corresponds to ref_stride frames of reference video.
    # Interval (seconds) per embedding step:
    step_sec = ref_stride / (ref_fps * max(1e-6, rtf))

    # Load user-specified REST intervals from JSON and convert to embedding indices
    rest_intervals_emb = []
    if rest_json and ref_video is not None:
        ref_base = os.path.basename(ref_video)
        rest_intervals_emb = load_rest_intervals_json(rest_json, ref_base, ref_fps, ref_stride)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"카메라 열기 실패: {camera}")

    pe = PoseExtractor(static_image_mode=False, model_complexity=model_complexity)
    matcher = OnlineMatcher(ref)
    prev_live_emb = None
    prev_ref_emb = None

    score_ema = 0.0
    last_feedback = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    cd_color = parse_bgr(countdown_color, default=(0,215,255))
    last_beep_whole = None  # to avoid beeping every frame
    ref_audio_proc = None

    # Timers: total (for warmup) and sync (for comparison after warmup)
    start_total = time.perf_counter()
    sync_start_t = None
    warmup_done = False

    ref_frame_cur = 0  # track where the ref_cap currently is
    window_name = "Fitness Dance - Live (left: YOU, right: REFERENCE)"

    while True:
        now = time.perf_counter()
        elapsed_total = now - start_total

        # Read camera
        ok, frame = cap.read()
        if not ok:
            break
        display_left = frame.copy()

        # Defaults for debug overlay
        static_sim = 0.0
        motion_sim = 0.0
        motion_mag_match = 1.0

        # ---- Warmup phase: delay ref video & similarity for warmup_sec ----
        if not warmup_done and elapsed_total < warmup_sec:
            # Prepare right panel: show first frame of reference if available (paused)
            display_right = None
            if ref_cap is not None:
                # Ensure we are at frame 0 and fetch one frame
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_ref, ref_frame = ref_cap.read()
                if ok_ref:
                    display_right = ref_frame
            # Compose side-by-side
            h, w = display_left.shape[:2]
            if display_right is not None:
                rh, rw = display_right.shape[:2]
                if rh != h:
                    scale = h / float(rh)
                    display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display_left, display_right])
            else:
                combined = display_left
            # Countdown overlay
            remain = max(0.0, warmup_sec - elapsed_total)
            cv2.putText(combined, f"준비하세요... {remain:0.1f}s", (20, 40), font, 1.0, cd_color, 2, cv2.LINE_AA)

            # Countdown beeps at 3, 2, 1 and a higher tone at 0
            if countdown_beep:
                whole = int(math.ceil(remain))
                if whole != last_beep_whole and 1 <= whole <= 3:
                    # beep: 3->800Hz, 2->900Hz, 1->1000Hz
                    freq = 700.0 + 100.0*whole
                    play_beep(freq=freq, dur=0.15)
                    last_beep_whole = whole
                if remain <= 0.05 and last_beep_whole != 0:
                    play_beep(freq=1400.0, dur=0.25)
                    last_beep_whole = 0

            if ref_cap is not None:
                cv2.putText(combined, "REFERENCE (5초 후 시작)", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            continue

        # Transition from warmup to sync start
        if not warmup_done and elapsed_total >= warmup_sec:
            warmup_done = True
            sync_start_t = time.perf_counter()
            score_ema = 0.0
            last_feedback = ["Start! Follow the reference video 💪"]
            if ref_cap is not None:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ref_frame_cur = 0
            # Start reference audio playback
            if play_ref_audio and ref_video is not None and os.path.exists(ref_video):
                stop_ref_audio_player(ref_audio_proc)
                ref_audio_proc = start_ref_audio_player(ref_video_path=ref_video, start_sec=0.0)
            # proceed to next loop iteration to start synced timing cleanly
            continue

        # Compute target embedding index aligned with reference cadence (after warmup)
        elapsed = time.perf_counter() - sync_start_t if sync_start_t is not None else 0.0
        hint_idx = int(elapsed / step_sec)
        if ref_lm is not None:
            hint_idx = min(hint_idx, len(ref_lm) - 1)
        hint_idx = min(hint_idx, ref.shape[0] - 1)

        # If current synchronized reference index falls in a REST interval, show REST overlay and skip comparison
        if in_intervals(hint_idx, rest_intervals_emb):
            # Prepare right panel (paused reference at current hint index)
            display_right = None
            if ref_cap is not None:
                target_ref_frame = hint_idx * ref_stride
                ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
                if ok_ref:
                    display_right = ref_frame
            # Compose side-by-side
            h, w = display_left.shape[:2]
            if display_right is not None:
                rh, rw = display_right.shape[:2]
                if rh != h:
                    scale = h / float(rh)
                    display_right = cv2.resize(display_right, (int(rw * scale), h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display_left, display_right])
            else:
                combined = display_left
            # REST overlay
            cv2.putText(combined, "REST", (20, 40), font, 1.4, (255, 200, 200), 3, cv2.LINE_AA)
            if ref_cap is not None:
                cv2.putText(combined, "REFERENCE", (w + 20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            # Do not update EMA/state when resting
            continue

        # Pose on live frame
        arr = pe.infer(frame)
        if arr is not None:
            lm = normalize_landmarks(arr.copy())
            emb = pose_embedding(lm)

            # Compare to reference at the synchronized index (with small tolerance)
            sim, ref_idx = matcher.step_with_hint(emb, hint_idx=hint_idx, search_radius=search_radius)

            # Build per-feature weights (use ref landmarks if available at this index)
            if ref_lm is not None and ref_idx < len(ref_lm):
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=ref_lm[ref_idx])
            else:
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)

            # --- Motion-aware similarity ---
            static_sim = weighted_cosine(matcher.ref[ref_idx], emb, w_feat)
            motion_sim = 0.0
            motion_mag_match = 1.0
            ref_emb = matcher.ref[ref_idx]

            if prev_live_emb is not None and prev_ref_emb is not None:
                d_live = emb - prev_live_emb
                d_ref  = ref_emb - prev_ref_emb
                motion_sim = weighted_cosine(d_live, d_ref, w_feat)
                # magnitude agreement: discourage being still when ref is moving (and vice versa)
                n_live = np.linalg.norm(d_live)
                n_ref  = np.linalg.norm(d_ref)
                if n_live > 1e-6 or n_ref > 1e-6:
                    motion_mag_match = min(n_live, n_ref) / (max(n_live, n_ref) + 1e-8)
                else:
                    motion_mag_match = 1.0  # both still -> neutral
            # Asymmetric time penalty (late-friendly):
            # - If user is LATE (ref_idx < hint_idx): allow 'late_grace' frames with zero penalty,
            #   then apply a smaller 'late_penalty'.
            # - If user is EARLY (ref_idx > hint_idx): apply the regular 'time_penalty'.
            if search_radius > 0:
                delta = ref_idx - hint_idx  # negative when user is late (behind the reference)
                if delta < 0:
                    # LATE case
                    late_dt = -delta
                    if late_dt <= late_grace:
                        align_factor = 1.0
                    else:
                        span = max(1, search_radius - late_grace)
                        align_factor = 1.0 - late_penalty * ((late_dt - late_grace) / span)
                elif delta > 0:
                    # EARLY case
                    span = max(1, search_radius)
                    align_factor = 1.0 - time_penalty * (delta / span)
                else:
                    align_factor = 1.0
                align_factor = max(0.0, min(1.0, align_factor))
            else:
                align_factor = 1.0

            # final blended similarity in [-1, 1]
            blended = (w_pose * static_sim) + (w_motion * motion_sim * motion_mag_match)
            # map to [0,1] and apply alignment factor
            score = ((blended + 1.0) * 0.5) * align_factor

            score_ema = exp_moving_avg(score_ema, score, alpha=ema_alpha)

            # update previous embeddings
            prev_live_emb = emb
            prev_ref_emb  = ref_emb

            msgs = []
            if show_feedback and ref_lm is not None and ref_idx < len(ref_lm):
                msgs = angle_feedback(lm, ref_lm[ref_idx], angle_tol_deg=10.0)
                last_feedback = msgs if len(msgs)>0 else ["Good! Nice pose 👏"]
        else:
            last_feedback = ["cannot extract pose"]

        # Prepare right panel (reference video frame)
        display_right = None
        if ref_cap is not None:
            # Compute the approximate source frame in the reference video
            target_ref_frame = hint_idx * ref_stride
            ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
            if not ok_ref:
                if loop_ref and ref_cap is not None and ref_total_frames:
                    # Loop: reset timeline
                    start_t = time.perf_counter()
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ref_frame_cur = 0
                    ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, 0, ref_frame_cur)
                else:
                    ref_frame = None
            if ref_frame is not None:
                display_right = ref_frame

        # Compose side-by-side
        h, w = display_left.shape[:2]
        if display_right is not None:
            # Resize right to match height
            rh, rw = display_right.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            combined = np.hstack([display_left, display_right])
        else:
            combined = display_left

        # Draw overlays (left side text)
        cv2.putText(combined, f"Similarity: {score_ema*100:.1f}%", (20, 40), font, 1.0, (0,255,0), 2, cv2.LINE_AA)
        y0 = 80
        for i, m in enumerate(last_feedback[:3]):
            cv2.putText(combined, m, (20, y0 + i*30), font, 0.8, (0,200,255), 2, cv2.LINE_AA)

        # Debug info for similarity components (optional; small font)
        cv2.putText(combined, f"pose={static_sim:+.2f}  motion={motion_sim:+.2f}  mag={motion_mag_match:.2f}",
                    (20, y0 + 3*30), font, 0.6, (200,200,200), 1, cv2.LINE_AA)

        if ref_cap is not None:
            cv2.putText(combined, "REFERENCE", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Fitness Dance - Live (left: YOU, right: REFERENCE)", combined)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        # If we reached the end of embeddings, loop or stop
        if hint_idx >= ref.shape[0] - 1:
            if loop_ref:
                sync_start_t = time.perf_counter()
                if ref_cap is not None:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ref_frame_cur = 0
                if play_ref_audio and ref_video is not None and os.path.exists(ref_video):
                    stop_ref_audio_player(ref_audio_proc)
                    ref_audio_proc = start_ref_audio_player(ref_video_path=ref_video, start_sec=0.0)
            else:
                break

    cap.release()
    if ref_cap is not None:
        ref_cap.release()
    # Stop audio player if running
    stop_ref_audio_player(ref_audio_proc)
    cv2.destroyAllWindows()

def live_play(ref_path, camera=0, search_radius=3, ema_alpha=0.2, show_feedback=True,
              ref_video=None, ref_stride=1, loop_ref=True, rtf=1.0, model_complexity=1,
              warmup_sec=5.0, countdown_color="0,215,255", countdown_beep=True, play_ref_audio=True,
              w_pose=0.7, w_motion=0.3, time_penalty=0.2, late_grace=3, late_penalty=0.05,
              rest_json: str | None = None,
              out_video: str | None = None,
              overlay_alpha: float = 0.5):
    """
    compare_videos()와 동일한 로직/연출을 유지하고,
    입력만 실시간 카메라로 바꾼 버전.
    """
    ref = np.load(ref_path)
    lm_path = ref_path.replace('.npy', '_lm.npy')
    try:
        ref_lm = np.load(lm_path, allow_pickle=True)
    except Exception:
        ref_lm = None

    # --- Reference video 준비 (동일) ---
    ref_cap = None
    ref_fps = 30.0
    ref_total_frames = None
    if ref_video is not None and os.path.exists(ref_video):
        ref_cap = cv2.VideoCapture(ref_video)
        if ref_cap.isOpened():
            fps_val = ref_cap.get(cv2.CAP_PROP_FPS)
            if fps_val and fps_val > 1e-3:
                ref_fps = float(fps_val)
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            ref_cap = None

    # 임베딩 한 스텝의 시간 간격(동일)
    step_sec = ref_stride / (ref_fps * max(1e-6, rtf))

    # --- 쉬는 구간 로드(동일) ---
    rest_intervals_emb = []
    if rest_json and ref_video is not None:
        ref_base = os.path.basename(ref_video)
        rest_intervals_emb = load_rest_intervals_json(rest_json, ref_base, ref_fps, ref_stride)

    # --- 카메라 ---
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"카메라 열기 실패: {camera}")

    # --- out_video 준비 (compare_videos와 동일한 합성폭으로) ---
    writer = None
    out_w = out_h = None
    if out_video:
        # 카메라 한 프레임 읽어 크기 파악
        ok_probe, probe = cap.read()
        if not ok_probe:
            cap.release()
            if ref_cap is not None:
                ref_cap.release()
            raise SystemExit("카메라 프레임을 읽을 수 없습니다.")
        # 우측 ref 첫 프레임
        display_right0 = None
        if ref_cap is not None:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_r, rframe0 = ref_cap.read()
            if ok_r:
                display_right0 = rframe0
        h, w = probe.shape[:2]
        if display_right0 is not None:
            rh, rw = display_right0.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right0 = cv2.resize(display_right0, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            out_w = w + display_right0.shape[1]
            out_h = h
        else:
            out_w, out_h = w, h
        # 프레임레이트는 ref_fps 또는 30으로 저장
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video, fourcc, ref_fps if ref_fps > 1e-3 else 30.0, (out_w, out_h))
        # ref_cap 포지션 복구
        if ref_cap is not None:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pe = PoseExtractor(static_image_mode=False, model_complexity=model_complexity)
    matcher = OnlineMatcher(ref)

    prev_live_emb = None
    prev_ref_emb  = None
    score_ema = 0.0
    score_window = []  # ← compare_videos와 동일: 롤링 평균 버퍼
    last_feedback = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    cd_color = parse_bgr(countdown_color, default=(0,215,255))
    last_beep_whole = None
    ref_audio_proc = None

    start_total = time.perf_counter()
    sync_start_t = None
    warmup_done = False
    ref_frame_cur = 0
    window_name = "Fitness Dance - Live (left: YOU, right: REFERENCE)"

    while True:
        now = time.perf_counter()
        elapsed_total = now - start_total

        ok, frame = cap.read()
        if not ok:
            break
        display_left = frame.copy()

        # ----- Warmup (그대로 유지) -----
        static_sim = 0.0
        motion_sim = 0.0
        motion_mag_match = 1.0

        if not warmup_done and elapsed_total < warmup_sec:
            display_right = None
            if ref_cap is not None:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_ref, ref_frame = ref_cap.read()
                if ok_ref:
                    display_right = ref_frame
            h, w = display_left.shape[:2]
            if display_right is not None:
                rh, rw = display_right.shape[:2]
                if rh != h:
                    scale = h / float(rh)
                    display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display_left, display_right])
            else:
                combined = display_left
            remain = max(0.0, warmup_sec - elapsed_total)
            cv2.putText(combined, f"준비하세요... {remain:0.1f}s", (20, 40), font, 1.0, cd_color, 2, cv2.LINE_AA)

            if countdown_beep:
                whole = int(math.ceil(remain))
                if whole != last_beep_whole and 1 <= whole <= 3:
                    play_beep(freq=700.0 + 100.0*whole, dur=0.15)
                    last_beep_whole = whole
                if remain <= 0.05 and last_beep_whole != 0:
                    play_beep(freq=1400.0, dur=0.25); last_beep_whole = 0

            if ref_cap is not None:
                cv2.putText(combined, "REFERENCE (5초 후 시작)", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, combined)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
            # out_video에는 워밍업 화면 저장 안 함(원하면 여기에서 writer.write(combined))
            continue

        if not warmup_done and elapsed_total >= warmup_sec:
            warmup_done = True
            sync_start_t = time.perf_counter()
            score_ema = 0.0
            score_window.clear()
            last_feedback = ["Start! Follow the reference video 💪"]
            if ref_cap is not None:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ref_frame_cur = 0
            if play_ref_audio and ref_video is not None and os.path.exists(ref_video):
                stop_ref_audio_player(ref_audio_proc)
                ref_audio_proc = start_ref_audio_player(ref_video_path=ref_video, start_sec=0.0)
            continue

        # ---- 동기 인덱스 ----
        elapsed = time.perf_counter() - sync_start_t if sync_start_t is not None else 0.0
        hint_idx = int(elapsed / step_sec)
        if ref_lm is not None: hint_idx = min(hint_idx, len(ref_lm) - 1)
        hint_idx = min(hint_idx, ref.shape[0] - 1)

        # ---- 쉬는 구간 처리(동일) ----
        if in_intervals(hint_idx, rest_intervals_emb):
            display_right = None
            if ref_cap is not None:
                target_ref_frame = hint_idx * ref_stride
                ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
                if ok_ref: display_right = ref_frame
            h, w = display_left.shape[:2]
            if display_right is not None:
                rh, rw = display_right.shape[:2]
                if rh != h:
                    scale = h / float(rh)
                    display_right = cv2.resize(display_right, (int(rw * scale), h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display_left, display_right])
            else:
                combined = display_left
            cv2.putText(combined, "REST", (20, 40), font, 1.4, (255,200,200), 3, cv2.LINE_AA)
            if ref_cap is not None:
                cv2.putText(combined, "REFERENCE", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            if writer is not None: writer.write(combined)
            cv2.imshow(window_name, combined)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
            continue

        # ---- 포즈 추론 ----
        arr = pe.infer(frame)
        if arr is not None:
            # (1) 스켈레톤 오버레이( compare_videos와 동일 )
            overlay = display_left.copy()
            draw_colored_skeleton(overlay, arr)
            display_left = cv2.addWeighted(overlay, overlay_alpha, display_left, 1 - overlay_alpha, 0)

            # (2) 임베딩 및 매칭
            lm = normalize_landmarks(arr.copy())
            emb = pose_embedding(lm)
            sim, ref_idx = matcher.step_with_hint(emb, hint_idx=hint_idx, search_radius=search_radius)

            # (3) 가중치, 정적/모션 유사도
            if ref_lm is not None and ref_idx < len(ref_lm):
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=ref_lm[ref_idx])
            else:
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)

            static_sim = weighted_cosine(matcher.ref[ref_idx], emb, w_feat)
            motion_sim = 0.0; motion_mag_match = 1.0
            ref_emb = matcher.ref[ref_idx]
            if prev_live_emb is not None and prev_ref_emb is not None:
                d_live = emb - prev_live_emb
                d_ref  = ref_emb - prev_ref_emb
                motion_sim = weighted_cosine(d_live, d_ref, w_feat)
                n_live = np.linalg.norm(d_live); n_ref = np.linalg.norm(d_ref)
                if n_live > 1e-6 or n_ref > 1e-6:
                    motion_mag_match = min(n_live, n_ref) / (max(n_live, n_ref) + 1e-8)

            # (4) 시간 이탈 패널티(동일)
            if search_radius > 0:
                delta = ref_idx - hint_idx
                if delta < 0:
                    late_dt = -delta
                    if late_dt <= late_grace: align_factor = 1.0
                    else:
                        span = max(1, search_radius - late_grace)
                        align_factor = 1.0 - late_penalty * ((late_dt - late_grace) / span)
                elif delta > 0:
                    span = max(1, search_radius)
                    align_factor = 1.0 - time_penalty * (delta / span)
                else:
                    align_factor = 1.0
                align_factor = max(0.0, min(1.0, align_factor))
            else:
                align_factor = 1.0

            blended = (w_pose * static_sim) + (w_motion * motion_sim * motion_mag_match)
            score = ((blended + 1.0) * 0.5) * align_factor

            score_ema = exp_moving_avg(score_ema, score, alpha=ema_alpha)

            prev_live_emb = emb
            prev_ref_emb  = ref_emb

            if show_feedback and ref_lm is not None and ref_idx < len(ref_lm):
                msgs = angle_feedback(lm, ref_lm[ref_idx], angle_tol_deg=10.0)
                last_feedback = msgs if len(msgs)>0 else ["Good! Nice pose 👏"]
        else:
            last_feedback = ["cannot extract pose"]

        # ---- 우측 ref 프레임 준비 (동일) ----
        display_right = None
        if ref_cap is not None:
            target_ref_frame = hint_idx * ref_stride
            ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
            if not ok_ref and loop_ref and ref_total_frames:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ref_frame_cur = 0
                ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, 0, ref_frame_cur)
            if ok_ref:
                display_right = ref_frame

        # ---- 합성 & 오버레이( compare_videos와 완전 동일 ) ----
        h, w = display_left.shape[:2]
        if display_right is not None:
            rh, rw = display_right.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            combined = np.hstack([display_left, display_right])
        else:
            combined = display_left

        # 롤링 평균(3초) + 50~95 -> 0~100 정규화 & 등급
        score_window.append(score_ema)
        # 카메라 fps를 모를 수 있어 ref_fps 기준 3초 윈도우로 사용
        if len(score_window) > int(ref_fps * 3):
            score_window.pop(0)
        avg_score = float(np.nan_to_num(np.mean(score_window) if len(score_window)>0 else score_ema, nan=0.0, posinf=1.0, neginf=0.0))

        avg_50_95 = avg_score * 100.0
        avg_pct   = ((avg_50_95 - 50.0) / 45.0) * 100.0
        # 보기 좋게 0~100로 클립
        avg_pct   = float(np.clip(avg_pct, 0.0, 100.0))

        grade_text, grade_color = "", (255,255,255)
        if avg_pct >= 65:
            grade_text, grade_color = "PERFECT!", (0,255,0)
        elif avg_pct >= 60:
            grade_text, grade_color = "GOOD", (0,200,255)
        else:
            grade_text, grade_color = "BAD", (255,0,0)

        cv2.putText(combined, f"Similarity(avg): {avg_pct:.1f}%  {avg_score*100:.1f}%", (20, 40), font, 1.0, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(combined, grade_text, (20, 90), font, 1.4, grade_color, 3, cv2.LINE_AA)
        y0 = 120
        for i, m in enumerate(last_feedback[:3]):
            cv2.putText(combined, m, (20, y0 + i*30), font, 0.8, (0,200,255), 2, cv2.LINE_AA)
        cv2.putText(combined, f"pose={static_sim:+.2f}  motion={motion_sim:+.2f}  mag={motion_mag_match:.2f}",
                    (20, y0 + 3*30), font, 0.6, (200,200,200), 1, cv2.LINE_AA)
        if ref_cap is not None:
            cv2.putText(combined, "REFERENCE", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)

        if writer is not None:
            if out_w is None or out_h is None:
                out_h, out_w = combined.shape[:2]
            writer.write(combined)

        cv2.imshow(window_name, combined)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

        # 임베딩 끝 처리(동일)
        if hint_idx >= ref.shape[0] - 1:
            if loop_ref:
                sync_start_t = time.perf_counter()
                if ref_cap is not None:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ref_frame_cur = 0
                if play_ref_audio and ref_video is not None and os.path.exists(ref_video):
                    stop_ref_audio_player(ref_audio_proc)
                    ref_audio_proc = start_ref_audio_player(ref_video_path=ref_video, start_sec=0.0)
            else:
                break

    cap.release()
    if ref_cap is not None:
        ref_cap.release()
    stop_ref_audio_player(ref_audio_proc)
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=str, required=True, help="레퍼런스 임베딩 .npy 경로")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--search_radius", type=int, default=0)
    ap.add_argument("--ema_alpha", type=float, default=0.9)
    ap.add_argument("--no_feedback", action="store_true")
    ap.add_argument("--ref_video", type=str, default=None, help="레퍼런스 영상 파일 경로(동시 표시 및 동기화)")
    ap.add_argument("--ref_stride", type=int, default=2, help="extract.py에서 사용한 stride 값 (동기화용)")
    ap.add_argument("--loop_ref", action="store_true", help="레퍼런스 재생을 반복(loop)")
    ap.add_argument("--rtf", type=float, default=1.0, help="재생 속도 배수 (1.0=원본)")
    ap.add_argument("--w_pose", type=float, default=1.0, help="정적 포즈 임베딩 유사도 가중치")
    ap.add_argument("--w_motion", type=float, default=0.0, help="모션(프레임 간 변화) 유사도 가중치")
    ap.add_argument("--time_penalty", type=float, default=0.0, help="윈도우 검색 시 시간 이탈 패널티 가중치(0~1)")
    ap.add_argument("--late_grace", type=int, default=5, help="늦게 따라할 때 무패널티 허용 프레임 수 (search_radius 기준)")
    ap.add_argument("--late_penalty", type=float, default=0.05, help="늦게 따라할 때(레퍼런스보다 뒤) 시간 이탈 패널티 가중치")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0,1,2], help="MediaPipe Pose 모델 복잡도")
    ap.add_argument("--warmup_sec", type=float, default=5.0, help="시작 후 N초 뒤에 레퍼런스 재생/유사도 비교 시작")
    ap.add_argument("--countdown_color", type=str, default="0,215,255", help="카운트다운 글자 B,G,R (예: 0,215,255)")
    ap.add_argument("--countdown_beep",default=True,  action="store_true", help="카운트다운 삐- 소리 재생")
    ap.add_argument("--play_ref_audio", default= True, action="store_true", help="레퍼런스 영상의 오디오 재생(ffplay/afplay 필요)")
    ap.add_argument("--rest_json", type=str, default="data/rest_exc.json", help="영상별 쉬는 구간 JSON 파일 경로 (예: rest_intervals.json)")
    ap.add_argument("--user_video", type=str, default=None, help="웹캠 대신 비교할 사용자 로컬 동영상 경로")
    ap.add_argument("--out_video", type=str, default=None, help="시각화 결과 저장 경로 (예: output.mp4)")
    args = ap.parse_args()
    if args.user_video:
        compare_videos(
            ref_path=args.ref,
            user_video=args.user_video,
            search_radius=args.search_radius,
            ema_alpha=args.ema_alpha,
            show_feedback=(not args.no_feedback),
            ref_video=args.ref_video,
            ref_stride=args.ref_stride,
            loop_ref=args.loop_ref,
            rtf=args.rtf,
            model_complexity=args.model_complexity,
            w_pose=args.w_pose,
            w_motion=args.w_motion,
            time_penalty=args.time_penalty,
            late_grace=args.late_grace,
            late_penalty=args.late_penalty,
            rest_json=args.rest_json,
            out_video=args.out_video,
        )
    else:
        live_play(
            args.ref,
            camera=args.camera,
            search_radius=args.search_radius,
            ema_alpha=args.ema_alpha,
            show_feedback=(not args.no_feedback),
            ref_video=args.ref_video,
            ref_stride=args.ref_stride,
            loop_ref=args.loop_ref,
            rtf=args.rtf,
            model_complexity=args.model_complexity,
            warmup_sec=args.warmup_sec,
            countdown_color=args.countdown_color,
            countdown_beep=args.countdown_beep,
            play_ref_audio=args.play_ref_audio,
            w_pose=args.w_pose,
            w_motion=args.w_motion,
            time_penalty=args.time_penalty,
            late_grace=args.late_grace,
            late_penalty=args.late_penalty,
            rest_json=args.rest_json
        )
