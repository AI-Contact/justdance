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
ANGLE_NAMES = ["왼무릎","오른무릎","왼엉덩","오른엉덩","왼팔꿈치","오른팔꿈치"]

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
            hint = "좀 더 펴기" if curr_angles[i] > ref_angles[i] else "좀 더 굽히기"
            msgs.append(f"{ANGLE_NAMES[i]} {hint} (차이 {d:.1f}°)")
    return msgs



def live_play(ref_path, camera=0, search_radius=3, ema_alpha=0.2, show_feedback=True,
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
            last_feedback = ["시작! 레퍼런스를 따라 해보세요 💪"]
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

            # --- Motion-aware similarity ---
            static_sim = sim  # cosine between current pose embeddings
            motion_sim = 0.0
            motion_mag_match = 1.0
            ref_emb = matcher.ref[ref_idx]

            if prev_live_emb is not None and prev_ref_emb is not None:
                d_live = emb - prev_live_emb
                d_ref  = ref_emb - prev_ref_emb
                motion_sim = cosine_sim(d_live, d_ref)
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
                last_feedback = msgs if len(msgs)>0 else ["굿! 자세 좋아요 👏"]
        else:
            last_feedback = ["포즈 미검출"]

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=str, required=True, help="레퍼런스 임베딩 .npy 경로")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--search_radius", type=int, default=5)
    ap.add_argument("--ema_alpha", type=float, default=0.5)
    ap.add_argument("--no_feedback", action="store_true")
    ap.add_argument("--ref_video", type=str, default=None, help="레퍼런스 영상 파일 경로(동시 표시 및 동기화)")
    ap.add_argument("--ref_stride", type=int, default=1, help="extract.py에서 사용한 stride 값 (동기화용)")
    ap.add_argument("--loop_ref", action="store_true", help="레퍼런스 재생을 반복(loop)")
    ap.add_argument("--rtf", type=float, default=1.0, help="재생 속도 배수 (1.0=원본)")
    ap.add_argument("--w_pose", type=float, default=0.7, help="정적 포즈 임베딩 유사도 가중치")
    ap.add_argument("--w_motion", type=float, default=0.3, help="모션(프레임 간 변화) 유사도 가중치")
    ap.add_argument("--time_penalty", type=float, default=0.2, help="윈도우 검색 시 시간 이탈 패널티 가중치(0~1)")
    ap.add_argument("--late_grace", type=int, default=5, help="늦게 따라할 때 무패널티 허용 프레임 수 (search_radius 기준)")
    ap.add_argument("--late_penalty", type=float, default=0.05, help="늦게 따라할 때(레퍼런스보다 뒤) 시간 이탈 패널티 가중치")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0,1,2], help="MediaPipe Pose 모델 복잡도")
    ap.add_argument("--warmup_sec", type=float, default=5.0, help="시작 후 N초 뒤에 레퍼런스 재생/유사도 비교 시작")
    ap.add_argument("--countdown_color", type=str, default="0,215,255", help="카운트다운 글자 B,G,R (예: 0,215,255)")
    ap.add_argument("--countdown_beep",default=True,  action="store_true", help="카운트다운 삐- 소리 재생")
    ap.add_argument("--play_ref_audio", default= True, action="store_true", help="레퍼런스 영상의 오디오 재생(ffplay/afplay 필요)")
    ap.add_argument("--rest_json", type=str, default="data/rest_exc.json", help="영상별 쉬는 구간 JSON 파일 경로 (예: rest_intervals.json)")
    args = ap.parse_args()
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
