# pose_extract.py
# -----------------
# Usage:
#   python3 extract.py --ref_video data/short_ref.mp4 --out data/skeleton/short_ref.npy --stride 2 --max_frames 600
# Output:
#   - ref.npy (T, D) pose embeddings
#   - ref_lm.npy (T, 33, 4) normalized landmarks for per-joint angle feedback
# python3 extract.py --ref_video data/clap.mp4 --out data/skeleton/clap.npy --preview --throttle --overlay

import time
import argparse
import math
import cv2
import numpy as np

# Enable OpenCV runtime optimizations
cv2.setUseOptimized(True)
try:
    # On some builds this is a no-op; safe to call.
    cv2.setNumThreads(0)  # let OS schedule; prevents over-subscription
except Exception:
    pass

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("pip install mediapipe opencv-python numpy")

POSE_LANDMARKS = list(range(33))

BONES = [
    #왼어깨, 왼팔꿈치 / 왼팔꿈치, 왼손목
    (11, 13), (13, 15),
    #오른어깨, 오른팔꿈치 / 오른팔꿈치, 오른손목
    (12, 14), (14, 16),
    #왼엉덩, 왼무릎 / 왼무릎, 왼발목
    (23, 25), (25, 27),
    #오른엉덩, 오른무릎 / 오른무릎, 오른발목
    (24, 26), (26, 28),
    # 왼어깨, 오른어깨
    (11, 12),
    #왼엉덩, 오른 엉덩
    (23, 24),
    #왼어깨, 왼엉덩 / 오른어깨, 오른엉덩
    (11, 23), (12, 24)
]

ANGLE_TRIPLES = [
    # 왼엉덩, 왼무릎, 왼발목 -> 무릎 굽힘
    (23, 25, 27),
    # 오른엉덩, 오른무릎, 오른발목
    (24, 26, 28),
    # 왼팔꿈치, 왼어깨, 왼엉덩 -> 겨드랑이
    (13, 11, 23),
    # 오른팔꿈치, 오른어깨, 오른엉덩
    (14, 12, 24),
    # 왼어깨, 왼팔꿈치, 왼손목 -> 팔꿈치
    (11, 13, 15),
    #오른어깨, 오른팔꿈치, 오른손목
    (12, 14, 16),
]

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
    # Translate to pelvis center
    if lm[23,3] > 0.5 and lm[24,3] > 0.5:
        pelvis = (lm[23,:2] + lm[24,:2]) / 2.0
    elif lm[23,3] > 0.5:
        pelvis = lm[23,:2]
    elif lm[24,3] > 0.5:
        pelvis = lm[24,:2]
    else:
        pelvis = np.nanmean(lm[:, :2], axis=0)
    lm[:,:2] -= pelvis
    # Scale by average of shoulder and hip width
    scale_refs = []
    if lm[11,3]>0.5 and lm[12,3]>0.5:
        scale_refs.append(np.linalg.norm(lm[11,:2]-lm[12,:2]))
    if lm[23,3]>0.5 and lm[24,3]>0.5:
        scale_refs.append(np.linalg.norm(lm[23,:2]-lm[24,:2]))
    scale = np.nanmean(scale_refs) if len(scale_refs)>0 else np.nan
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.nanmax(np.linalg.norm(lm[:,:2], axis=1)) + 1e-6
    lm[:,:2] /= scale
    # Rotate shoulders to x-axis
    if lm[11,3]>0.5 and lm[12,3]>0.5:
        v = lm[12,:2]-lm[11,:2]
        ang = math.atan2(v[1], v[0])
        c, s = math.cos(-ang), math.sin(-ang)
        R = np.array([[c,-s],[s,c]], dtype=np.float32)
        lm[:,:2] = (R @ lm[:,:2].T).T
    return lm

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

def extract_reference_with_landmarks(video_path, stride=1, max_frames=None, out="ref.npy",
                                     preview=False, throttle=False, overlay=False, vis_th=0.5,
                                     rtf=1.0, viz_stride=1, model_complexity=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    # 원본 FPS 추출 (없으면 30으로 fallback)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    # stride가 2면, 처리 프레임 간 시간은 stride/fps 초
    # real-time factor (rtf): 1.0 = 원본 속도, 1.2 = 20% 빠르게
    target_frame_interval = stride / (fps * max(1e-6, rtf))  # seconds

    pe = PoseExtractor(static_image_mode=False, model_complexity=model_complexity)
    embs, lms, raw_lms = [], [], []
    idx = 0
    shown = 0

    # 스켈레톤 그리는 간단 함수 (원본 좌표계)
    def draw_skeleton(frame, lm_xyv, color=(0, 255, 0), thickness=1, radius=2):
        h, w = frame.shape[:2]
        # 점
        for i in range(33):
            x, y, v = lm_xyv[i, 0], lm_xyv[i, 1], lm_xyv[i, 3]
            if v >= vis_th and 0 <= x <= 1 and 0 <= y <= 1:
                cv2.circle(frame, (int(x * w), int(y * h)), radius, color, -1, cv2.LINE_AA)
        # 선
        for i, j in BONES:
            xi, yi, vi = lm_xyv[i, 0], lm_xyv[i, 1], lm_xyv[i, 3]
            xj, yj, vj = lm_xyv[j, 0], lm_xyv[j, 1], lm_xyv[j, 3]
            if vi >= vis_th and vj >= vis_th and 0 <= xi <= 1 and 0 <= yi <= 1 and 0 <= xj <= 1 and 0 <= yj <= 1:
                cv2.line(frame, (int(xi * w), int(yi * h)), (int(xj * w), int(yj * h)), color, thickness, cv2.LINE_AA)

    while True:
        start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        # stride 샘플링
        if idx % stride != 0:
            idx += 1
            # 프리뷰가 켜져 있고, 프레임을 스킵하더라도 원본 속도 유지하고 싶으면 여기서도 쓰로틀
            if throttle or preview:
                elapsed = time.perf_counter() - start
                wait_sec = max(0.0, target_frame_interval - elapsed)
                if preview:
                    key = cv2.waitKey(int(wait_sec * 1000)) & 0xFF
                    if key == 27 or key == ord('q'):
                        break
                else:
                    time.sleep(wait_sec)
            continue

        arr = pe.infer(frame)
        if arr is not None:
            raw_lms.append(arr.copy())
            nlm = normalize_landmarks(arr.copy())
            emb = pose_embedding(nlm)
            embs.append(emb)
            lms.append(nlm)

            # 미리보기: 원본 프레임에 스켈레톤 오버레이
            if preview and (shown % max(1, viz_stride) == 0):
                if overlay:
                    vis_frame = frame.copy()
                    draw_skeleton(vis_frame, arr)
                    cv2.imshow("extract preview", vis_frame)
                else:
                    cv2.imshow("extract preview", frame)

        idx += 1
        shown += 1
        if max_frames and len(embs) >= max_frames:
            break

        # 속도 제어: 처리에 걸린 시간만큼 빼고 남은 시간을 기다림
        if throttle or preview:
            elapsed = time.perf_counter() - start
            wait_sec = max(0.0, target_frame_interval - elapsed)
            if preview:
                key = cv2.waitKey(int(wait_sec * 1000)) & 0xFF
                if key == 27 or key == ord('q'):
                    break
            else:
                time.sleep(wait_sec)

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    if len(embs) == 0:
        raise RuntimeError("레퍼런스에서 포즈를 추출하지 못했습니다.")

    ref = np.stack(embs, axis=0)
    np.save(out, ref)
    np.save(out.replace('.npy', '_lm.npy'), np.stack(lms, axis=0))
    np.save(out.replace('.npy', '_raw.npy'), np.stack(raw_lms, axis=0))  # 추가 저장
    print(f"Saved: {out}, {out.replace('.npy', '_lm.npy')}, {out.replace('.npy', '_raw.npy')} (frames={len(ref)}, fps≈{fps:.2f}, stride={stride}, rtf={rtf}, viz_stride={viz_stride}, mc={model_complexity})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_video", type=str, required=True, help="레퍼런스 비디오 경로")
    ap.add_argument("--out", type=str, default="ref.npy", help="임베딩 저장 경로")
    ap.add_argument("--stride", type=int, default=2, help="프레임 샘플링 간격")
    ap.add_argument("--max_frames", type=int, default=0, help="최대 프레임(0=무제한)")
    ap.add_argument("--preview", action="store_true", help="창으로 프레임을 원본 속도에 맞춰 미리보기")
    ap.add_argument("--throttle", action="store_true", help="프리뷰 없이도 처리 속도를 원본 FPS/stride에 맞춰 제한")
    ap.add_argument("--overlay", action="store_true", help="미리보기 시 영상 위에 스켈레톤을 덧그림")
    ap.add_argument("--vis_th", type=float, default=0.5, help="스켈레톤 표시 visibility 임계값")
    ap.add_argument("--rtf", type=float, default=1.0, help="재생 속도 배수 (1.0=원본, 1.2=20% 빠르게)")
    ap.add_argument("--viz_stride", type=int, default=1, help="프리뷰/오버레이를 N프레임마다 1회만 갱신")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0,1,2], help="MediaPipe Pose 모델 복잡도(0이 가장 빠름)")
    args = ap.parse_args()
    mf = None if args.max_frames == 0 else args.max_frames
    extract_reference_with_landmarks(
        args.ref_video,
        stride=args.stride,
        max_frames=mf,
        out=args.out,
        preview=args.preview,
        throttle=args.throttle,
        overlay=args.overlay,
        vis_th=args.vis_th,
        rtf=args.rtf,
        viz_stride=args.viz_stride,
        model_complexity=args.model_complexity
    )

