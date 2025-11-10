# viz_video.py
# Usage:
#   python3 viz_video.py --video data/short_ref.mp4 --raw_npy data/skeleton/short_ref_raw.npy --preview
#   # 미리보기만(저장없이) 하고 싶으면 --preview 만 주면 되고 --out 생략 가능

import argparse
import cv2
import numpy as np

# MediaPipe Pose 인덱스 기준 (총 33개)
BONES = [
    (11, 13), (13, 15),  # L-Shoulder->Elbow, Elbow->Wrist
    (12, 14), (14, 16),  # R-Shoulder->Elbow, Elbow->Wrist
    (23, 25), (25, 27),  # L-Hip->Knee, Knee->Ankle
    (24, 26), (26, 28),  # R-Hip->Knee, Knee->Ankle
    (11, 12),            # Shoulder-Shoulder
    (23, 24),            # Hip-Hip
    (11, 23), (12, 24)   # Shoulder->Hip (Left/Right)
]

def draw_skeleton(frame, lm_xyv, color=(0,255,0), radius=4, thickness=2, vis_th=0.5):
    """ lm_xyv: (33,4) with [x,y,z,visibility], x/y in [0,1] (영상 정규화 좌표) """
    h, w = frame.shape[:2]
    # 점
    for i in range(lm_xyv.shape[0]):
        x, y, v = lm_xyv[i,0], lm_xyv[i,1], lm_xyv[i,3]
        if v >= vis_th and 0 <= x <= 1 and 0 <= y <= 1:
            cx, cy = int(x*w), int(y*h)
            cv2.circle(frame, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)

    # 뼈대
    for i, j in BONES:
        xi, yi, vi = lm_xyv[i,0], lm_xyv[i,1], lm_xyv[i,3]
        xj, yj, vj = lm_xyv[j,0], lm_xyv[j,1], lm_xyv[j,3]
        if vi >= vis_th and vj >= vis_th:
            if 0 <= xi <= 1 and 0 <= yi <= 1 and 0 <= xj <= 1 and 0 <= yj <= 1:
                pi = (int(xi*w), int(yi*h))
                pj = (int(xj*w), int(yj*h))
                cv2.line(frame, pi, pj, color, thickness, lineType=cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str, help="원본 비디오 경로")
    ap.add_argument("--raw_npy", required=True, type=str, help="추출된 원본 랜드마크 (T,33,4)")
    ap.add_argument("--out", type=str, default="", help="저장 파일 경로 (mp4). 미지정 시 저장 안 함")
    ap.add_argument("--preview", action="store_true", help="윈도우 미리보기")
    ap.add_argument("--color", type=str, default="0,255,0", help="스켈레톤 색 (B,G,R)")
    args = ap.parse_args()

    color = tuple(map(int, args.color.split(",")))  # "0,255,0" -> (0,255,0)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    lms = np.load(args.raw_npy)  # (T,33,4) in [0..1] coords
    T = lms.shape[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (W,H))

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if t < T:
            draw_skeleton(frame, lms[t], color=color)
        # 저장 or 미리보기
        if writer is not None:
            writer.write(frame)
        if args.preview:
            cv2.imshow("overlay", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        t += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()