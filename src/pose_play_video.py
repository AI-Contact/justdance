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
from pipeline import compare_videos, live_play

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=str, required=True, help="레퍼런스 임베딩 .npy 경로")
    ap.add_argument("--ref_lm", type=str, required=True, help="레퍼런스 임베딩 .npy 경로")
    ap.add_argument("--ref_video", type=str, default=None, help="레퍼런스 영상 파일 경로(동시 표시 및 동기화)")
    ap.add_argument("--user_video", type=str, default=None, help="웹캠 대신 비교할 사용자 로컬 동영상 경로")
    args = ap.parse_args()
    if args.user_video:
        compare_videos(
            ref_path=args.ref,
            ref_lm_path = args.ref_lm,
            user_video=args.user_video,
            ref_video=args.ref_video,
        )
    else:
        live_play(
            args.ref,
            ref_lm_path=args.ref_lm,
            ref_video=args.ref_video,
        )
