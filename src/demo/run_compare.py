"""run_compare.py
사용자 업로드된 4개 파일을 받아 compare_videos를 최소 인자로 실행.
OpenCV 창에서 결과를 바로 보여주고 종료되면 프로세스도 끝남.
사용 예:
  python3 src/demo/run_compare.py \
    --ref /path/to/ref.npy \
    --ref-lm /path/to/ref_lm.npy \
    --user-video /path/to/user.mp4 \
    --ref-video /path/to/ref_video.mp4
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

CUR = Path(__file__).resolve()
SRC_DIR = CUR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    os.chdir(SRC_DIR)
except Exception:
    pass

from pipeline import compare_videos  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True)
    p.add_argument('--ref-lm', required=True)
    p.add_argument('--user-video', required=True)
    p.add_argument('--ref-video', required=True)
    return p.parse_args()


def main():
    a = parse_args()
    for path in [a.ref, a.ref_lm, a.user_video, a.ref_video]:
        if not os.path.exists(path):
            print(f'[run_compare error] 파일을 찾을 수 없습니다: {path}')
            sys.exit(1)
    try:
        compare_videos(
            ref_path=a.ref,
            ref_lm_path=a.ref_lm,
            user_video=a.user_video,
            ref_video=a.ref_video,
        )
    except Exception as e:
        print(f'[run_compare error] {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()

