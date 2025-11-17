"""
run_live.py: pipeline.live_play를 별도 프로세스에서 실행하기 위한 간단한 런너
사용 예:
  python3 src/demo/run_live.py \
    --ref /tmp/ref.npy \
    --ref-lm /tmp/ref_lm.npy \
    --ref-video /tmp/ref.mp4 \
    --search-radius 3 --ema 0.2 --ref-stride 1 --model 1 --w-pose 0.7 --w-motion 0.3

주의: CWD는 src로 설정되어 있어야 pipeline의 data/* 상대경로가 동작합니다.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# src 루트를 sys.path에 추가
CUR = Path(__file__).resolve()
SRC_DIR = CUR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# CWD를 src로 변경 (data/* 상대경로 호환)
try:
    os.chdir(SRC_DIR)
except Exception:
    pass

from pipeline import live_play  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--ref-lm", dest="ref_lm", required=True)
    p.add_argument("--ref-video", dest="ref_video", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    try:
        live_play(
            ref_path=a.ref,
            ref_lm_path=a.ref_lm,
            ref_video=a.ref_video,
        )
    except Exception as e:
        print(f"[run_live error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

