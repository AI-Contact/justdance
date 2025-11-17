"""
쉬는 구간(REST intervals) 관련 함수들
"""

import os
import json
import sys


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

