"""
app.py

웹에서 웹캠과 레퍼런스 영상을 실시간으로 비교하여 보여주는 Flask 앱.
기존 pipeline 함수(live_play)의 기본 설정을 유지하면서 브라우저에서 시각화.

사용법:
  python3 src/demo/app.py
  브라우저: http://localhost:5001
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 경로 설정
CUR_DIR = Path(__file__).resolve().parent
SRC_DIR = CUR_DIR.parent
ROOT_DIR = SRC_DIR.parent

# sys.path에 SRC_DIR 추가
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 작업 디렉토리를 ROOT_DIR로 변경 (data/ 등 접근을 위해)
try:
    os.chdir(ROOT_DIR)
except Exception as e:
    print(f"Warning: Could not change directory to {ROOT_DIR}: {e}")
    # SRC_DIR로 fallback
    try:
        os.chdir(SRC_DIR)
    except Exception:
        pass

# Pipeline 임포트 (최소 인자 사용)
from pipeline import (
    PoseExtractor, OnlineMatcher, read_frame_at,
    draw_colored_skeleton, normalize_landmarks, pose_embedding, angle_feedback,
    BONES, ANGLE_TRIPLES,
)
from pipeline.similarity import (
    build_static_feature_weights, weighted_cosine, exp_moving_avg, load_weights_for_video
)
from pipeline import load_rest_intervals_json, in_intervals

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('WEB_LIVE_DEMO_SECRET', 'web-live-demo-secret')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

# 업로드 디렉터리
TEMP_DIR = Path(os.getenv('TMPDIR', '/tmp'))
UPLOAD_DIR = TEMP_DIR / 'web_live_demo_uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
ALLOWED_NPY = {'.npy'}

# ============================================================
# 설정 섹션 (여기서 기본값을 수정하세요)
# ============================================================
CONFIG = {
    # 웹캠 설정
    'CAMERA_ID': 0,  # 0: 기본 웹캠, 1: 외장 웹캠

    # 레퍼런스 영상 재생 설정
    'REF_FPS': 30.0,  # 레퍼런스 영상 기본 FPS (실제 영상 FPS로 자동 조정됨)
    'REF_STRIDE': 2,  # 레퍼런스 프레임 stride (1: 모든 프레임, 2: 1프레임 건너뛰기)

    # 점수 계산 설정
    'SCORE_EMA_ALPHA': 0.95,  # EMA(지수이동평균) 알파값 (0.0~1.0, 높을수록 최근 값에 민감)

    # 피드백 설정
    'ANGLE_TOLERANCE_DEG': 10.0,  # 각도 피드백 허용 오차 (도 단위)

    # 포즈 추출 설정
    'POSE_MODEL_COMPLEXITY': 1,  # MediaPipe 모델 복잡도 (0: Lite, 1: Full, 2: Heavy)
    'STATIC_IMAGE_MODE': False,  # 정적 이미지 모드 (False: 비디오 모드)
    'MIN_DETECTION_CONFIDENCE': 0.15,  # 포즈 감지 최소 확신도 (0.3으로 낮춰 옆모습 대응)
    'MIN_TRACKING_CONFIDENCE': 0.15,   # 포즈 추적 최소 확신도 (0.3으로 낮춰 옆모습 대응)

    # 매칭 설정
    'SEARCH_RADIUS': 0,  # 온라인 매칭 검색 반경 (0: hint_idx만 사용)

    # REST 구간 설정
    'REST_JSON_PATH': 'src/data/rest_exc.json',  # REST 구간 JSON 파일 경로

    # 프레임 레이트 설정
    'WEBCAM_FPS': 30,  # 웹캠 스트리밍 FPS

    # JPEG 인코딩 품질
    'JPEG_QUALITY': 75,  # JPEG 압축 품질 (1~100, 높을수록 고품질)

    # 점수 등급 기준 (정규화된 점수 기준)
    'GRADE_PERFECT_THRESHOLD': 65.0,  # PERFECT 등급 최소 점수 (정규화 후)
    'GRADE_GOOD_THRESHOLD': 60.0,     # GOOD 등급 최소 점수 (정규화 후)

    # Warmup 설정
    'WARMUP_SEC': 5.0,  # 운동 시작 전 준비 시간 (초)
    'COUNTDOWN_BEEP': True,  # 카운트다운 비프음 활성화
}
# ============================================================

# 세션 상태 저장소 (웹캠 전용)
active_session = {
    'is_running': False,
    'ref_path': None,
    'ref_lm_path': None,
    'ref_video_path': None,
    'ref': None,
    'ref_lm': None,
    'ref_cap': None,
    'webcam_cap': None,
    'ref_fps': CONFIG['REF_FPS'],
    'ref_stride': CONFIG['REF_STRIDE'],
    'step_sec': 0.0,
    'rest_intervals': [],
    'region_w': None,
    'angle_w': None,
    'pe': None,
    'matcher': None,
    'prev_live_emb': None,
    'prev_ref_emb': None,
    'score_ema': 0.5,  # 초기값을 중간 점수로 설정 (0.5 = 50점 원본, 정규화 후 0점)
    'feedback': [],
    'ref_frame_cur': 0,
    'start_t': 0.0,
    'warmup_start_t': 0.0,  # warmup 시작 시간
    'warmup_done': False,    # warmup 완료 여부
    'last_beep_sec': None,   # 마지막 비프음 재생 초
    'grade_counts': {'PERFECT': 0, 'GOOD': 0, 'BAD': 0},  # 등급 카운트
    'graded_total': 0,       # 총 등급 매긴 횟수
    'last_grade_time': 0.0,  # 마지막으로 등급을 매긴 시간
    'current_grade': None,   # 현재 등급
    'grade_timestamp': 0.0,  # 등급 표시 시작 시간
    'grade_history': [],     # 3초간 등급 히스토리 [(timestamp, grade), ...]
    'final_rank': None,      # 최종 랭크 (S-F)
    'session_ended': False,  # 세션 종료 여부
    'end_time': 0.0,         # 세션 종료 시간
    'is_rest': False,        # 현재 REST 구간 여부
    'is_last_rest': False,   # 마지막 REST 구간 여부
    'frame_buffer': None,  # 최신 합성 프레임
    'lock': threading.Lock(),
}

def _ext_ok(name: str, kind: str) -> bool:
    ext = Path(name).suffix.lower()
    if kind == 'video':
        return ext in ALLOWED_VIDEO
    if kind == 'npy':
        return ext in ALLOWED_NPY
    return False

def _save(fs, d: Path) -> Path:
    p = d / secure_filename(fs.filename)
    fs.save(str(p))
    return p

def _fix_rotation(frame):
    """프레임 회전 자동 보정 (세로 영상을 가로로 변환)"""
    if frame is None:
        return frame

    h, w = frame.shape[:2]

    # 세로 영상인 경우 (높이 > 너비) 90도 회전
    if h > w:
        # 시계 반대방향 90도 회전 (ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame

def _combine(left, right):
    """좌우 프레임 합성"""
    if right is None:
        return left

    # 레퍼런스 프레임 회전 보정
    right = _fix_rotation(right)

    h, w = left.shape[:2]
    rh, rw = right.shape[:2]
    if rh != h:
        scale = h / float(rh)
        right = cv2.resize(right, (int(rw * scale), h), interpolation=cv2.INTER_LINEAR)
    return np.hstack([left, right])

def _normalize_score(score_ema):
    """점수 정규화 (등급 판정과 동일한 방식)"""
    # 50~95 -> 0~100 정규화
    score_50_95 = score_ema * 100.0
    normalized = ((score_50_95 - 50.0) / 45.0) * 100.0
    return float(np.clip(normalized, 0.0, 100.0))

def init_session(ref_path: str, ref_lm_path: str, ref_video_path: str):
    """세션 초기화 (웹캠 전용)"""
    with active_session['lock']:
        # 기존 리소스 정리
        if active_session['ref_cap']:
            active_session['ref_cap'].release()
        if active_session['webcam_cap']:
            active_session['webcam_cap'].release()

        # 레퍼런스 로드
        ref = np.load(ref_path, allow_pickle=True)
        ref_lm = np.load(ref_lm_path, allow_pickle=True)
        ref_cap = cv2.VideoCapture(ref_video_path)

        # 레퍼런스 영상 FPS 추출 (실제 영상 FPS 우선, 없으면 CONFIG 기본값)
        ref_fps = CONFIG['REF_FPS']
        if ref_cap.isOpened():
            fps_val = ref_cap.get(cv2.CAP_PROP_FPS)
            if fps_val and fps_val > 1e-3:
                ref_fps = float(fps_val)

        ref_stride = CONFIG['REF_STRIDE']
        rtf = 1.0
        step_sec = ref_stride / (ref_fps * max(1e-6, rtf))

        # REST 구간 / 가중치
        try:
            rest_intervals = load_rest_intervals_json(
                CONFIG['REST_JSON_PATH'],
                os.path.basename(ref_video_path),
                ref_fps,
                ref_stride
            )
        except Exception:
            rest_intervals = []
        try:
            # weights.json 경로를 명시적으로 지정 (작업 디렉토리 변경으로 인한 문제 방지)
            weights_json_path = str(SRC_DIR / 'data' / 'weights.json')
            region_w, angle_w = load_weights_for_video(ref_video_path, weights_json_path)

        except Exception as e:
            print(f"[Warning] Failed to load weights for {os.path.basename(ref_video_path)}: {e}")
            region_w, angle_w = None, None

        # 웹캠 열기 (CONFIG에서 카메라 ID 사용)
        webcam_cap = cv2.VideoCapture(CONFIG['CAMERA_ID'])
        if not webcam_cap.isOpened():
            raise RuntimeError(f'Failed to open webcam (CAMERA_ID={CONFIG["CAMERA_ID"]})')

        # 세션 업데이트
        active_session.update({
            'is_running': True,
            'ref_path': ref_path,
            'ref_lm_path': ref_lm_path,
            'ref_video_path': ref_video_path,
            'ref': ref,
            'ref_lm': ref_lm,
            'ref_cap': ref_cap,
            'webcam_cap': webcam_cap,
            'ref_fps': ref_fps,
            'ref_stride': ref_stride,
            'step_sec': step_sec,
            'rest_intervals': rest_intervals,
            'region_w': region_w,
            'angle_w': angle_w,
            'pe': PoseExtractor(
                static_image_mode=CONFIG['STATIC_IMAGE_MODE'],
                model_complexity=CONFIG['POSE_MODEL_COMPLEXITY'],
                min_detection_confidence=CONFIG['MIN_DETECTION_CONFIDENCE'],
                min_tracking_confidence=CONFIG['MIN_TRACKING_CONFIDENCE']
            ),
            'matcher': OnlineMatcher(ref),
            'prev_live_emb': None,
            'prev_ref_emb': None,
            'score_ema': 0.5,  # 초기값을 중간 점수로 설정 (0.5 = 50점 원본, 정규화 후 0점)
            'feedback': [],
            'ref_frame_cur': 0,
            'start_t': time.perf_counter(),
            'warmup_start_t': time.perf_counter(),  # warmup 시작 시간
            'warmup_done': False,                    # warmup 상태
            'last_beep_sec': None,                   # 비프음 상태 초기화
            'grade_counts': {'PERFECT': 0, 'GOOD': 0, 'BAD': 0},
            'graded_total': 0,
            'last_grade_time': 0.0,
            'current_grade': None,
            'grade_timestamp': 0.0,
            'grade_history': [],
            'final_rank': None,
            'session_ended': False,
            'end_time': 0.0,
            'is_rest': False,
            'is_last_rest': False,
            'frame_buffer': None,
        })

def generate_frames():
    """프레임 생성 (MJPEG 스트리밍) - 웹캠 전용"""
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        with active_session['lock']:
            if not active_session['is_running']:
                break

            webcam_cap = active_session['webcam_cap']
            ref_cap = active_session['ref_cap']
            pe = active_session['pe']
            matcher = active_session['matcher']
            ref = active_session['ref']
            ref_lm = active_session['ref_lm']
            ref_stride = active_session['ref_stride']
            step_sec = active_session['step_sec']
            ref_fps = active_session['ref_fps']
            rest_intervals = active_session['rest_intervals']
            region_w = active_session['region_w']
            angle_w = active_session['angle_w']

            if not webcam_cap or not ref_cap:
                break

            # 웹캠 프레임 읽기
            ret_webcam, webcam_frame = webcam_cap.read()
            if not ret_webcam:
                # 웹캠 재시도
                continue

        # Warmup 체크
        warmup_elapsed = time.perf_counter() - active_session['warmup_start_t']
        is_warmup = warmup_elapsed < CONFIG['WARMUP_SEC']
        warmup_remaining = max(0, CONFIG['WARMUP_SEC'] - warmup_elapsed)

        # Warmup 완료 시 start_t 재설정 및 레퍼런스 영상 시작 (한 번만)
        if not is_warmup and not active_session['warmup_done']:
            with active_session['lock']:
                active_session['warmup_done'] = True
                active_session['start_t'] = time.perf_counter()
                # 등급 평가 시작 시점도 초기화 (Warmup 후부터 3초 카운트 시작)
                active_session['last_grade_time'] = 0.0
                active_session['grade_history'] = []
                # 레퍼런스 영상을 처음으로 되돌림 (Warmup 동안 진행되지 않도록)
                if ref_cap and ref_cap.isOpened():
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    active_session['ref_frame_cur'] = 0

        # Warmup 중에는 카운트다운만 표시
        if is_warmup:
            display_left = webcam_frame.copy()

            # 레퍼런스 첫 프레임 표시
            with active_session['lock']:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_ref, ref_frame = ref_cap.read()
                active_session['ref_frame_cur'] = 0

            display_right = ref_frame if ok_ref else None
            combined = _combine(display_left, display_right)

            # 카운트다운 표시
            countdown_text = f"Ready... {int(warmup_remaining) + 1}"
            cv2.putText(combined, countdown_text, (20, 60), font, 1.5, (0, 215, 255), 3, cv2.LINE_AA)
            cv2.putText(combined, "WARMUP", (20, 120), font, 1.0, (0, 215, 255), 2, cv2.LINE_AA)

            if display_right is not None:
                cv2.putText(combined, "REFERENCE (Start Soon)",
                           (display_left.shape[1] + 20, 60), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # JPEG 인코딩
            ret, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG['JPEG_QUALITY']])
            if ret:
                with active_session['lock']:
                    active_session['frame_buffer'] = combined
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(1 / CONFIG['WEBCAM_FPS'])
            continue

        # 동기 인덱스 계산 (warmup 완료 후부터)
        elapsed = time.perf_counter() - active_session['start_t']
        hint_idx = int(elapsed / step_sec)
        hint_idx = min(hint_idx, ref.shape[0] - 1)
        if ref_lm is not None:
            hint_idx = min(hint_idx, len(ref_lm) - 1)

        # ref 영상 종료 체크 (마지막 10프레임 이내)
        is_near_end = hint_idx >= ref.shape[0] - 10

        if is_near_end and not active_session.get('session_ended', False):
            with active_session['lock']:
                # 최종 랭크 계산 (한 번만 실행)
                graded_total = active_session['graded_total']
                if graded_total > 0 and active_session.get('final_rank') is None:
                    p_cnt = active_session['grade_counts']['PERFECT']
                    g_cnt = active_session['grade_counts']['GOOD']
                    b_cnt = active_session['grade_counts']['BAD']
                    p_ratio = p_cnt / graded_total
                    pg_ratio = (p_cnt + g_cnt) / graded_total

                    # 랭크 규칙 (live_play와 동일)
                    if p_ratio >= 0.70:
                        final_rank = "S"
                    elif p_ratio >= 0.50 or pg_ratio >= 0.85:
                        final_rank = "A"
                    elif pg_ratio >= 0.70:
                        final_rank = "B"
                    elif pg_ratio >= 0.50:
                        final_rank = "C"
                    else:
                        final_rank = "F"

                    active_session['final_rank'] = final_rank
                    active_session['session_ended'] = True
                    active_session['end_time'] = time.perf_counter()

                    print("\n===== WEB LIVE PLAY SUMMARY =====")
                    print(f"Graded: {graded_total}")
                    print(f"PERFECT: {p_cnt} ({p_ratio*100:.1f}%)")
                    print(f"GOOD   : {g_cnt} ({(g_cnt/graded_total)*100:.1f}%)")
                    print(f"BAD    : {b_cnt} ({(b_cnt/graded_total)*100:.1f}%)")
                    print(f"FINAL RANK: {final_rank}")

        # session_ended 후 바로 세션 종료
        if active_session.get('session_ended', False):
            with active_session['lock']:
                active_session['is_running'] = False
            break


        display_left = webcam_frame.copy()

        # 포즈 추론 (항상 수행하여 스켈레톤 그리기)
        arr = pe.infer(webcam_frame)
        if arr is not None:
            overlay = display_left.copy()
            draw_colored_skeleton(overlay, arr)
            display_left = cv2.addWeighted(overlay, 0.5, display_left, 0.5, 0)

        # REST 구간
        if in_intervals(hint_idx, rest_intervals):
            # 마지막 REST 구간인지 확인
            is_last_rest = False
            if rest_intervals:
                # rest_intervals를 시간 순서로 정렬하여 마지막 구간 찾기
                sorted_intervals = sorted(rest_intervals, key=lambda x: x[0])
                last_rest_start, last_rest_end = sorted_intervals[-1]

                # 현재 hint_idx가 마지막 REST 구간 내에 있는지 확인
                if last_rest_start <= hint_idx <= last_rest_end:
                    is_last_rest = True
                    #print(f"[DEBUG] Last REST detected! hint_idx={hint_idx}, last_rest=({last_rest_start}, {last_rest_end})")
                #else:
                    #print(f"[DEBUG] Regular REST. hint_idx={hint_idx}, current_rest_intervals={rest_intervals}")

            with active_session['lock']:
                active_session['is_rest'] = True
                active_session['is_last_rest'] = is_last_rest
                ok_ref, ref_frame, active_session['ref_frame_cur'] = read_frame_at(
                    ref_cap, hint_idx * ref_stride, active_session['ref_frame_cur']
                )
            display_right = ref_frame if ok_ref else None
            combined = _combine(display_left, display_right)
            # REST 구간에서는 점수 계산 및 등급 평가 하지 않음
        else:
            with active_session['lock']:
                active_session['is_rest'] = False
                active_session['is_last_rest'] = False

            # 레퍼런스 프레임 먼저 가져오기
            with active_session['lock']:
                ok_ref, ref_frame, active_session['ref_frame_cur'] = read_frame_at(
                    ref_cap, hint_idx * ref_stride, active_session['ref_frame_cur']
                )
            display_right = ref_frame if ok_ref else None
            combined = _combine(display_left, display_right)

            # 포즈 분석 및 점수 계산 (포즈가 감지되었을 때만)
            if arr is not None:
                # 디버깅: visibility 확인
                avg_vis = np.mean(arr[:, 3])
                critical_joints = [11, 12, 23, 24, 13, 14, 25, 26]
                critical_vis = np.mean([arr[i, 3] for i in critical_joints if i < len(arr)])

                # 매우 낮은 visibility 체크 (옆모습 디버깅용)
                if critical_vis < 0.25:
                    print(f"[DEBUG] Low visibility detected - avg: {avg_vis:.2f}, critical: {critical_vis:.2f}")

                lm = normalize_landmarks(arr.copy())
                emb = pose_embedding(lm)

                # 임베딩에 유효한 값이 있는지 확인
                valid_ratio = np.sum(np.abs(emb) > 1e-6) / len(emb)
                if valid_ratio < 0.15:  # 0.3 → 0.15로 낮춤 (옆모습 대응)
                    print(f"[DEBUG] Very low valid embedding ratio: {valid_ratio:.2f}, skipping score calculation")
                    # 유효한 임베딩이 너무 적으면 점수 계산 스킵
                    combined = _combine(display_left, display_right)
                else:
                    # 옆모습 디버깅: 유효 임베딩 비율 출력
                    if valid_ratio < 0.3:
                        print(f"[DEBUG] Low but acceptable embedding ratio: {valid_ratio:.2f}, proceeding with score calculation")
                    _, ref_idx = matcher.step_with_hint(emb, hint_idx=hint_idx, search_radius=CONFIG['SEARCH_RADIUS'])

                    # 가중치
                    try:
                        if ref_lm is not None and ref_idx < len(ref_lm):
                            w_feat, _, _ = build_static_feature_weights(
                                BONES, ANGLE_TRIPLES, lm_for_vis=ref_lm[ref_idx],
                                region_w=region_w, angle_w=angle_w
                            )
                        else:
                            w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)[0]
                    except Exception:
                        w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)[0]

                    # 점수 계산
                    #print(f"[DEBUG] weights: {w_feat}")
                    static_sim = weighted_cosine(matcher.ref[ref_idx], emb, w_feat)
                    ref_emb = matcher.ref[ref_idx]

                    with active_session['lock']:
                        if active_session['prev_live_emb'] is not None and active_session['prev_ref_emb'] is not None:
                            d_live = emb - active_session['prev_live_emb']
                            d_ref = ref_emb - active_session['prev_ref_emb']
                            motion_sim = weighted_cosine(d_live, d_ref, w_feat)
                        else:
                            motion_sim = 0.0

                        blended = static_sim
                        score = ((blended + 1.0) * 0.5)
                        active_session['score_ema'] = exp_moving_avg(
                            active_session['score_ema'],
                            score,
                            alpha=CONFIG['SCORE_EMA_ALPHA']
                        )
                        active_session['prev_live_emb'] = emb
                        active_session['prev_ref_emb'] = ref_emb

                        # 피드백
                        try:
                            if ref_lm is not None and ref_idx < len(ref_lm):
                                msgs = angle_feedback(
                                    lm,
                                    ref_lm[ref_idx],
                                    angle_tol_deg=CONFIG['ANGLE_TOLERANCE_DEG']
                                )
                                active_session['feedback'] = msgs if msgs else ['Good!']
                        except Exception:
                            pass

                    # 점수/등급 계산 (포즈가 감지되었을 때만 히스토리에 추가)
                    with active_session['lock']:
                        # 현재 점수 정규화 (50~95 -> 0~100)
                        score_50_95 = active_session['score_ema'] * 100.0
                        normalized_score = ((score_50_95 - 50.0) / 45.0) * 100.0
                        normalized_score = float(np.clip(normalized_score, 0.0, 100.0))

                        current_time = time.time()

                        # 매 프레임마다 정규화된 점수를 히스토리에 추가
                        active_session['grade_history'].append((current_time, normalized_score))

                        # 3초보다 오래된 히스토리 제거
                        grade_interval = 3.0
                        cutoff_time = current_time - grade_interval
                        active_session['grade_history'] = [
                            (t, score) for t, score in active_session['grade_history'] if t > cutoff_time
                        ]

                        # 3초마다 한 번씩만 등급 평가
                        if elapsed - active_session['last_grade_time'] >= grade_interval:
                            # 3초 동안의 점수 히스토리에서 평균 점수 계산
                            if active_session['grade_history']:
                                avg_score = np.mean([score for t, score in active_session['grade_history']])

                                # 평균 점수로 등급 판정
                                if avg_score >= 70.0:
                                    final_grade = 'PERFECT'
                                elif avg_score >= 55.0:
                                    final_grade = 'GOOD'
                                else:
                                    final_grade = 'BAD'

                                active_session['grade_counts'][final_grade] += 1
                                active_session['graded_total'] += 1
                                active_session['last_grade_time'] = elapsed
                                active_session['current_grade'] = final_grade
                                active_session['grade_timestamp'] = current_time
                                # 히스토리 클리어 (새로운 3초 시작)
                                active_session['grade_history'] = []
            else:
                # 포즈가 감지되지 않은 경우
                print("[DEBUG] No pose detected (arr is None)")
                combined = _combine(display_left, display_right)


        # JPEG 인코딩 (CONFIG 품질 사용)
        ret, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG['JPEG_QUALITY']])
        if not ret:
            continue

        with active_session['lock']:
            active_session['frame_buffer'] = combined

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # 프레임 레이트 제어 (CONFIG에서 웹캠 FPS 사용)
        time.sleep(1 / CONFIG['WEBCAM_FPS'])

@app.route('/')
def index():
    return render_template('web_live_index.html')

@app.route('/api/start', methods=['POST'])
def start_session():
    """웹캠 세션 시작"""
    data = request.get_json()

    # 파일 업로드는 별도 엔드포인트에서 처리했다고 가정
    ref_path = data.get('ref_path')
    ref_lm_path = data.get('ref_lm_path')
    ref_video_path = data.get('ref_video_path')

    if not all([ref_path, ref_lm_path, ref_video_path]):
        return jsonify({'success': False, 'message': '필수 파일 경로가 누락되었습니다.'}), 400

    try:
        init_session(ref_path, ref_lm_path, ref_video_path)
        return jsonify({'success': True, 'message': '웹캠 세션이 시작되었습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'세션 시작 실패: {e}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """파일 업로드 (레퍼런스 파일만)"""
    ref_emb = request.files.get('ref_embedding')
    ref_lm = request.files.get('ref_landmarks')
    ref_vid = request.files.get('ref_video')

    if not all([ref_emb, ref_lm, ref_vid]):
        return jsonify({'success': False, 'message': '필수 파일이 누락되었습니다.'}), 400

    if not _ext_ok(ref_emb.filename, 'npy') or not _ext_ok(ref_lm.filename, 'npy'):
        return jsonify({'success': False, 'message': '.npy 파일만 가능합니다.'}), 400

    if not _ext_ok(ref_vid.filename, 'video'):
        return jsonify({'success': False, 'message': '레퍼런스 영상 형식이 잘못되었습니다.'}), 400

    sess_id = uuid.uuid4().hex
    sess_dir = UPLOAD_DIR / sess_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    ref_path = _save(ref_emb, sess_dir)
    ref_lm_path = _save(ref_lm, sess_dir)
    ref_video_path = _save(ref_vid, sess_dir)

    return jsonify({
        'success': True,
        'ref_path': str(ref_path),
        'ref_lm_path': str(ref_lm_path),
        'ref_video_path': str(ref_video_path),
    })

@app.route('/api/stop', methods=['POST'])
def stop_session():
    """세션 중지"""
    with active_session['lock']:
        active_session['is_running'] = False
        if active_session['ref_cap']:
            active_session['ref_cap'].release()
        if active_session['webcam_cap']:
            active_session['webcam_cap'].release()
        active_session['ref_cap'] = None
        active_session['webcam_cap'] = None
    return jsonify({'success': True, 'message': '세션이 중지되었습니다.'})

@app.route('/api/status')
def get_status():
    """상태 조회"""
    with active_session['lock']:
        # Warmup 관련 정보
        warmup_elapsed = 0.0
        warmup_remaining = 0.0
        is_warmup = False
        beep_signal = None

        if active_session['is_running'] and not active_session['warmup_done']:
            warmup_elapsed = time.perf_counter() - active_session['warmup_start_t']
            is_warmup = warmup_elapsed < CONFIG['WARMUP_SEC']
            warmup_remaining = max(0, CONFIG['WARMUP_SEC'] - warmup_elapsed)

            # 비프음 신호 (카운트다운 3, 2, 1, 0)
            if CONFIG['COUNTDOWN_BEEP'] and is_warmup:
                current_sec = int(warmup_remaining) + 1
                if current_sec <= 3 and current_sec != active_session.get('last_beep_sec'):
                    active_session['last_beep_sec'] = current_sec
                    # 주파수: 3초=800Hz, 2초=900Hz, 1초=1000Hz
                    beep_signal = {
                        'frequency': 700 + current_sec * 100,
                        'duration': 150
                    }
            elif not is_warmup and active_session.get('last_beep_sec') != 0:
                # 시작 비프 (0초)
                active_session['last_beep_sec'] = 0
                beep_signal = {
                    'frequency': 1400,
                    'duration': 250
                }

        response = {
            'is_running': active_session['is_running'],
            'score': _normalize_score(active_session['score_ema']) if active_session['score_ema'] is not None else 0.0,
            'is_warmup': is_warmup,
            'warmup_remaining': warmup_remaining,
            'is_rest': active_session['is_rest'],
            'is_last_rest': active_session.get('is_last_rest', False),
            'grade_counts': active_session['grade_counts'],
            'graded_total': active_session['graded_total'],
            'current_grade': active_session['current_grade'],
            'grade_timestamp': active_session['grade_timestamp'],
            'final_rank': active_session['final_rank'],
            'session_ended': active_session['session_ended'],
        }

        if beep_signal:
            response['beep'] = beep_signal

        return jsonify(response)

@app.route('/video_feed')
def video_feed():
    """비디오 스트리밍"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/ref_video')
def get_ref_video():
    """레퍼런스 비디오 파일 제공 (오디오 재생용)"""
    with active_session['lock']:
        ref_video_path = active_session.get('ref_video_path')
        if not ref_video_path or not Path(ref_video_path).exists():
            return jsonify({'success': False, 'message': '레퍼런스 비디오가 없습니다.'}), 404

    from flask import send_file
    import mimetypes

    # 파일 확장자에 따라 MIME 타입 자동 설정
    video_path = Path(ref_video_path)
    mime_type = mimetypes.guess_type(str(video_path))[0] or 'video/mp4'

    return send_file(
        ref_video_path,
        mimetype=mime_type,
        as_attachment=False,
        conditional=True  # Range 요청 지원
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)), debug=True, threaded=True, use_reloader=False)

