"""
web_wrap_demo.py

기존 pipeline 함수(compare_videos, live_play)를 '최소 인자'로 그대로 호출하면서
OpenCV 창 대신 브라우저에서 프레임을 스트리밍(MJPEG)으로 볼 수 있게 하는 별도 데모.

요구 조건 반영:
- 기존 파일 수정 없음. (새 파일 생성)
- compare_videos 호출 시: ref_path, ref_lm_path, user_video, ref_video 4개 인자만.
- live_play 호출 시: ref_path, ref_lm_path, ref_video 3개 인자만.
- 함수 내부 기본값( stride 등 ) 변경 없이 유지.
- out_video, 기타 파라미터 추가 전달 금지.

방법:
- 호출 전 cv2.imshow / cv2.waitKey / cv2.destroyAllWindows 를 monkeypatch 하여
  프레임을 메모리에 저장하고 브라우저에 MJPEG로 제공.
- compare_videos는 영상 길이만큼 진행 후 종료.
- live_play는 ESC/q 입력을 기다리므로, 브라우저에서 /stop 세션 호출 시 waitKey가 ESC 반환하여 종료.

제약:
- 여러 세션 동시 지원은 간단화를 위해 1개만 활성 (새 세션 시작 시 이전 중지)
- 성능/지연 최소화 위해 JPEG 품질 70, 프레임 가져오기 주기 짧음.

실행:
  python3 src/demo/web_wrap_demo.py
  브라우저: http://localhost:5004
"""
from __future__ import annotations

import os
import sys
import time
import threading
import uuid
import io
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any

from flask import (
    Flask, request, redirect, url_for, flash, render_template, Response, jsonify
)
from werkzeug.utils import secure_filename

# 경로 설정 & pipeline 상대 경로 호환
CUR_DIR = Path(__file__).parent
SRC_DIR = CUR_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
try:
    os.chdir(SRC_DIR)
except Exception:
    pass

# 최소 인자 호출용 함수 import (파라미터 재정의 없이)
from pipeline import compare_videos, live_play  # type: ignore

import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('POSE_WRAP_DEMO_SECRET', 'pose-wrap-demo-secret')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

TEMP_DIR = Path(os.getenv('TMPDIR', '/tmp'))
UPLOAD_DIR = TEMP_DIR / 'pose_wrap_demo_uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
ALLOWED_NPY = {'.npy'}

# 세션 상태 ----------------------------------------------------
ACTIVE_SESSION: Dict[str, Any] = {}
SESSION_LOCK = threading.Lock()
FRAME_BUFFER = deque(maxlen=2)  # 최근 프레임 2개만 유지
STOP_FLAG = threading.Event()   # live_play 중단용

# Monkeypatch 원본 보관
ORIG_IMSHOW = cv2.imshow
ORIG_WAITKEY = cv2.waitKey
ORIG_DESTROY = cv2.destroyAllWindows


def allowed(filename: str, kind: str) -> bool:
    ext = Path(filename).suffix.lower()
    if kind == 'video':
        return ext in ALLOWED_VIDEO
    if kind == 'npy':
        return ext in ALLOWED_NPY
    return False


def save_file(fs, target_dir: Path) -> Path:
    fname = secure_filename(fs.filename)
    p = target_dir / fname
    fs.save(str(p))
    return p

# Monkeypatch --------------------------------------------------

def patch_opencv(session_id: str):
    """cv2 함수를 프레임 캡처 버전으로 교체"""
    def patched_imshow(win: str, frame):
        # frame: BGR
        # 프레임 복사(원본 변형 방지)
        FRAME_BUFFER.append(frame.copy())
    def patched_waitKey(delay: int = 1):
        # stop 요청 시 ESC 반환
        if STOP_FLAG.is_set():
            return 27  # ESC
        time.sleep(max(0, delay / 1000.0))
        return 1  # 아무 키도 안 눌림
    def patched_destroyAllWindows():
        pass  # 별도 처리 불필요
    cv2.imshow = patched_imshow  # type: ignore
    cv2.waitKey = patched_waitKey  # type: ignore
    cv2.destroyAllWindows = patched_destroyAllWindows  # type: ignore


def restore_opencv():
    cv2.imshow = ORIG_IMSHOW  # type: ignore
    cv2.waitKey = ORIG_WAITKEY  # type: ignore
    cv2.destroyAllWindows = ORIG_DESTROY  # type: ignore

# 세션 실행 -----------------------------------------------------

def run_compare_session(sess_id: str, ref_path: str, ref_lm_path: str, user_video: str, ref_video: str):
    try:
        patch_opencv(sess_id)
        compare_videos(
            ref_path=ref_path,
            ref_lm_path=ref_lm_path,
            user_video=user_video,
            ref_video=ref_video,
        )
    except Exception as e:
        with SESSION_LOCK:
            ACTIVE_SESSION['error'] = f'compare_videos 오류: {e}'
    finally:
        restore_opencv()
        STOP_FLAG.clear()
        with SESSION_LOCK:
            ACTIVE_SESSION['running'] = False


def run_live_session(sess_id: str, ref_path: str, ref_lm_path: str, ref_video: str):
    try:
        patch_opencv(sess_id)
        live_play(
            ref_path=ref_path,
            ref_lm_path=ref_lm_path,
            ref_video=ref_video,
        )
    except Exception as e:
        with SESSION_LOCK:
            ACTIVE_SESSION['error'] = f'live_play 오류: {e}'
    finally:
        restore_opencv()
        STOP_FLAG.clear()
        with SESSION_LOCK:
            ACTIVE_SESSION['running'] = False

# Routes -------------------------------------------------------

@app.route('/')
def index():
    return render_template('web_wrap_index.html')

@app.route('/start_compare', methods=['POST'])
def start_compare():
    ref_emb = request.files.get('ref_embedding')
    ref_lm = request.files.get('ref_landmarks')
    ref_vid = request.files.get('ref_video')
    user_vid = request.files.get('user_video')
    if not all([ref_emb, ref_lm, ref_vid, user_vid]):
        flash('모든 파일(ref 임베딩, 랜드마크, 레퍼런스 영상, 사용자 영상)을 업로드하세요.', 'error')
        return redirect(url_for('index'))
    if not allowed(ref_emb.filename, 'npy') or not allowed(ref_lm.filename, 'npy'):
        flash('.npy 파일만 가능합니다.', 'error')
        return redirect(url_for('index'))
    if not allowed(ref_vid.filename, 'video') or not allowed(user_vid.filename, 'video'):
        flash('영상 형식이 잘못되었습니다.', 'error')
        return redirect(url_for('index'))

    sess_id = uuid.uuid4().hex
    sess_dir = UPLOAD_DIR / sess_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    ref_path = save_file(ref_emb, sess_dir)
    ref_lm_path = save_file(ref_lm, sess_dir)
    ref_video_path = save_file(ref_vid, sess_dir)
    user_video_path = save_file(user_vid, sess_dir)

    with SESSION_LOCK:
        ACTIVE_SESSION.clear()
        ACTIVE_SESSION.update({
            'id': sess_id,
            'running': True,
            'mode': 'compare',
            'error': None,
        })
    STOP_FLAG.clear()

    t = threading.Thread(target=run_compare_session, args=(sess_id, str(ref_path), str(ref_lm_path), str(user_video_path), str(ref_video_path)), daemon=True)
    t.start()
    flash('비교 세션 시작. 아래 스트림을 확인하세요.', 'success')
    return redirect(url_for('index'))

@app.route('/start_live', methods=['POST'])
def start_live():
    ref_emb = request.files.get('ref_embedding')
    ref_lm = request.files.get('ref_landmarks')
    ref_vid = request.files.get('ref_video')
    if not all([ref_emb, ref_lm, ref_vid]):
        flash('필수 파일(ref 임베딩, 랜드마크, 레퍼런스 영상)을 모두 업로드하세요.', 'error')
        return redirect(url_for('index'))
    if not allowed(ref_emb.filename, 'npy') or not allowed(ref_lm.filename, 'npy') or not allowed(ref_vid.filename, 'video'):
        flash('파일 형식 오류.', 'error')
        return redirect(url_for('index'))

    sess_id = uuid.uuid4().hex
    sess_dir = UPLOAD_DIR / sess_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    ref_path = save_file(ref_emb, sess_dir)
    ref_lm_path = save_file(ref_lm, sess_dir)
    ref_video_path = save_file(ref_vid, sess_dir)

    with SESSION_LOCK:
        ACTIVE_SESSION.clear()
        ACTIVE_SESSION.update({
            'id': sess_id,
            'running': True,
            'mode': 'live',
            'error': None,
        })
    STOP_FLAG.clear()

    t = threading.Thread(target=run_live_session, args=(sess_id, str(ref_path), str(ref_lm_path), str(ref_video_path)), daemon=True)
    t.start()
    flash('실시간 세션 시작. 아래 스트림을 확인하세요. 중지 버튼으로 종료 가능합니다.', 'success')
    return redirect(url_for('index'))

@app.route('/stop', methods=['POST'])
def stop_session():
    STOP_FLAG.set()
    flash('종료 요청을 전달했습니다. 몇 초 안에 세션이 종료됩니다.', 'info')
    return redirect(url_for('index'))

@app.route('/status')
def status():
    with SESSION_LOCK:
        return jsonify(ACTIVE_SESSION)

@app.route('/stream')
def stream():
    def gen():
        last_none_time = time.time()
        while True:
            with SESSION_LOCK:
                running = ACTIVE_SESSION.get('running', False)
            if not running and not FRAME_BUFFER:
                break
            if FRAME_BUFFER:
                frame = FRAME_BUFFER[-1]
                # JPEG 인코딩
                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    jpg_bytes = buf.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
                last_none_time = time.time()
            else:
                # 프레임 없으면 잠깐 대기
                time.sleep(0.05)
                # 타임아웃(10초) 후 종료
                if time.time() - last_none_time > 10:
                    break
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 템플릿 --------------------------------------------------------
# web_wrap_index.html 과 구분되도록 별도 이름
@app.route('/wrap_help')
def wrap_help():
    return '<pre>이 데모는 monkeypatch로 cv2.imshow를 후킹하여 브라우저 스트림을 제공합니다.</pre>'

# Jinja 템플릿 파일(web_wrap_index.html) 없으면 간단한 기본 페이지 제공
@app.route('/fallback')
def fallback():
    return '''<html><body><h3>웹 래핑 데모</h3><p>/ 로 이동해 주세요.</p></body></html>'''

# 실제 index 템플릿이 존재하는지 확인 후 없으면 동적 생성
@app.context_processor
def inject_flags():
    return {'has_active': bool(ACTIVE_SESSION.get('running'))}

# ---------------- 실행 ----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5004)), debug=True)

