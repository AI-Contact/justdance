"""
web_stream_demo.py

별도의 Flask + Socket.IO 기반 웹 데모.
요구사항:
  - 기존 파일(app.py 등)을 수정하지 않고 새 파일로 제공.
  - compare_videos / live_play 호출 시 최소 인자만(지정된 3~4개) 전달.
  - 웹에서 실시간(WebSocket)으로 브라우저 웹캠 프레임을 보내어 레퍼런스와 비교 → 결과 이미지를 다시 전송.
  - 업로드된 사용자 영상과 레퍼런스를 비교할 때는 compare_videos(ref_path, ref_lm_path, user_video, ref_video) 만 호출.
  - 실시간 로컬(OpenCV 창) 비교를 원할 경우 live_play(ref_path, ref_lm_path, ref_video) 만 subprocess로 실행.

주의:
  - compare_videos / live_play 내부 기본 파라미터값을 유지하기 위해 추가 인자(out_video 등)는 넘기지 않음.
  - compare_videos를 서버에서 호출하면 OpenCV 창이 서버 측에 뜨므로, 브라우저 결과 영상 표시 기능은 이 최소 인자 정책 하에서는 제공되지 않음.
  - 실시간 웹 스트리밍 비교는 pipeline 구성요소(PoseExtractor 등)를 직접 사용하며 live_play를 호출하지 않음(요구된 제한은 호출 시 인자에만 적용).

실행:
  python3 src/demo/web_stream_demo.py
  브라우저: http://localhost:5003
"""
from __future__ import annotations

import os
import sys
import uuid
import time
import base64
from pathlib import Path

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    flash,
    render_template,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_socketio import SocketIO, emit

# ---------------- 경로 설정 ----------------
CUR_DIR = Path(__file__).parent
SRC_DIR = CUR_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
try:
    os.chdir(SRC_DIR)  # pipeline 내부 상대경로(data/*) 호환
except Exception:
    pass

# 최소 인자 호출용: compare_videos / live_play
from pipeline import compare_videos  # type: ignore
from pipeline import live_play  # type: ignore

# 스트리밍용 개별 구성요소( live_play 동작을 웹으로 복제 )
from pipeline import (
    PoseExtractor, OnlineMatcher, read_frame_at,
    draw_colored_skeleton, normalize_landmarks, pose_embedding, angle_feedback,
    BONES, ANGLE_TRIPLES,
)
from pipeline.similarity import (
    build_static_feature_weights, weighted_cosine, exp_moving_avg, load_weights_for_video
)
from pipeline import (
    load_rest_intervals_json, in_intervals
)

import cv2
import numpy as np

# ---------------- Flask / Socket.IO 설정 ----------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('POSE_STREAM_DEMO_SECRET', 'pose-stream-demo')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# ---------------- 업로드 / 세션 ----------------
TEMP_DIR = Path(os.getenv('TMPDIR', '/tmp'))
UPLOAD_DIR = TEMP_DIR / 'pose_stream_demo_uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
ALLOWED_NPY = {'.npy'}

STREAM_SESSIONS: dict[str, dict] = {}

# ---------------- 유틸 ----------------

def _ok_ext(name: str, kind: str) -> bool:
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


@app.errorhandler(RequestEntityTooLarge)
def handle_large(e):  # noqa: D401
    flash('업로드 파일이 너무 큽니다 (최대 1GB).', 'error')
    return redirect(url_for('index'))

# ---------------- 라우트: 메인 페이지 ----------------
@app.route('/', methods=['GET'])
def index():
    return render_template('web_stream_index.html')

# ---------------- 라우트: 업로드 비교 시작 ----------------
@app.route('/start_upload_compare', methods=['POST'])
def start_upload_compare():
    ref_emb = request.files.get('ref_embedding')
    ref_lm = request.files.get('ref_landmarks')
    ref_vid = request.files.get('ref_video')
    user_vid = request.files.get('user_video')
    if not all([ref_emb, ref_lm, ref_vid, user_vid]):
        flash('레퍼런스 임베딩/랜드마크/영상 + 사용자 영상 모두 업로드해야 합니다.', 'error')
        return redirect(url_for('index'))
    if not _ok_ext(ref_emb.filename, 'npy') or not _ok_ext(ref_lm.filename, 'npy'):
        flash('임베딩/랜드마크는 .npy 파일이어야 합니다.', 'error')
        return redirect(url_for('index'))
    if not _ok_ext(ref_vid.filename, 'video') or not _ok_ext(user_vid.filename, 'video'):
        flash('영상 파일 형식이 올바르지 않습니다.', 'error')
        return redirect(url_for('index'))

    req_dir = UPLOAD_DIR / uuid.uuid4().hex
    req_dir.mkdir(parents=True, exist_ok=True)
    ref_path = _save(ref_emb, req_dir)
    ref_lm_path = _save(ref_lm, req_dir)
    ref_video_path = _save(ref_vid, req_dir)
    user_video_path = _save(user_vid, req_dir)

    # 최소 인자 compare_videos 호출(동기) → OpenCV 창 출력
    try:
        compare_videos(
            ref_path=str(ref_path),
            ref_lm_path=str(ref_lm_path),
            user_video=str(user_video_path),
            ref_video=str(ref_video_path),
        )
        flash('비교 완료. 로컬 창에서 확인했거나 종료되었습니다.', 'success')
    except Exception as exc:
        flash(f'compare_videos 실행 오류: {exc}', 'error')
    return redirect(url_for('index'))

# ---------------- 라우트: 로컬 live_play 실행 ----------------
@app.route('/start_local_live', methods=['POST'])
def start_local_live():
    ref_emb = request.files.get('ref_embedding')
    ref_lm = request.files.get('ref_landmarks')
    ref_vid = request.files.get('ref_video')
    if not all([ref_emb, ref_lm, ref_vid]):
        flash('레퍼런스 임베딩/랜드마크/영상 3개 모두 업로드하세요.', 'error')
        return redirect(url_for('index'))
    if not _ok_ext(ref_emb.filename, 'npy') or not _ok_ext(ref_lm.filename, 'npy') or not _ok_ext(ref_vid.filename, 'video'):
        flash('파일 형식이 올바르지 않습니다.', 'error')
        return redirect(url_for('index'))

    req_dir = UPLOAD_DIR / uuid.uuid4().hex
    req_dir.mkdir(parents=True, exist_ok=True)
    ref_path = _save(ref_emb, req_dir)
    ref_lm_path = _save(ref_lm, req_dir)
    ref_video_path = _save(ref_vid, req_dir)

    # 최소 인자만 사용하여 live_play를 별도 subprocess로 실행
    cmd = [
        sys.executable,
        str(CUR_DIR / 'run_live.py'),
        '--ref', str(ref_path),
        '--ref-lm', str(ref_lm_path),
        '--ref-video', str(ref_video_path),
    ]
    try:
        import subprocess as sp
        sp.Popen(cmd)
        flash('실시간 웹캠 비교가 로컬 창에서 시작되었습니다 (ESC/q로 종료).', 'success')
    except Exception as exc:
        flash(f'live_play 실행 실패: {exc}', 'error')
    return redirect(url_for('index'))

# ---------------- 라우트: 실시간 스트리밍 세션 초기화 ----------------
@app.route('/stream_init', methods=['POST'])
def stream_init():
    ref_emb = request.files.get('ref_embedding')
    ref_lm = request.files.get('ref_landmarks')
    ref_vid = request.files.get('ref_video')
    if not all([ref_emb, ref_lm, ref_vid]):
        return {'ok': False, 'error': 'missing files'}, 400
    if not _ok_ext(ref_emb.filename, 'npy') or not _ok_ext(ref_lm.filename, 'npy') or not _ok_ext(ref_vid.filename, 'video'):
        return {'ok': False, 'error': 'invalid extensions'}, 400

    sess_id = uuid.uuid4().hex
    sess_dir = UPLOAD_DIR / f'session_{sess_id}'
    sess_dir.mkdir(parents=True, exist_ok=True)
    ref_path = _save(ref_emb, sess_dir)
    ref_lm_path = _save(ref_lm, sess_dir)
    ref_video_path = _save(ref_vid, sess_dir)

    try:
        ref = np.load(str(ref_path))
        ref_lm_arr = np.load(str(ref_lm_path))
    except Exception as exc:
        return {'ok': False, 'error': f'npy load failed: {exc}'}, 400

    # 레퍼런스 영상 fps 추출
    ref_cap = cv2.VideoCapture(str(ref_video_path))
    ref_fps = 30.0
    if ref_cap.isOpened():
        fps_val = ref_cap.get(cv2.CAP_PROP_FPS)
        if fps_val and fps_val > 1e-3:
            ref_fps = float(fps_val)
    ref_stride = 2  # live_play/compare_videos 기본값과 동기화 (내부 기본값 확인 필요)
    step_sec = ref_stride / (ref_fps * 1.0)

    # REST / 가중치 로드 (존재하지 않아도 진행)
    try:
        rest_intv = load_rest_intervals_json('data/rest_exc.json', os.path.basename(str(ref_video_path)), ref_fps, ref_stride)
    except Exception:
        rest_intv = []
    try:
        region_w, angle_w = load_weights_for_video(str(ref_video_path))
    except Exception:
        region_w, angle_w = None, None

    session = {
        'dir': str(sess_dir),
        'ref': ref,
        'ref_lm': ref_lm_arr,
        'ref_cap': ref_cap,
        'ref_fps': ref_fps,
        'ref_stride': ref_stride,
        'step_sec': step_sec,
        'rest_intv': rest_intv,
        'region_w': region_w,
        'angle_w': angle_w,
        'pe': PoseExtractor(static_image_mode=False, model_complexity=1),
        'matcher': OnlineMatcher(ref),
        'prev_emb_live': None,
        'prev_emb_ref': None,
        'score_ema': 0.0,
        'score_window': [],
        'feedback': [],
        'ref_frame_cur': 0,
        'start_t': time.perf_counter(),
        'font': cv2.FONT_HERSHEY_SIMPLEX,
    }
    STREAM_SESSIONS[sess_id] = session
    return {'ok': True, 'session_id': sess_id}

# ---------------- Socket.IO: 실시간 프레임 처리 ----------------
@socketio.on('stream_frame')
def on_stream_frame(payload):
    sess_id = payload.get('session_id')
    data_url = payload.get('image')
    if not (sess_id and data_url) or sess_id not in STREAM_SESSIONS:
        emit('stream_result', {'ok': False, 'error': 'invalid session'})
        return
    sess = STREAM_SESSIONS[sess_id]

    # dataURL -> BGR
    try:
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        jpg = base64.b64decode(data_url)
        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('decode fail')
    except Exception as exc:
        emit('stream_result', {'ok': False, 'error': f'decode error: {exc}'})
        return

    ref = sess['ref']; ref_lm_arr = sess['ref_lm']
    ref_cap = sess['ref_cap']; ref_stride = sess['ref_stride']
    step_sec = sess['step_sec']; ref_fps = sess['ref_fps']
    rest_intv = sess['rest_intv']; region_w = sess['region_w']; angle_w = sess['angle_w']
    matcher = sess['matcher']; pe = sess['pe']
    font = sess['font']

    elapsed = time.perf_counter() - sess['start_t']
    hint_idx = int(elapsed / step_sec)
    hint_idx = min(hint_idx, ref.shape[0]-1)
    if ref_lm_arr is not None:
        hint_idx = min(hint_idx, len(ref_lm_arr)-1)

    display_left = frame.copy()

    # REST 구간
    if in_intervals(hint_idx, rest_intv):
        right = None
        if ref_cap is not None:
            ok_r, rframe, sess['ref_frame_cur'] = read_frame_at(ref_cap, hint_idx * ref_stride, sess['ref_frame_cur'])
            if ok_r:
                right = rframe
        combined = _combine(display_left, right)
        cv2.putText(combined, 'REST', (20, 40), font, 1.2, (200,200,255), 3, cv2.LINE_AA)
        _emit_frame(combined)
        return

    arr_pose = pe.infer(frame)
    static_sim = 0.0; motion_sim = 0.0; motion_mag = 1.0
    if arr_pose is not None:
        overlay = display_left.copy()
        draw_colored_skeleton(overlay, arr_pose)
        display_left = cv2.addWeighted(overlay, 0.5, display_left, 0.5, 0)
        lm = normalize_landmarks(arr_pose.copy())
        emb = pose_embedding(lm)
        _, ref_idx = matcher.step_with_hint(emb, hint_idx=hint_idx, search_radius=0)
        if ref_lm_arr is not None and ref_idx < len(ref_lm_arr):
            try:
                w_feat, _, _ = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=ref_lm_arr[ref_idx], region_w=region_w, angle_w=angle_w)
            except Exception:
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)[0]
        else:
            w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)[0]
        static_sim = weighted_cosine(matcher.ref[ref_idx], emb, w_feat)
        ref_emb = matcher.ref[ref_idx]
        if sess['prev_emb_live'] is not None and sess['prev_emb_ref'] is not None:
            d_live = emb - sess['prev_emb_live']
            d_ref = ref_emb - sess['prev_emb_ref']
            motion_sim = weighted_cosine(d_live, d_ref, w_feat)
            n_live = np.linalg.norm(d_live); n_ref = np.linalg.norm(d_ref)
            if n_live > 1e-6 or n_ref > 1e-6:
                motion_mag = min(n_live, n_ref) / (max(n_live, n_ref) + 1e-8)
        blended = static_sim  # w_pose=1.0, w_motion=0.0 기본값 반영
        score = ((blended + 1.0) * 0.5)
        sess['score_ema'] = exp_moving_avg(sess['score_ema'], score, alpha=0.9)
        sess['prev_emb_live'] = emb
        sess['prev_emb_ref'] = ref_emb
        if ref_lm_arr is not None and ref_idx < len(ref_lm_arr):
            try:
                msgs = angle_feedback(lm, ref_lm_arr[ref_idx], angle_tol_deg=10.0)
                sess['feedback'] = msgs if msgs else ['Good!']
            except Exception:
                pass

    # Reference frame
    right = None
    if ref_cap is not None:
        ok_r, rframe, sess['ref_frame_cur'] = read_frame_at(ref_cap, hint_idx * ref_stride, sess['ref_frame_cur'])
        if ok_r:
            right = rframe
    combined = _combine(display_left, right)
    pct = sess['score_ema'] * 100.0
    grade = 'PERFECT' if pct >= 85 else ('GOOD' if pct >= 70 else 'BAD')
    cv2.putText(combined, f'Sim: {pct:5.1f}% ({grade})', (20,40), font, 1.0, (0,255,0), 2, cv2.LINE_AA)
    y0 = 80
    for i, m in enumerate(sess['feedback'][:3]):
        cv2.putText(combined, m, (20, y0 + i*30), font, 0.8, (0,200,255), 2, cv2.LINE_AA)
    _emit_frame(combined)

# ---------------- 헬퍼 ----------------

def _combine(left, right):
    if right is None:
        return left
    h, w = left.shape[:2]
    rh, rw = right.shape[:2]
    if rh != h:
        scale = h / float(rh)
        right = cv2.resize(right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
    return np.hstack([left, right])


def _emit_frame(img):
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        emit('stream_result', {'ok': False, 'error': 'encode fail'})
        return
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    emit('stream_result', {'ok': True, 'image': 'data:image/jpeg;base64,' + b64})

# ---------------- 실행 ----------------
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5003)), debug=True)

