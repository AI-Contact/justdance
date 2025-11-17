"""
Flask 기반 Pose Comparison 웹 데모 (실시간 웹 소켓 제거 버전)
"""

import os
import sys
import time
import uuid
import tempfile
from pathlib import Path
import subprocess

# 상위 디렉터리를 Python 경로에 추가하여 pipeline 모듈을 찾을 수 있도록 함
demo_dir = Path(__file__).parent
project_root = demo_dir.parent  # src 디렉터리
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# pipeline이 내부에서 data/* 상대경로를 사용하므로 CWD를 src로 변경
try:
    os.chdir(project_root)
except Exception:
    pass

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# compare_videos는 사용하지 않으므로 임포트 제거
# from pipeline import compare_videos

# 기본 디렉터리 설정
TEMP_DIR = Path(tempfile.gettempdir())
UPLOAD_DIR = TEMP_DIR / "pose_demo_uploads"
OUTPUT_DIR = TEMP_DIR / "pose_demo_outputs"
for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_NPY_EXTS = {".npy"}
ALLOWED_JSON_EXTS = {".json"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("POSE_DEMO_SECRET_KEY", "pose-demo-dev-secret")
# 대용량 업로드 허용 상향 (1GB)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """업로드 용량 초과 시 사용자에게 안내하고 홈으로 이동"""
    flash("Request Entity Too Large: 업로드 파일이 너무 큽니다. 더 작은 파일을 업로드하거나 파일 길이를 줄여주세요.", "error")
    return redirect(url_for("index"))


def cleanup_old_files(max_age_hours: int = 6) -> None:
    """임시 파일 디렉터리에서 오래된 파일을 정리한다."""
    cutoff = time.time() - max_age_hours * 3600
    for directory in (UPLOAD_DIR, OUTPUT_DIR):
        for path in directory.glob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except FileNotFoundError:
                continue


def allowed_file(filename: str, file_type: str = "video") -> bool:
    """파일 확장자 검증"""
    if file_type == "video":
        return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTS
    elif file_type == "npy":
        return Path(filename).suffix.lower() in ALLOWED_NPY_EXTS
    elif file_type == "json":
        return Path(filename).suffix.lower() in ALLOWED_JSON_EXTS
    return False


def parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@app.before_request
def before_request_cleanup() -> None:
    cleanup_old_files()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 필수 파일 업로드 검증 (레퍼런스 임베딩/랜드마크/레퍼런스 영상은 필수)
        ref_file = request.files.get("ref_embedding")
        if ref_file is None or ref_file.filename == "":
            flash("레퍼런스 임베딩(.npy) 파일을 업로드해주세요.", "error")
            return redirect(url_for("index"))
        if not allowed_file(ref_file.filename, "npy"):
            flash("레퍼런스 임베딩은 .npy 파일만 업로드 가능합니다.", "error")
            return redirect(url_for("index"))

        ref_lm_file = request.files.get("ref_landmarks")
        if ref_lm_file is None or ref_lm_file.filename == "":
            flash("레퍼런스 랜드마크(.npy) 파일을 업로드해주세요.", "error")
            return redirect(url_for("index"))
        if not allowed_file(ref_lm_file.filename, "npy"):
            flash("레퍼런스 랜드마크는 .npy 파일만 업로드 가능합니다.", "error")
            return redirect(url_for("index"))

        ref_video_file = request.files.get("ref_video")
        if ref_video_file is None or ref_video_file.filename == "":
            flash("레퍼런스 영상 파일(.mp4 등)을 업로드해주세요.", "error")
            return redirect(url_for("index"))
        if not allowed_file(ref_video_file.filename, "video"):
            flash("레퍼런스 영상은 동영상 파일만 업로드 가능합니다.", "error")
            return redirect(url_for("index"))

        # 사용자 영상은 선택 (없으면 별도 프로세스 live_play 실행)
        user_file = request.files.get("user_video")
        user_has_video = bool(user_file and user_file.filename and allowed_file(user_file.filename, "video"))

        # 선택적 REST JSON
        rest_json_file = request.files.get("rest_json")

        # 요청별 디렉터리 생성(원본 파일명 유지)
        req_id = uuid.uuid4().hex
        req_dir = UPLOAD_DIR / req_id
        req_dir.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        ref_path = req_dir / secure_filename(ref_file.filename)
        ref_file.save(str(ref_path))
        ref_lm_path = req_dir / secure_filename(ref_lm_file.filename)
        ref_lm_file.save(str(ref_lm_path))
        ref_video_path = req_dir / secure_filename(ref_video_file.filename)
        ref_video_file.save(str(ref_video_path))

        user_path = None
        if user_has_video:
            user_path = req_dir / secure_filename(user_file.filename)
            user_file.save(str(user_path))
        elif user_file and user_file.filename and not allowed_file(user_file.filename, "video"):
            flash("지원하지 않는 사용자 동영상 형식입니다.", "error")
            try:
                for p in [ref_path, ref_lm_path, ref_video_path]:
                    Path(p).unlink(missing_ok=True)
                req_dir.rmdir()
            except Exception:
                pass
            return redirect(url_for("index"))

        rest_json_path = None
        if rest_json_file and rest_json_file.filename:
            if not allowed_file(rest_json_file.filename, "json"):
                flash("REST JSON은 .json 파일만 업로드 가능합니다.", "error")
                try:
                    for p in [ref_path, ref_lm_path, ref_video_path, user_path]:
                        if p:
                            Path(p).unlink(missing_ok=True)
                    req_dir.rmdir()
                except Exception:
                    pass
                return redirect(url_for("index"))
            rest_json_path = req_dir / secure_filename(rest_json_file.filename)
            rest_json_file.save(str(rest_json_path))

        if user_has_video and user_path is not None:
            # 비교 모드: out_video 생성 없이 즉시 비교 화면(OpenCV 창) 표시
            cmd = [
                sys.executable,
                str(demo_dir / "run_compare.py"),
                "--ref", str(ref_path),
                "--ref-lm", str(ref_lm_path),
                "--user-video", str(user_path),
                "--ref-video", str(ref_video_path),
            ]
            try:
                subprocess.Popen(cmd)
            except Exception as exc:
                flash(f"비교 세션 시작 실패: {exc}", "error")
                # 업로드 정리(최소한의 롤백)
                try:
                    for p in [ref_path, ref_lm_path, ref_video_path, user_path, rest_json_path]:
                        if p:
                            Path(p).unlink(missing_ok=True)
                    req_dir.rmdir()
                except Exception:
                    pass
                return redirect(url_for("index"))

            flash("비교가 시작되었습니다. OpenCV 창에서 결과를 확인하세요 (종료: ESC/q)", "success")
            # 업로드 파일은 비교 프로세스에서 사용하므로 즉시 삭제하지 않음(주기 정리에 맡김)
            return redirect(url_for("index"))
        else:
            # 실시간 live_play 별도 프로세스 실행 (기본값 유지)
            cmd = [
                sys.executable,
                str(demo_dir / "run_live.py"),
                "--ref", str(ref_path),
                "--ref-lm", str(ref_lm_path),
                "--ref-video", str(ref_video_path),
            ]
            try:
                subprocess.Popen(cmd)
            except Exception as exc:
                print(f"[live_play subprocess error] {exc}")
                flash("실시간 비교 시작에 실패했습니다.", "error")
                return redirect(url_for("index"))

            flash("실시간 웹캠 비교가 별도 창에서 시작되었습니다. 종료: ESC/q", "success")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    """결과 동영상 파일 제공"""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/outputs/<path:filename>/download")
def download_output(filename: str):
    """결과 동영상 파일 다운로드"""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
