"""
유틸리티 함수들: 색상 파싱, 비프음 재생, 오디오 재생 등
"""

import subprocess
import shutil
import numpy as np


def parse_bgr(color_str, default=(0, 215, 255)):
    """
    Parse 'B,G,R' into a BGR tuple of ints. Fallback to default on error.
    """
    try:
        parts = [int(x.strip()) for x in color_str.split(',')]
        if len(parts) != 3:
            return default
        return tuple(parts[:3])
    except Exception:
        return default


def play_beep(freq=1000.0, dur=0.2, sr=44100, amp=0.2):
    """
    Play a short sine beep using sounddevice if available.
    Non-blocking; failures are safely ignored.
    """
    try:
        import sounddevice as sd
        t = np.linspace(0, dur, int(sr*dur), endpoint=False, dtype=np.float32)
        wave = (amp*np.sin(2*np.pi*freq*t)).astype(np.float32)
        sd.play(wave, samplerate=sr, blocking=False)
    except Exception:
        pass


def start_ref_audio_player(ref_video_path, start_sec=0.0):
    """
    Try to play audio from the reference video file using ffplay (preferred) or afplay (macOS).
    Returns the subprocess.Popen handle or None on failure.
    """
    try:
        # Prefer ffplay if present
        if shutil.which("ffplay"):
            # -nodisp: no window, -autoexit: exit when done, -loglevel quiet: silent
            # -ss start time
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if start_sec and start_sec > 0:
                cmd += ["-ss", f"{start_sec}"]
            cmd += [ref_video_path]
            return subprocess.Popen(cmd)
        # Fallback to afplay on macOS (cannot seek reliably before Big Sur for video; we start from 0)
        if shutil.which("afplay"):
            cmd = ["afplay", ref_video_path]
            return subprocess.Popen(cmd)
    except Exception:
        return None
    return None


def stop_ref_audio_player(proc):
    """Stop the reference audio player process if running."""
    try:
        if proc and proc.poll() is None:
            proc.terminate()
    except Exception:
        pass

