"""
실시간 웹캠 비교 모드
"""

import os
import time
import math
import cv2
import numpy as np

from .extractor import PoseExtractor
from .matcher import OnlineMatcher, read_frame_at
from .pose_utils import (
    BONES, ANGLE_TRIPLES,
    normalize_landmarks, pose_embedding, angle_feedback,
    draw_colored_skeleton
)
from .similarity import (
    build_static_feature_weights, weighted_cosine, exp_moving_avg, load_weights_for_video
)
from .rest_intervals import load_rest_intervals_json, in_intervals
from .utils import parse_bgr, play_beep, start_ref_audio_player, stop_ref_audio_player


def live_play(ref_path, ref_lm_path, camera=0, search_radius=0, ema_alpha=0.9, show_feedback=True,
              ref_video=None, ref_stride=2, loop_ref=False, rtf=1.0, model_complexity=1,
              warmup_sec=5.0, countdown_color="0,215,255", countdown_beep=True, play_ref_audio=True,
              w_pose=1.0, w_motion=0.0, time_penalty=0.0, late_grace=5, late_penalty=0.05,
              rest_json: str | None = None,
              out_video: str | None = None,
              overlay_alpha: float = 0.5):
    """
    compare_videos()와 동일한 로직/연출을 유지하고,
    입력만 실시간 카메라로 바꾼 버전.
    """
    rest_json = "data/rest_exc.json"
    ref = np.load(ref_path)
    #lm_path = ref_path.replace('.npy', '_lm.npy')
    #try:
    #    ref_lm = np.load(lm_path, allow_pickle=True)
    #except Exception:
    #    ref_lm = None
    ref_lm = np.load(ref_lm_path)
    camera = 0
    search_radius = 0
    ema_alpha = 0.95
    show_feedback = True
    ref_stride = 2
    rtf = 1.0
    model_complexity = 1
    warmup_sec = 5.0
    countdown_beep = True
    play_ref_audio = True
    w_pose = 1.0
    w_motion = 0.0
    time_penalty = 0.0
    late_grace = 5
    late_penalty = 0.05

    # --- Reference video 준비 (동일) ---
    ref_cap = None
    ref_fps = 30.0
    ref_total_frames = None
    if ref_video is not None and os.path.exists(ref_video):
        ref_cap = cv2.VideoCapture(ref_video)
        if ref_cap.isOpened():
            fps_val = ref_cap.get(cv2.CAP_PROP_FPS)
            if fps_val and fps_val > 1e-3:
                ref_fps = float(fps_val)
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            ref_cap = None

    # 임베딩 한 스텝의 시간 간격(동일)
    step_sec = ref_stride / (ref_fps * max(1e-6, rtf))

    # --- 쉬는 구간 로드(동일) ---
    rest_intervals_emb = []
    if rest_json and ref_video is not None:
        ref_base = os.path.basename(ref_video)
        rest_intervals_emb = load_rest_intervals_json(rest_json, ref_base, ref_fps, ref_stride)

    # --- 카메라 ---
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"카메라 열기 실패: {camera}")

    # --- out_video 준비 (compare_videos와 동일한 합성폭으로) ---
    writer = None
    out_w = out_h = None
    if out_video:
        # 카메라 한 프레임 읽어 크기 파악
        ok_probe, probe = cap.read()
        if not ok_probe:
            cap.release()
            if ref_cap is not None:
                ref_cap.release()
            raise SystemExit("카메라 프레임을 읽을 수 없습니다.")
        # 우측 ref 첫 프레임
        display_right0 = None
        if ref_cap is not None:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_r, rframe0 = ref_cap.read()
            if ok_r:
                display_right0 = rframe0
        h, w = probe.shape[:2]
        if display_right0 is not None:
            rh, rw = display_right0.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right0 = cv2.resize(display_right0, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            out_w = w + display_right0.shape[1]
            out_h = h
        else:
            out_w, out_h = w, h
        # 프레임레이트는 ref_fps 또는 30으로 저장
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video, fourcc, ref_fps if ref_fps > 1e-3 else 30.0, (out_w, out_h))
        # ref_cap 포지션 복구
        if ref_cap is not None:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pe = PoseExtractor(static_image_mode=False, model_complexity=model_complexity)
    matcher = OnlineMatcher(ref)

    prev_live_emb = None
    prev_ref_emb  = None
    score_ema = 0.0
    score_window = []  # ← compare_videos와 동일: 롤링 평균 버퍼
    last_feedback = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    cd_color = parse_bgr(countdown_color, default=(0,215,255))
    last_beep_whole = None
    ref_audio_proc = None

    start_total = time.perf_counter()
    sync_start_t = None
    warmup_done = False
    ref_frame_cur = 0
    grade_counts = {"PERFECT": 0, "GOOD": 0, "BAD": 0}
    graded_total = 0
    window_name = "Fitness Dance - Live (left: YOU, right: REFERENCE)"

    while True:
        now = time.perf_counter()
        elapsed_total = now - start_total

        ok, frame = cap.read()
        if not ok:
            break
        display_left = frame.copy()

        # ----- Warmup (그대로 유지) -----
        static_sim = 0.0
        motion_sim = 0.0
        motion_mag_match = 1.0

        if not warmup_done and elapsed_total < warmup_sec:
            display_right = None
            if ref_cap is not None:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_ref, ref_frame = ref_cap.read()
                if ok_ref:
                    display_right = ref_frame
            h, w = display_left.shape[:2]
            if display_right is not None:
                rh, rw = display_right.shape[:2]
                if rh != h:
                    scale = h / float(rh)
                    display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display_left, display_right])
            else:
                combined = display_left
            remain = max(0.0, warmup_sec - elapsed_total)
            cv2.putText(combined, f"준비하세요... {remain:0.1f}s", (20, 40), font, 1.0, cd_color, 2, cv2.LINE_AA)

            if countdown_beep:
                whole = int(math.ceil(remain))
                if whole != last_beep_whole and 1 <= whole <= 3:
                    play_beep(freq=700.0 + 100.0*whole, dur=0.15)
                    last_beep_whole = whole
                if remain <= 0.05 and last_beep_whole != 0:
                    play_beep(freq=1400.0, dur=0.25); last_beep_whole = 0

            if ref_cap is not None:
                cv2.putText(combined, "REFERENCE (5초 후 시작)", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, combined)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
            # out_video에는 워밍업 화면 저장 안 함(원하면 여기에서 writer.write(combined))
            continue

        if not warmup_done and elapsed_total >= warmup_sec:
            warmup_done = True
            sync_start_t = time.perf_counter()
            score_ema = 0.0
            score_window.clear()
            last_feedback = ["Start! Follow the reference video 💪"]
            if ref_cap is not None:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ref_frame_cur = 0
            if play_ref_audio and ref_video is not None and os.path.exists(ref_video):
                stop_ref_audio_player(ref_audio_proc)
                ref_audio_proc = start_ref_audio_player(ref_video_path=ref_video, start_sec=0.0)
            continue

        # ---- 동기 인덱스 ----
        elapsed = time.perf_counter() - sync_start_t if sync_start_t is not None else 0.0
        hint_idx = int(elapsed / step_sec)
        if ref_lm is not None: hint_idx = min(hint_idx, len(ref_lm) - 1)
        hint_idx = min(hint_idx, ref.shape[0] - 1)

        # ---- 쉬는 구간 처리(동일) ----
        if in_intervals(hint_idx, rest_intervals_emb):
            display_right = None
            if ref_cap is not None:
                target_ref_frame = hint_idx * ref_stride
                ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
                if ok_ref: display_right = ref_frame
            h, w = display_left.shape[:2]
            if display_right is not None:
                rh, rw = display_right.shape[:2]
                if rh != h:
                    scale = h / float(rh)
                    display_right = cv2.resize(display_right, (int(rw * scale), h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display_left, display_right])
            else:
                combined = display_left
            cv2.putText(combined, "REST", (20, 40), font, 1.4, (255,200,200), 3, cv2.LINE_AA)
            if ref_cap is not None:
                cv2.putText(combined, "REFERENCE", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            if writer is not None: writer.write(combined)
            cv2.imshow(window_name, combined)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
            continue

        # ---- 포즈 추론 ----
        arr = pe.infer(frame)
        if arr is not None:
            # (1) 스켈레톤 오버레이( compare_videos와 동일 )
            overlay = display_left.copy()
            draw_colored_skeleton(overlay, arr)
            display_left = cv2.addWeighted(overlay, overlay_alpha, display_left, 1 - overlay_alpha, 0)

            # (2) 임베딩 및 매칭
            lm = normalize_landmarks(arr.copy())
            emb = pose_embedding(lm)
            sim, ref_idx = matcher.step_with_hint(emb, hint_idx=hint_idx, search_radius=search_radius)

            # (3) 가중치, 정적/모션 유사도
            if ref_lm is not None and ref_idx < len(ref_lm):
                #w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=ref_lm[ref_idx])
                region_w, angle_w = load_weights_for_video(ref_video)
                w_feat, region_w, angle_w = build_static_feature_weights(
                    BONES, ANGLE_TRIPLES,
                    lm_for_vis=ref_lm[ref_idx],
                    region_w=region_w,
                    angle_w=angle_w
                )
                #print(w_feat)
            else:
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)

            static_sim = weighted_cosine(matcher.ref[ref_idx], emb, w_feat)
            motion_sim = 0.0; motion_mag_match = 1.0
            ref_emb = matcher.ref[ref_idx]
            if prev_live_emb is not None and prev_ref_emb is not None:
                d_live = emb - prev_live_emb
                d_ref  = ref_emb - prev_ref_emb
                motion_sim = weighted_cosine(d_live, d_ref, w_feat)
                n_live = np.linalg.norm(d_live); n_ref = np.linalg.norm(d_ref)
                if n_live > 1e-6 or n_ref > 1e-6:
                    motion_mag_match = min(n_live, n_ref) / (max(n_live, n_ref) + 1e-8)

            # (4) 시간 이탈 패널티(동일)
            if search_radius > 0:
                delta = ref_idx - hint_idx
                if delta < 0:
                    late_dt = -delta
                    if late_dt <= late_grace: align_factor = 1.0
                    else:
                        span = max(1, search_radius - late_grace)
                        align_factor = 1.0 - late_penalty * ((late_dt - late_grace) / span)
                elif delta > 0:
                    span = max(1, search_radius)
                    align_factor = 1.0 - time_penalty * (delta / span)
                else:
                    align_factor = 1.0
                align_factor = max(0.0, min(1.0, align_factor))
            else:
                align_factor = 1.0

            blended = (w_pose * static_sim) + (w_motion * motion_sim * motion_mag_match)
            score = ((blended + 1.0) * 0.5) * align_factor

            score_ema = exp_moving_avg(score_ema, score, alpha=ema_alpha)

            prev_live_emb = emb
            prev_ref_emb  = ref_emb

            if show_feedback and ref_lm is not None and ref_idx < len(ref_lm):
                msgs = angle_feedback(lm, ref_lm[ref_idx], angle_tol_deg=10.0)
                last_feedback = msgs if len(msgs)>0 else ["Good! Nice pose 👏"]
        else:
            last_feedback = ["cannot extract pose"]

        # ---- 우측 ref 프레임 준비 (동일) ----
        display_right = None
        if ref_cap is not None:
            target_ref_frame = hint_idx * ref_stride
            ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
            if not ok_ref and loop_ref and ref_total_frames:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ref_frame_cur = 0
                ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, 0, ref_frame_cur)
            if ok_ref:
                display_right = ref_frame

        # ---- 합성 & 오버레이( compare_videos와 완전 동일 ) ----
        h, w = display_left.shape[:2]
        if display_right is not None:
            rh, rw = display_right.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            combined = np.hstack([display_left, display_right])
        else:
            combined = display_left

        # 롤링 평균(3초) + 50~95 -> 0~100 정규화 & 등급
        score_window.append(score_ema)
        # 카메라 fps를 모를 수 있어 ref_fps 기준 3초 윈도우로 사용
        if len(score_window) > int(ref_fps * 3):
            score_window.pop(0)
        avg_score = float(np.nan_to_num(np.mean(score_window) if len(score_window)>0 else score_ema, nan=0.0, posinf=1.0, neginf=0.0))

        avg_50_95 = avg_score * 100.0
        avg_pct   = ((avg_50_95 - 50.0) / 45.0) * 100.0
        # 보기 좋게 0~100로 클립
        avg_pct   = float(np.clip(avg_pct, 0.0, 100.0))

        grade_text, grade_color = "", (255,255,255)
        if avg_pct >= 65:
            grade_text, grade_color = "PERFECT", (0,255,0)
        elif avg_pct >= 60:
            grade_text, grade_color = "GOOD", (0,200,255)
        else:
            grade_text, grade_color = "BAD", (255,0,0)

        if grade_text:
            key = grade_text
            if key in grade_counts:
                grade_counts[key] += 1
                graded_total += 1

        cv2.putText(combined, f"Similarity(avg): {avg_pct:.1f}%  {avg_score*100:.1f}%", (20, 40), font, 1.0, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(combined, grade_text, (20, 90), font, 1.4, grade_color, 3, cv2.LINE_AA)
        y0 = 120
        for i, m in enumerate(last_feedback[:3]):
            cv2.putText(combined, m, (20, y0 + i*30), font, 0.8, (0,200,255), 2, cv2.LINE_AA)
        cv2.putText(combined, f"pose={static_sim:+.2f}  motion={motion_sim:+.2f}  mag={motion_mag_match:.2f}",
                    (20, y0 + 3*30), font, 0.6, (200,200,200), 1, cv2.LINE_AA)
        if ref_cap is not None:
            cv2.putText(combined, "REFERENCE", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)

        if writer is not None:
            if out_w is None or out_h is None:
                out_h, out_w = combined.shape[:2]
            writer.write(combined)

        cv2.imshow(window_name, combined)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

        # 임베딩 끝 처리(동일)
        if hint_idx >= ref.shape[0] - 1:
            if loop_ref:
                sync_start_t = time.perf_counter()
                if ref_cap is not None:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ref_frame_cur = 0
                if play_ref_audio and ref_video is not None and os.path.exists(ref_video):
                    stop_ref_audio_player(ref_audio_proc)
                    ref_audio_proc = start_ref_audio_player(ref_video_path=ref_video, start_sec=0.0)
            else:
                break

    # --- Final summary & rank ---
    if graded_total > 0:
        p_cnt = grade_counts["PERFECT"]
        g_cnt = grade_counts["GOOD"]
        b_cnt = grade_counts["BAD"]
        p_ratio = p_cnt / graded_total
        pg_ratio = (p_cnt + g_cnt) / graded_total

        # 랭크 규칙(원하면 숫자만 바꾸면 됨):
        # S: PERFECT >= 70%
        # A: PERFECT >= 50% 또는 (PERFECT+GOOD) >= 85%
        # B: (PERFECT+GOOD) >= 70%
        # C: (PERFECT+GOOD) >= 50%
        # F: 그 외
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

        # 콘솔 출력
        print("\n===== LIVE PLAY SUMMARY =====")
        print(f"Frames graded: {graded_total}")
        print(f"PERFECT: {p_cnt} ({p_ratio*100:.1f}%)")
        print(f"GOOD   : {g_cnt} ({(g_cnt/graded_total)*100:.1f}%)")
        print(f"BAD    : {b_cnt} ({(b_cnt/graded_total)*100:.1f}%)")
        print(f"FINAL RANK: {final_rank}")

        # 화면에 2초간 오버레이(가능하면)
        try:
            if 'combined' in locals():
                canvas = combined.copy()
            else:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            y = 200
            cv2.putText(canvas, f"FINAL RANK: {final_rank}", (20, y), font, 1.4, (0,255,0), 3, cv2.LINE_AA); y += 50
            cv2.putText(canvas, f"PERFECT: {p_cnt} ({p_ratio*100:.1f}%)", (20, y), font, 1.0, (0,255,0), 2, cv2.LINE_AA); y += 35
            cv2.putText(canvas, f"GOOD   : {g_cnt} ({(g_cnt/graded_total)*100:.1f}%)", (20, y), font, 1.0, (0,200,255), 2, cv2.LINE_AA); y += 35
            cv2.putText(canvas, f"BAD    : {b_cnt} ({(b_cnt/graded_total)*100:.1f}%)", (20, y), font, 1.0, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, canvas)
            if writer is not None:
                writer.write(canvas)
            cv2.waitKey(2000)
        except Exception:
            pass
    cap.release()
    if ref_cap is not None:
        ref_cap.release()
    stop_ref_audio_player(ref_audio_proc)
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

