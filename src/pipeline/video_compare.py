"""
비디오 비교 모드: 로컬 사용자 동영상과 레퍼런스를 비교
"""

import os
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


def compare_videos(ref_path,
                   ref_lm_path,
                   user_video,
                   search_radius=0,
                   ema_alpha=0.9,
                   show_feedback=True,
                   ref_video=None,
                   ref_stride=2,
                   loop_ref=False,
                   rtf=1.0,
                   model_complexity=1,
                   w_pose=1.0,
                   w_motion=0.0,
                   time_penalty=0.0,
                   late_grace=5,
                   late_penalty=0.05,
                   rest_json: str | None = None,
                   out_video: str | None = None):
    """
    로컬 사용자 동영상(user_video)과 레퍼런스(임베딩/영상)를 프레임 기준으로 동기화하여 유사도를 비교/시각화.
    - ref_path: extract.py로 만든 레퍼런스 임베딩 .npy
    - user_video: 비교 대상 로컬 동영상 경로 (웹캠 대신 사용)
    - ref_video: 레퍼런스 가이드 영상(선택). 있으면 오른쪽에 표시하고 타임라인 동기화에 사용
    - ref_stride: extract 시 사용한 stride (레퍼런스 임베딩 1스텝 = ref_stride 프레임)
    - rtf: 재생 속도 배수. 1.0이면 사용자 영상의 실제 fps 기준으로 동기화
    - out_video: 지정 시, 시각화 결과를 동영상으로 저장(mp4/h264 등, FourCC 자동 설정)
    """
    # --- Load reference embeddings & (optional) per-frame landmarks for feedback ---
    rest_json = "data/rest_exc.json"
    ref = np.load(ref_path)
    ref_lm = np.load(ref_lm_path)
    #lm_path = ref_path.replace('.npy', '_lm.npy')
    #ref_lm = None
    #try:
    #    ref_lm = np.load(lm_path)
    #except Exception:
    #    ref_lm = None

    # --- Open user/local video ---
    ucap = cv2.VideoCapture(user_video)
    if not ucap.isOpened():
        raise SystemExit(f"사용자 영상 열기 실패: {user_video}")
    user_fps = ucap.get(cv2.CAP_PROP_FPS) or 30.0
    user_fps = float(user_fps if user_fps > 1e-3 else 30.0)
    user_total_frames = int(ucap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # --- Prepare reference video (optional) ---
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

    # --- Pacing: each embedding step corresponds to ref_stride frames of reference video ---
    step_sec = ref_stride / (ref_fps * max(1e-6, rtf))

    # --- REST intervals (ref timeline -> embedding indices) ---
    rest_intervals_emb = []
    if rest_json and ref_video is not None:
        ref_base = os.path.basename(ref_video)
        rest_intervals_emb = load_rest_intervals_json(rest_json, ref_base, ref_fps, ref_stride)

    # --- Pose extractor ---
    pe = PoseExtractor(static_image_mode=False, model_complexity=model_complexity)
    matcher = OnlineMatcher(ref)

    # --- Optional video writer ---
    writer = None
    if out_video:
        # Probe a first frame to get size
        ok_u, uframe = ucap.read()
        if not ok_u:
            ucap.release()
            if ref_cap is not None:
                ref_cap.release()
            raise SystemExit("사용자 영상에서 프레임을 읽을 수 없습니다.")
        # Prepare a matching ref frame for width/height
        display_right = None
        if ref_cap is not None:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_r, rframe = ref_cap.read()
            if ok_r:
                display_right = rframe
        h, w = uframe.shape[:2]
        if display_right is not None:
            rh, rw = display_right.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            out_w = uframe.shape[1] + display_right.shape[1]
            out_h = h
        else:
            out_w, out_h = w, h
        # Reset user cap to frame 0 for actual processing
        ucap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video, fourcc, user_fps, (out_w, out_h))

    # --- Main loop over user video frames ---
    prev_live_emb = None
    prev_ref_emb = None
    score_ema = 0.0
    last_feedback = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    grade_counts = {"PERFECT": 0, "GOOD": 0, "BAD": 0}
    graded_total = 0
    ref_frame_cur = 0
    user_idx = 0

    while True:
        ok_u, uframe = ucap.read()
        if not ok_u:
            break

        # Compute synchronized hint index from *user timeline* (no wall-clock)
        elapsed_user_sec = (user_idx / user_fps)
        hint_idx = int(elapsed_user_sec / step_sec)
        hint_idx = min(hint_idx, ref.shape[0] - 1)
        if ref_lm is not None:
            hint_idx = min(hint_idx, len(ref_lm) - 1)

        # If REST interval on ref timeline, overlay and skip scoring
        if in_intervals(hint_idx, rest_intervals_emb):
            display_left = uframe
            display_right = None
            if ref_cap is not None:
                target_ref_frame = hint_idx * ref_stride
                ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
                if ok_ref:
                    display_right = ref_frame
            # compose
            h, w = display_left.shape[:2]
            if display_right is not None:
                rh, rw = display_right.shape[:2]
                if rh != h:
                    scale = h / float(rh)
                    display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([display_left, display_right])
            else:
                combined = display_left
            cv2.putText(combined, "REST", (20, 40), font, 1.4, (255, 200, 200), 3, cv2.LINE_AA)
            if ref_cap is not None:
                cv2.putText(combined, "REFERENCE", (display_left.shape[1] + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
            if writer is not None:
                writer.write(combined)
            else:
                cv2.imshow("Fitness Dance - Video Compare (left: USER, right: REFERENCE)", combined)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
            user_idx += 1
            continue

        # Pose on user frame
        arr = pe.infer(uframe)
        if arr is not None:

            # arr: (33,4) with x,y,z,visibility (x,y는 0~1 정규화)
            overlay = uframe.copy()
            draw_colored_skeleton(overlay, arr)  # 원본 좌표 기준
            # 혹은 normalize 후 좌표로 그릴거면 normalize_landmarks(arr.copy())의 x,y가 이미 정규화된 좌표이므로
            # 화면 합성은 별도 좌표계 주의!
            alpha = 0.5
            # 오버레이를 반투명으로 합성
            uframe = cv2.addWeighted(overlay, alpha, uframe, 1 - alpha, 0)

            # 기준점(골반 중심)으로 평행이동, 스케일 정규화, 좌우 어깨 방향으로 회전 보정
            lm = normalize_landmarks(arr.copy())
            emb = pose_embedding(lm)
            sim, ref_idx = matcher.step_with_hint(emb, hint_idx=hint_idx, search_radius=search_radius)

            # Build per-feature weights (use ref landmarks if available at this index)
            if ref_lm is not None and ref_idx < len(ref_lm):
                #w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=ref_lm[ref_idx])
                region_w, angle_w = load_weights_for_video(ref_video)
                w_feat, region_w, angle_w = build_static_feature_weights(
                    BONES, ANGLE_TRIPLES,
                    lm_for_vis=ref_lm[ref_idx],
                    region_w=region_w,
                    angle_w=angle_w
                )
            else:
                w_feat = build_static_feature_weights(BONES, ANGLE_TRIPLES, lm_for_vis=None)

            static_sim = weighted_cosine(matcher.ref[ref_idx], emb, w_feat)
            motion_sim = 0.0
            motion_mag_match = 1.0
            ref_emb = matcher.ref[ref_idx]

            if prev_live_emb is not None and prev_ref_emb is not None:
                d_live = emb - prev_live_emb
                d_ref  = ref_emb - prev_ref_emb
                motion_sim = weighted_cosine(d_live, d_ref, w_feat)
                n_live = np.linalg.norm(d_live)
                n_ref  = np.linalg.norm(d_ref)
                if n_live > 1e-6 or n_ref > 1e-6:
                    motion_mag_match = min(n_live, n_ref) / (max(n_live, n_ref) + 1e-8)
                    #motion_mag_match = 1.0
                else:
                    motion_mag_match = 1.0

            # alignment penalty (late-friendly)
            if search_radius > 0:
                delta = ref_idx - hint_idx
                if delta < 0:
                    late_dt = -delta
                    if late_dt <= late_grace:
                        align_factor = 1.0
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

            #normalize
            #score = np.clip((score - 50) / 45 * 100, 0, 100)
            score_ema = exp_moving_avg(score_ema, score, alpha=ema_alpha)

            prev_live_emb = emb
            prev_ref_emb  = ref_emb

            msgs = []
            if show_feedback and ref_lm is not None and ref_idx < len(ref_lm):
                msgs = angle_feedback(lm, ref_lm[ref_idx], angle_tol_deg=10.0)
                last_feedback = msgs if len(msgs)>0 else ["Good! Nice pose"]


        else:
            static_sim = 0.0
            motion_sim = 0.0
            motion_mag_match = 1.0
            last_feedback = ["cannot extract pose"]

        # Prepare right (reference frame)
        display_right = None
        if ref_cap is not None:
            target_ref_frame = hint_idx * ref_stride
            ok_ref, ref_frame, ref_frame_cur = read_frame_at(ref_cap, target_ref_frame, ref_frame_cur)
            if ok_ref:
                display_right = ref_frame

        # Compose and overlay
        display_left = uframe
        h, w = display_left.shape[:2]
        if display_right is not None:
            rh, rw = display_right.shape[:2]
            if rh != h:
                scale = h / float(rh)
                display_right = cv2.resize(display_right, (int(rw*scale), h), interpolation=cv2.INTER_LINEAR)
            combined = np.hstack([display_left, display_right])
        else:
            combined = display_left

        # --- Rolling similarity buffer for stability ---
        if 'score_window' not in locals():
            score_window = []
        score_window.append(score_ema)
        if len(score_window) > int(user_fps * 3):  # 3초 평균
            score_window.pop(0)
        avg_score = float(np.nan_to_num(np.mean(score_window) if len(score_window) > 0 else score_ema, nan=0.0, posinf=1.0, neginf=0.0))

        avg_50_95 = avg_score * 100
        avg_pct = (avg_50_95 - 50) / 45.0
        if avg_pct > 1.0:
            avg_pct = 1.0
        avg_pct = avg_pct * 100.0

        grade_text = ""
        grade_color = (255, 255, 255)
        if avg_pct >= 70:
            grade_text = "PERFECT"
            grade_color = (0, 255, 0)
        elif avg_pct >= 65:
            grade_text = "GOOD"
            grade_color = (0, 200, 255)
        else:
            grade_text = "BAD"
            grade_color = (255, 0, 0)

        if grade_text:
            key = grade_text
            if key in grade_counts:
                grade_counts[key] += 1
                graded_total += 1

        cv2.putText(combined, f"Similarity(avg): {avg_pct:.1f}%  {avg_score*100:.1f}%", (20, 40), font, 1.0, (0,255,0), 2, cv2.LINE_AA)
        if grade_text:
            cv2.putText(combined, grade_text, (20, 90), font, 1.4, grade_color, 3, cv2.LINE_AA)
        y0 = 120
        for i, m in enumerate(last_feedback[:3]):
            cv2.putText(combined, m, (20, y0 + i*30), font, 0.8, (0,200,255), 2, cv2.LINE_AA)
        cv2.putText(combined, f"pose={static_sim:+.2f}  motion={motion_sim:+.2f}  mag={motion_mag_match:.2f}",
                    (20, y0 + 3*30), font, 0.6, (200,200,200), 1, cv2.LINE_AA)
        if ref_cap is not None:
            cv2.putText(combined, "REFERENCE", (w + 20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(combined)
        else:
            cv2.imshow("Fitness Dance - Video Compare (left: USER, right: REFERENCE)", combined)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break

        user_idx += 1

        # End of embeddings handling: if hint exceeds, loop ref video timeline or stop
        if hint_idx >= ref.shape[0] - 1:
            if loop_ref and ref_cap is not None:
                if ref_total_frames:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ref_frame_cur = 0
            else:
                # No loop: keep showing last ref frame; processing continues until user video ends
                pass


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
            y = 500
            cv2.putText(canvas, f"FINAL RANK: {final_rank}", (20, y), font, 1.4, (0,255,0), 3, cv2.LINE_AA); y += 50
            cv2.putText(canvas, f"PERFECT: {p_cnt} ({p_ratio*100:.1f}%)", (20, y), font, 1.0, (0,255,0), 2, cv2.LINE_AA); y += 35
            cv2.putText(canvas, f"GOOD   : {g_cnt} ({(g_cnt/graded_total)*100:.1f}%)", (20, y), font, 1.0, (0,200,255), 2, cv2.LINE_AA); y += 35
            cv2.putText(canvas, f"BAD    : {b_cnt} ({(b_cnt/graded_total)*100:.1f}%)", (20, y), font, 1.0, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow("Fitness Dance - Video Compare (left: USER, right: REFERENCE)", canvas)
            if writer is not None:
                writer.write(canvas)
            cv2.waitKey(3000)
        except Exception:
            pass
    # cleanup
    ucap.release()
    if ref_cap is not None:
        ref_cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

