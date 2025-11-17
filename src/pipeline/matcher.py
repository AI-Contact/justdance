"""
임베딩 매칭 관련 클래스 및 함수들
"""

import cv2
import numpy as np
from .similarity import cosine_sim


class OnlineMatcher:
    """레퍼런스 임베딩과 실시간 임베딩을 비교하는 매칭 클래스"""
    
    def __init__(self, ref_embs):
        self.ref = ref_embs
        self.T = ref_embs.shape[0]

    def step_with_hint(self, emb, hint_idx: int, search_radius: int = 0):
        """
        Compute similarity against a window centered at hint_idx.
        If search_radius==0, compare only with ref[hint_idx].
        Returns (best_sim, best_global_idx).
        """
        if self.T == 0:
            return 0.0, 0
        hint_idx = int(max(0, min(self.T - 1, hint_idx)))
        if search_radius <= 0:
            sim = cosine_sim(self.ref[hint_idx], emb)
            return float(sim), hint_idx
        lo = max(0, hint_idx - search_radius)
        hi = min(self.T, hint_idx + search_radius + 1)
        candidates = self.ref[lo:hi]
        sims = np.array([cosine_sim(c, emb) for c in candidates])
        k = int(np.argmax(sims))
        return float(sims[k]), lo + k


def read_frame_at(cap, target_idx, current_idx):
    """
    Try to advance to target_idx. If target is far behind, perform a seek.
    Returns (ok, frame, new_current_idx)
    """
    if target_idx < 0:
        target_idx = 0
    # If we are behind by more than 5 frames, seek for efficiency.
    if target_idx < current_idx or (target_idx - current_idx) > 5:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        return ok, frame, target_idx + 1
    # Otherwise, step forward incrementally.
    while current_idx < target_idx:
        ok = cap.grab()
        if not ok:
            return False, None, current_idx
        current_idx += 1
    ok, frame = cap.read()
    return ok, frame, current_idx + 1

