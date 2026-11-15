# Implement dynamic time warping algorithm to align two sequences of features
#
# Generic DTW helper (no domain knowledge). Used by align_audio_score.py,
# align_video_score.py, and reference_alignment.py.
# Use Accurate Audio-to-Score Alignment for Expressive Violin Recordings (Syue et al., ISMIR 2017)
import numpy as np


def dtw(cost):
    """Classic DTW over a precomputed cost matrix (n x m), steps (1,1), (1,0), (0,1).

    Returns (path, total_cost) where path is a list of (i, j) index pairs from (0, 0) to (n-1, m-1).
    The caller defines what "cost" means (pitch distance, feature distance, ...).
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])

    # backtrack from the end along the cheapest predecessors
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return path[::-1], float(acc[n, m])
