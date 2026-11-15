# Implement dynamic time warping algorithm to align two sequences of features
#
# Generic DTW helper (no domain knowledge). Used by align_audio_score.py,
# align_video_score.py, and reference_alignment.py.
# Open-begin/open-end ("subsequence") mode lets the second sequence match any stretch of the first,
# which is how a student who starts halfway down the page is aligned (see align_audio_score.py).
# Use Accurate Audio-to-Score Alignment for Expressive Violin Recordings (Syue et al., ISMIR 2017)
import numpy as np


def dtw(cost, open_begin=False, open_end=False):
    """Classic DTW over a precomputed cost matrix (n x m), steps (1,1), (1,0), (0,1).

    Returns (path, total_cost) where path is a list of (i, j) index pairs. Normally it runs from (0, 0)
    to (n-1, m-1). With open_begin/open_end the path may instead start/end at *any row* i, while still
    covering every column j: the m-sequence is matched against the best-fitting stretch of the n-sequence
    (ties go to the earliest start and the latest end, i.e. the longest stretch). The caller defines what "cost" means (pitch distance, ...).
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    if open_begin:
        acc[:, 0] = 1e-6 * np.arange(n + 1)  # may start in any row; a hair of cost makes ties start early
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])

    # backtrack from the end along the cheapest predecessors
    if open_end:                             # cheapest end; among exact ties the latest row
        i = int(np.flatnonzero(acc[1:, m] <= acc[1:, m].min() + 1e-9)[-1]) + 1
    else:
        i = n
    j, path, total = m, [], float(acc[i, m])
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return path[::-1], total
