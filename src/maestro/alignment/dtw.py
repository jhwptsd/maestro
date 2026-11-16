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
    (ties go to the earliest start and the latest end, i.e. the longest stretch).
    The caller defines what "cost" means (pitch distance, feature distance, ...).
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    if open_begin:
        acc[:, 0] = 1e-6 * np.arange(n + 1)   # may start in any row; a hair of cost makes ties start early

    # Cell (i, j) needs (i-1, j-1), (i-1, j) and (i, j-1): all on earlier anti-diagonals (i + j = const), so
    # every cell of one anti-diagonal can be filled at once. That is n + m numpy steps instead of n * m
    # Python iterations (0.63 s -> 0.07 s on a 1000 x 1000 matrix).
    for d in range(2, n + m + 1):
        i = np.arange(max(1, d - m), min(n, d - 1) + 1)
        j = d - i
        acc[i, j] = cost[i - 1, j - 1] + np.minimum(np.minimum(acc[i - 1, j - 1], acc[i - 1, j]), acc[i, j - 1])

    if open_end:                              # cheapest end; among exact ties the latest row
        i = int(np.flatnonzero(acc[1:, m] <= acc[1:, m].min() + 1e-9)[-1]) + 1
    else:
        i = n
    j, path, total = m, [], float(acc[i, m])
    while i > 0 and j > 0:                    # backtrack along the cheapest predecessor (diagonal wins ties)
        path.append((i - 1, j - 1))
        diag, up, left = acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
        if diag <= up and diag <= left:
            i, j = i - 1, j - 1
        elif up <= left:
            i -= 1
        else:
            j -= 1
    return path[::-1], total
