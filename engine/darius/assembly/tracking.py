"""Plan-conditioned progress tracking: how far along the build is, from noisy looks at it.

A vision model looking at one frame is unreliable on its own: a panel at the wrong
angle hides a joint, a hand covers a dowel. But a build cannot be anywhere: it
follows the manual, mostly one step at a time, and almost never goes backwards. So
progress is tracked as a hidden Markov model over k, the number of completed steps:

    transitions   k -> k (stay), k -> k+1 (finish a step), k -> k+2 (two at once),
                  k -> k-1 (undo a mistake); mostly 'stay' between frames
    emissions     each frame's observation gives P(step i done) for the steps it can
                  see; state k predicts steps 1..k done and k+1..N not, so
                      log P(obs | k) = w * sum_i [ log p_i if i <= k else log(1 - p_i) ]
                  with each p_i capped to [0.15, 0.85] (one image is never conclusive)
                  and w < 1 tempering correlated evidence from the same image

Live, the forward algorithm updates the belief after every frame (`Tracker`). For a
whole video, Viterbi finds the most likely monotone path and forward-backward gives
the confidence of every moment (`decode`), which turns a list of noisy frame
readings into a clean timeline of "step i finished at t". Reconstructing the
assembly graph is then exact given the plan: the joints of steps 1..k.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .schema import Belief, Observation

STAY = 0.86
ADVANCE = 0.11
SKIP = 0.02
BACK = 0.01
TEMPER = 0.7
# A single frame is never allowed to be surer than this about any step: vision models are
# overconfident on one image, and one bad frame must not move the build two steps ahead.
EVIDENCE_CAP = 0.85


def transition_matrix(steps: int) -> np.ndarray:
    """T[a, b] = P(next = b | current = a) over 0..steps completed steps."""
    n = steps + 1
    t = np.zeros((n, n))
    for a in range(n):
        for b, p in ((a, STAY), (a + 1, ADVANCE), (a + 2, SKIP), (a - 1, BACK)):
            if 0 <= b < n:
                t[a, b] += p
        t[a] /= t[a].sum()  # the ends have fewer moves
    return t


def log_emission(observation: Observation, steps: int) -> np.ndarray:
    """log P(observation | k completed), k = 0..steps."""
    out = np.zeros(steps + 1)
    for index, p in observation.step_done.items():
        if not 1 <= index <= steps:
            continue
        p = min(max(float(p), 1 - EVIDENCE_CAP), EVIDENCE_CAP)
        done, not_done = math.log(p), math.log(1 - p)
        # States with k >= index predict this step done.
        out[index:] += TEMPER * done
        out[:index] += TEMPER * not_done
    return out


def _normalize_log(values: np.ndarray) -> np.ndarray:
    values = values - values.max()
    p = np.exp(values)
    return p / p.sum()


def _belief(p: np.ndarray) -> Belief:
    k = int(np.argmax(p))
    return Belief(probabilities=[round(float(v), 5) for v in p], completed=k, confidence=round(float(p[k]), 4))


@dataclass
class Tracker:
    """Online belief over progress, updated one observation at a time (the live AR session)."""

    steps: int
    prior: np.ndarray | None = None
    belief: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.transitions = transition_matrix(self.steps)
        start = np.zeros(self.steps + 1)
        start[0] = 1.0
        self.belief = start if self.prior is None else np.asarray(self.prior, float) / np.sum(self.prior)

    def update(self, observation: Observation) -> Belief:
        predicted = self.belief @ self.transitions
        posterior = np.log(np.maximum(predicted, 1e-300)) + log_emission(observation, self.steps)
        self.belief = _normalize_log(posterior)
        return _belief(self.belief)

    def current(self) -> Belief:
        return _belief(self.belief)


@dataclass(frozen=True)
class Timeline:
    """A decoded video: the most likely number of completed steps at every observation, its
    posterior confidence, and when each step was finished."""

    times: list[float]
    path: list[int]
    confidence: list[float]
    posteriors: list[list[float]]
    finished_at: dict[int, float]


def decode(observations: list[Observation], steps: int) -> Timeline:
    """Viterbi path and forward-backward posteriors over a whole sequence of observations."""
    if not observations:
        return Timeline([], [], [], [], {})
    n = steps + 1
    log_t = np.log(np.maximum(transition_matrix(steps), 1e-300))
    emissions = np.array([log_emission(o, steps) for o in observations])
    start = np.full(n, -np.inf)
    start[0] = 0.0

    # Viterbi.
    score = start + emissions[0]
    back = np.zeros((len(observations), n), dtype=int)
    for t in range(1, len(observations)):
        candidates = score[:, None] + log_t
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(n)] + emissions[t]
    path = [int(np.argmax(score))]
    for t in range(len(observations) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()

    # Forward-backward, in log space.
    alpha = np.zeros((len(observations), n))
    alpha[0] = start + emissions[0]
    for t in range(1, len(observations)):
        alpha[t] = np.logaddexp.reduce(alpha[t - 1][:, None] + log_t, axis=0) + emissions[t]
    beta = np.zeros((len(observations), n))
    for t in range(len(observations) - 2, -1, -1):
        beta[t] = np.logaddexp.reduce(log_t + emissions[t + 1] + beta[t + 1], axis=1)
    posteriors = [_normalize_log(alpha[t] + beta[t]) for t in range(len(observations))]

    times = [o.time_s for o in observations]
    finished: dict[int, float] = {}
    for t in range(1, len(path)):
        for k in range(path[t - 1] + 1, path[t] + 1):
            finished.setdefault(k, times[t])
    return Timeline(
        times=times,
        path=path,
        confidence=[round(float(p[k]), 4) for p, k in zip(posteriors, path, strict=True)],
        posteriors=[[round(float(v), 5) for v in p] for p in posteriors],
        finished_at=finished,
    )
