"""
Gait-Cycle and Rhythm Analysis (future work, plan Section 6 / experiment E3)
===========================================================================
Treats walking as a periodic process rather than a bag of frames. Pure numpy
(no scipy), operating on full-frame-rate canonical pose sequences.

Pipeline for one clip:

  1. Walking axis.  PCA of the left-minus-right heel displacement gives the
     axis along which the feet alternate (image x in a side view, image y /
     perspective in a frontal view). Its sign is fixed so toes point forward.
  2. Stride period.  Autocorrelation of the heel-separation signal; its
     fundamental is the stride frequency.
  3. Gait events.  Coordinate-based detection of Zeni et al. (2008):
     heel strike = local max of (heel - pelvis) along the walking axis,
     toe off = local min of (toe - pelvis).
  4. Parameters.  Cadence, stride time + variability, step asymmetry,
     step/stride regularity from the trunk-vertical autocorrelation
     (Moe-Nilssen & Helbostad 2004), harmonic ratio (Menz et al. 2003),
     spectral entropy, left-right phase synchrony (Hilbert phase), step
     length, step width, stance / double-support fractions, foot clearance.
  5. Cycle normalisation.  Any per-frame signal can be re-sampled to
     %-of-gait-cycle between heel strikes (RQ4: speed-invariant comparison).

Also provides DTW (distance, path, and warping a query onto a reference) for
the DTW-template verifier and DTW-aligned comparison.

Coordinates are expected isotropic (x rescaled by width/height), which the
descriptor builder in utils/gait_descriptors.py guarantees.

Author: DeepFake Detection Project
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

L_HIP, R_HIP = 23, 24
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32

# Typical adult values used only when a clip is too short/noisy to measure a
# parameter, so a missing measurement does not become a NaN in the network.
_RHYTHM_DEFAULTS = {
    "cadence_spm": 110.0,
    "stride_time_s": 1.1,
    "stride_time_cv": 0.05,
    "step_time_asym": 0.05,
    "step_regularity": 0.5,
    "stride_regularity": 0.5,
    "regularity_symmetry": 1.0,
    "harmonic_ratio_v": 1.5,
    "spectral_entropy": 0.5,
    "phase_offset": 0.0,
    "phase_locking": 0.8,
}
_STRIDE_DEFAULTS = {
    "stride_length": 0.8,
    "step_length_l": 0.4,
    "step_length_r": 0.4,
    "step_length_asym": 0.05,
    "step_width": 0.1,
    "stance_frac_l": 0.6,
    "stance_frac_r": 0.6,
    "double_support_frac": 0.2,
    "clearance_l": 0.1,
    "clearance_r": 0.1,
}


# ============================================================
# Signal utilities
# ============================================================


def lowpass(x: np.ndarray, fps: float, cutoff: float) -> np.ndarray:
    """Zero-phase FFT low-pass along axis 0 with a raised-cosine roll-off.
    Reflect-padded to suppress edge ringing."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 4:
        return x.copy()
    pad = min(n - 1, int(fps))
    xp = np.concatenate([x[pad:0:-1], x, x[-2 : -pad - 2 : -1]], axis=0)
    spec = np.fft.rfft(xp, axis=0)
    freqs = np.fft.rfftfreq(xp.shape[0], d=1.0 / fps)
    lo, hi = cutoff, cutoff * 1.5
    gain = np.clip((hi - freqs) / (hi - lo), 0, 1)
    gain = 0.5 - 0.5 * np.cos(np.pi * gain)
    shape = (-1,) + (1,) * (x.ndim - 1)
    out = np.fft.irfft(spec * gain.reshape(shape), n=xp.shape[0], axis=0)
    return out[pad : pad + n]


def bandpass(x: np.ndarray, fps: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase FFT band-pass (hard mask) along axis 0, mean removed."""
    x = np.asarray(x, dtype=float) - np.mean(x, axis=0)
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fps)
    mask = ((freqs >= lo) & (freqs <= hi)).astype(float)
    shape = (-1,) + (1,) * (x.ndim - 1)
    return np.fft.irfft(spec * mask.reshape(shape), n=x.shape[0], axis=0)


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """FFT Hilbert transform (scipy.signal.hilbert equivalent), 1-D."""
    n = len(x)
    spec = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[1 : (n + 1) // 2] = 2
    return np.fft.ifft(spec * h)


def autocorrelation(x: np.ndarray, unbiased: bool = False) -> np.ndarray:
    """Normalised autocorrelation of a 1-D signal (lag 0 = 1). The default
    biased estimate down-weights long lags, which are noisy in short clips."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if unbiased:
        ac = ac / (n - np.arange(n))
    return ac / (ac[0] + 1e-12)


def lagged_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Pearson correlation of a(t) with b(t + lag)."""
    if lag <= 0 or lag >= len(a) - 3:
        return np.nan
    x, y = a[:-lag], b[lag:]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _max_lagged_corr(a, b, lag: float, tol: float) -> float:
    """Best lagged correlation within lag * (1 +/- tol)."""
    lo, hi = int(np.floor(lag * (1 - tol))), int(np.ceil(lag * (1 + tol)))
    vals = [lagged_corr(a, b, k) for k in range(max(lo, 1), hi + 1)]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else np.nan


def pick_period(ac: np.ndarray, lo: int, hi: int, rel: float = 0.8) -> int:
    """First local autocorrelation peak in [lo, hi) reaching `rel` of the
    highest peak there -- avoids locking onto 2x or 3x the true period."""
    seg = ac[lo:hi]
    peaks = [
        i
        for i in range(1, len(seg) - 1)
        if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]
    ]
    if not peaks:
        return lo + int(np.argmax(seg))
    best = max(seg[i] for i in peaks)
    for i in peaks:
        if seg[i] >= rel * best:
            return lo + i
    return lo + peaks[0]


def local_extrema(x: np.ndarray, half_window: int, maxima: bool = True) -> np.ndarray:
    """Indices that are the max (or min) of their +/- half_window
    neighbourhood, excluding the first/last 2 samples (edge artefacts)."""
    s = x if maxima else -x
    n = len(s)
    w = max(1, int(half_window))
    idx = []
    for i in range(2, n - 2):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        if s[i] >= s[lo:hi].max() and s[i] > np.median(s):
            if not idx or i - idx[-1] > w:
                idx.append(i)
    return np.array(idx, dtype=int)


# ============================================================
# Gait analysis
# ============================================================


@dataclass
class GaitCycleAnalysis:
    fps: float
    t: np.ndarray
    axis: np.ndarray  # (2,) unit walking axis in the image plane
    heel_fwd_l: np.ndarray
    heel_fwd_r: np.ndarray
    toe_fwd_l: np.ndarray
    toe_fwd_r: np.ndarray
    heel_lat: np.ndarray  # L-R heel separation perpendicular to the axis
    trunk_vertical: np.ndarray  # detrended mid-hip vertical position
    toe_height_l: np.ndarray
    toe_height_r: np.ndarray
    stride_period: float  # seconds
    period_confidence: float  # autocorrelation peak height
    hs_l: np.ndarray  # frame indices
    hs_r: np.ndarray
    to_l: np.ndarray
    to_r: np.ndarray
    phase_l: np.ndarray  # radians, per frame
    phase_r: np.ndarray

    RHYTHM_NAMES = tuple(_RHYTHM_DEFAULTS)
    STRIDE_NAMES = tuple(_STRIDE_DEFAULTS)

    @property
    def stride_freq(self) -> float:
        return 1.0 / self.stride_period

    @property
    def n_cycles(self) -> int:
        return max(len(self.hs_l) - 1, 0) + max(len(self.hs_r) - 1, 0)

    # ---------------- rhythm ----------------

    def _times(self, idx):
        return self.t[idx] if len(idx) else np.array([])

    def stride_times(self) -> np.ndarray:
        out = []
        for hs in (self.hs_l, self.hs_r):
            if len(hs) > 1:
                out.extend(np.diff(self._times(hs)))
        return np.array(out)

    def step_times(self) -> Tuple[np.ndarray, np.ndarray]:
        """(left steps, right steps): time from the opposite heel strike to
        this side's heel strike."""
        ev = sorted([(i, "L") for i in self.hs_l] + [(i, "R") for i in self.hs_r])
        left, right = [], []
        for (i0, s0), (i1, s1) in zip(ev[:-1], ev[1:]):
            if s0 == s1:
                continue
            dt = self.t[i1] - self.t[i0]
            (left if s1 == "L" else right).append(dt)
        return np.array(left), np.array(right)

    def harmonic_amplitudes(self, signal: np.ndarray, n_harmonics: int = 10):
        """Amplitudes at k * stride frequency (k = 1..n), least-squares fit of
        sin/cos over the detrended signal."""
        t = self.t - self.t[0]
        s = np.asarray(signal, float)
        s = s - np.polyval(np.polyfit(t, s, 1), t)
        amps = []
        for k in range(1, n_harmonics + 1):
            w = 2 * np.pi * k * self.stride_freq
            basis = np.stack([np.sin(w * t), np.cos(w * t)], axis=1)
            coef, *_ = np.linalg.lstsq(basis, s, rcond=None)
            amps.append(np.hypot(*coef))
        return np.array(amps)

    def rhythm_dict(self) -> dict:
        d = dict(_RHYTHM_DEFAULTS)
        st = self.stride_times()
        st = st[(st > 0.5 * self.stride_period) & (st < 1.8 * self.stride_period)]
        if len(st):
            d["stride_time_s"] = float(np.median(st))
            d["cadence_spm"] = 120.0 / d["stride_time_s"]
            if len(st) > 1:
                d["stride_time_cv"] = float(np.std(st) / np.mean(st))
        else:
            d["stride_time_s"] = self.stride_period
            d["cadence_spm"] = 120.0 / self.stride_period
        left, right = self.step_times()
        if len(left) and len(right):
            ml, mr = np.median(left), np.median(right)
            d["step_time_asym"] = float(abs(ml - mr) / (0.5 * (ml + mr) + 1e-8))

        # Regularity in the spirit of Moe-Nilssen & Helbostad (2004), but on
        # the strongly periodic foot signals: pose jitter swamps the ~0.3%-of-
        # frame pelvis bounce that an accelerometer would measure.
        #   stride regularity: each foot signal vs itself one stride later
        #   step regularity:   left foot vs right foot half a stride later
        lag = int(round(self.stride_period * self.fps))
        feet_l = (self.heel_fwd_l, self.toe_fwd_l)
        feet_r = (self.heel_fwd_r, self.toe_fwd_r)
        stride_reg = np.nanmean(
            [_max_lagged_corr(s, s, lag, 0.2) for s in feet_l + feet_r]
        )
        step_reg = np.nanmean(
            [_max_lagged_corr(a, b, lag / 2, 0.2) for a, b in zip(feet_l, feet_r)]
            + [_max_lagged_corr(b, a, lag / 2, 0.2) for a, b in zip(feet_l, feet_r)]
        )
        if np.isfinite(step_reg):
            d["step_regularity"] = step_reg
        if np.isfinite(stride_reg):
            d["stride_regularity"] = stride_reg
        if np.isfinite(step_reg) and np.isfinite(stride_reg) and stride_reg > 0.05:
            d["regularity_symmetry"] = float(np.clip(step_reg / stride_reg, 0, 2))

        # Harmonic ratio (vertical): even / odd harmonics of stride frequency
        amps = self.harmonic_amplitudes(self.trunk_vertical, 10)
        d["harmonic_ratio_v"] = float(amps[1::2].sum() / (amps[0::2].sum() + 1e-8))

        # Spectral entropy of the heel-separation signal (0..1)
        sep = self.heel_fwd_l - self.heel_fwd_r
        p = np.abs(np.fft.rfft(sep - sep.mean())) ** 2
        freqs = np.fft.rfftfreq(len(sep), 1.0 / self.fps)
        p = p[(freqs > 0.2) & (freqs < 6.0)]
        if p.sum() > 0 and len(p) > 1:
            p = p / p.sum()
            d["spectral_entropy"] = float(
                -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
            )

        # Left-right phase synchrony (ideal: anti-phase, offset pi)
        dphi = self.phase_l - self.phase_r
        z = np.mean(np.exp(1j * dphi))
        d["phase_locking"] = float(np.abs(z))
        d["phase_offset"] = float(1.0 - abs(np.angle(z)) / np.pi)
        return d

    def rhythm_vector(self) -> np.ndarray:
        d = self.rhythm_dict()
        return np.nan_to_num(np.array([d[k] for k in self.RHYTHM_NAMES]))

    # ---------------- spatial ----------------

    def stride_dict(self, leg_len: float) -> dict:
        d = dict(_STRIDE_DEFAULTS)
        sep = self.heel_fwd_l - self.heel_fwd_r
        if len(self.hs_l):
            d["step_length_l"] = float(np.median(sep[self.hs_l]) / leg_len)
        if len(self.hs_r):
            d["step_length_r"] = float(np.median(-sep[self.hs_r]) / leg_len)
        d["stride_length"] = d["step_length_l"] + d["step_length_r"]
        mean_step = 0.5 * (d["step_length_l"] + d["step_length_r"])
        d["step_length_asym"] = float(
            abs(d["step_length_l"] - d["step_length_r"]) / (abs(mean_step) + 1e-8)
        )
        events = np.concatenate([self.hs_l, self.hs_r]).astype(int)
        if len(events):
            d["step_width"] = float(np.median(np.abs(self.heel_lat[events])) / leg_len)

        def _stance(hs, to):
            fr = []
            for a, c in zip(hs[:-1], hs[1:]):
                b = to[(to > a) & (to < c)]
                if len(b):
                    fr.append((self.t[b[0]] - self.t[a]) / (self.t[c] - self.t[a]))
            return float(np.median(fr)) if fr else np.nan

        sl, sr = _stance(self.hs_l, self.to_l), _stance(self.hs_r, self.to_r)
        if np.isfinite(sl):
            d["stance_frac_l"] = sl
        if np.isfinite(sr):
            d["stance_frac_r"] = sr

        def _ds(hs_a, to_b):
            out = []
            for a in hs_a:
                b = to_b[to_b > a]
                if len(b) and self.t[b[0]] - self.t[a] < 0.5 * self.stride_period:
                    out.append((self.t[b[0]] - self.t[a]) / self.stride_period)
            return float(np.median(out)) if out else np.nan

        ds = [_ds(self.hs_l, self.to_r), _ds(self.hs_r, self.to_l)]
        if all(np.isfinite(ds)):
            d["double_support_frac"] = float(sum(ds))
        d["clearance_l"] = float(np.percentile(self.toe_height_l, 95) / leg_len)
        d["clearance_r"] = float(np.percentile(self.toe_height_r, 95) / leg_len)
        return d

    def stride_vector(self, leg_len: float) -> np.ndarray:
        d = self.stride_dict(leg_len)
        return np.nan_to_num(np.array([d[k] for k in self.STRIDE_NAMES]))


_MEMO: dict = {}


def analyse_gait(
    pose: np.ndarray,
    t: np.ndarray,
    fps: float,
    min_period: float = 0.75,
    max_period: float = 2.2,
) -> GaitCycleAnalysis:
    """Full gait-cycle analysis of one clip. `pose` is (N, 33, 3) isotropic
    canonical keypoints at the native frame rate; `t` is seconds.

    Memoised on the identity of `pose` (several feature families call this on
    the same clip); the cache keeps a reference so ids are never recycled.
    """
    key = (id(pose), id(t))
    hit = _MEMO.get("last")
    if hit is not None and hit[0] == key and hit[1] is pose and hit[2] is t:
        return hit[3]
    res = _analyse(pose, t, fps, min_period, max_period)
    _MEMO["last"] = (key, pose, t, res)
    return res


def _analyse(pose, t, fps, min_period, max_period) -> GaitCycleAnalysis:
    xy = lowpass(pose[:, :, :2], fps, 5.0)
    pelvis = (xy[:, L_HIP] + xy[:, R_HIP]) / 2

    # 1. walking axis
    d = xy[:, L_HEEL] - xy[:, R_HEEL]
    d = d - d.mean(axis=0)
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    axis = vt[0]
    foot_dir = np.concatenate(
        [xy[:, L_TOE] - xy[:, L_HEEL], xy[:, R_TOE] - xy[:, R_HEEL]]
    ).mean(axis=0)
    if foot_dir @ axis < 0:
        axis = -axis
    lat = np.array([-axis[1], axis[0]])

    def fwd(j):
        return (xy[:, j] - pelvis) @ axis

    heel_l, heel_r, toe_l, toe_r = fwd(L_HEEL), fwd(R_HEEL), fwd(L_TOE), fwd(R_TOE)
    heel_lat = (xy[:, L_HEEL] - xy[:, R_HEEL]) @ lat

    # 2. stride period from the heel-separation autocorrelation
    sep = heel_l - heel_r
    ac = autocorrelation(sep)
    lo = int(min_period * fps)
    hi = min(int(max_period * fps), int(0.75 * len(sep)))
    if hi > lo + 2:
        lag = pick_period(ac, lo, hi)
        period, conf = lag / fps, float(ac[lag])
    else:
        period, conf = 1.1, 0.0

    # 3. Zeni events
    w = int(round(0.3 * period * fps))
    hs_l, hs_r = local_extrema(heel_l, w), local_extrema(heel_r, w)
    to_l = local_extrema(toe_l, w, maxima=False)
    to_r = local_extrema(toe_r, w, maxima=False)

    # vertical trunk signal and toe heights (image y grows downward)
    hip_y = pelvis[:, 1]
    trunk_v = hip_y - np.polyval(np.polyfit(t, hip_y, 2), t)
    ground_l = np.maximum(xy[:, R_HEEL, 1], xy[:, R_TOE, 1])
    ground_r = np.maximum(xy[:, L_HEEL, 1], xy[:, L_TOE, 1])
    toe_h_l = np.clip(ground_l - xy[:, L_TOE, 1], 0, None)
    toe_h_r = np.clip(ground_r - xy[:, R_TOE, 1], 0, None)

    # per-frame phase: Hilbert phase of each heel's forward signal around f
    f = 1.0 / period
    ph_l = np.angle(analytic_signal(bandpass(heel_l, fps, 0.5 * f, 1.5 * f)))
    ph_r = np.angle(analytic_signal(bandpass(heel_r, fps, 0.5 * f, 1.5 * f)))

    return GaitCycleAnalysis(
        fps=fps,
        t=t,
        axis=axis,
        heel_fwd_l=heel_l,
        heel_fwd_r=heel_r,
        toe_fwd_l=toe_l,
        toe_fwd_r=toe_r,
        heel_lat=heel_lat,
        trunk_vertical=trunk_v,
        toe_height_l=toe_h_l,
        toe_height_r=toe_h_r,
        stride_period=period,
        period_confidence=conf,
        hs_l=hs_l,
        hs_r=hs_r,
        to_l=to_l,
        to_r=to_r,
        phase_l=ph_l,
        phase_r=ph_r,
    )


# ============================================================
# Cycle normalisation
# ============================================================


def cycle_normalise(
    signal: np.ndarray, t: np.ndarray, events: np.ndarray, points: int = 60
) -> Optional[np.ndarray]:
    """Resample every event-to-event cycle of `signal` (N, D) to `points`
    samples of %-gait-cycle. Returns (n_cycles, points, D) or None."""
    events = np.asarray(events, dtype=int)
    if len(events) < 2:
        return None
    cycles = []
    for a, b in zip(events[:-1], events[1:]):
        if b - a < 4:
            continue
        grid = np.linspace(t[a], t[b], points)
        seg_t, seg = t[a : b + 1], signal[a : b + 1]
        cycles.append(
            np.stack(
                [np.interp(grid, seg_t, seg[:, j]) for j in range(seg.shape[1])],
                axis=1,
            )
        )
    return np.array(cycles) if cycles else None


# ============================================================
# Dynamic Time Warping
# ============================================================


def dtw(
    a: np.ndarray, b: np.ndarray, band: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """DTW between (n, D) and (m, D) sequences with an optional Sakoe-Chiba
    band. Returns (path-length-normalised distance, path as (k, 2) indices)."""
    a = np.atleast_2d(a.T).T if a.ndim == 1 else a
    b = np.atleast_2d(b.T).T if b.ndim == 1 else b
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    if band is not None:
        band = max(band, abs(n - m))
        ii, jj = np.indices((n, m))
        cost = np.where(np.abs(ii * m / n - jj) <= band, cost, np.inf)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row, prev = acc[i], acc[i - 1]
        ci = cost[i - 1]
        for j in range(1, m + 1):
            c = ci[j - 1]
            if c == np.inf:
                continue
            row[j] = c + min(prev[j], row[j - 1], prev[j - 1])
    # backtrack
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path = np.array(path[::-1])
    return float(acc[n, m] / len(path)), path


def dtw_warp(query: np.ndarray, reference: np.ndarray, band: int = 10) -> np.ndarray:
    """Warp `query` onto the time axis of `reference` (both (T, D)): each
    reference frame receives the mean of the query frames aligned to it."""
    _, path = dtw(query, reference, band)
    out = np.zeros_like(reference, dtype=float)
    count = np.zeros(len(reference))
    for qi, ri in path:
        out[ri] += query[qi]
        count[ri] += 1
    return out / np.maximum(count, 1)[:, None]
