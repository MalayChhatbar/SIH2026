"""
idr_core.py - DISHA / IDR v2 core library (SIH 26168, AI-ML Intelligent Dead Reckoning)

Pure NumPy/SciPy (no deep-learning dependency) so the SAME engine runs on a phone
(Chaquopy / NDK port), on an edge box with a FOG IMU @ 200 Hz, and in the notebook.

What changed w.r.t. v1 (see notebook section 0):
  * every calibration routine accepts t_max  -> strictly CAUSAL (no peeking into blackouts)
  * grid_channels(): phone channels are mapped onto the 10 Hz grid once, not per call
  * EKF6.update_heading_innov(): delayed + adaptive GNSS-course aiding (the course of two
    fixes is the heading at the MID-POINT of the interval, not "now")
  * RoadMap / OnlineMapMatcher: routing-aware HMM on the road graph; its output is fed BACK
    into the EKF as a cross-track + road-heading measurement (EKF6.update_map)
  * run_ekf(): causal initialisation, slice support (i_start/i_end), map hook, per-step AI speed
"""
import heapq
import itertools
import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

S_COLS = ["gps_lat", "gps_lon", "gps_alt", "gps_speed", "gps_acc_m", "gps_orient",
          "gps_sats", "t_ms", "date", "ax", "ay", "az", "gx", "gy", "gz",
          "gy_yaw", "gy_pitch", "gy_roll", "mx", "my", "mz", "o_yaw", "o_pitch", "o_roll"]
V_COLS = ["sats", "tod_s", "lat", "lon", "v_kmh", "heading", "height_km", "vvel_kmh",
          "dt", "steer", "wfl", "wfr", "wrl", "wrr", "yaw_rate", "ind_speed",
          "ind_long_g", "ind_lat_g", "handbrake", "gear_req", "gear", "rpm",
          "coolant", "clutch", "brake_psi", "brake_pos", "batt", "air_temp", "pedal"]


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =========================================================================== parsers
def parse_s(path):
    """Parse a smartphone ('S-') IO-VNBD CSV.

    Returns dict: t (N,) phone clock [s], acc/gyr/mag/grav (N,3), fix{i,t,lat,lon,spd,acc}
    (only rows where the GNSS actually updated - AndroSensor holds values between fixes).
    """
    s = pd.read_csv(path, skiprows=1, header=None, names=S_COLS, encoding="latin-1")
    raw = s["date"].astype(str).str.strip()
    dts = pd.to_datetime(raw.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ms = pd.to_numeric(raw.str.slice(20), errors="coerce").fillna(0)
    t = (dts.astype("datetime64[ns]").astype("int64") / 1e9 + ms / 1e3).to_numpy(float)
    lat = s.gps_lat.to_numpy(float)
    lon = s.gps_lon.to_numpy(float)
    spd = s.gps_speed.to_numpy(float)          # labelled 'Kmh' but is m/s (verified on the dataset)
    new = np.zeros(len(s), dtype=bool)
    new[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1]) | (spd[1:] != spd[:-1])
    new[0] = True
    fi = np.where(new)[0]
    return dict(
        t=t,
        acc=s[["ax", "ay", "az"]].to_numpy(float),
        gyr=s[["gy_yaw", "gy_pitch", "gy_roll"]].to_numpy(float),
        mag=s[["mx", "my", "mz"]].to_numpy(float),
        grav=s[["gx", "gy", "gz"]].to_numpy(float),
        fix=dict(i=fi, t=t[fi], lat=lat[fi], lon=lon[fi],
                 spd=spd[fi], acc=s.gps_acc_m.to_numpy(float)[fi]),
    )


def parse_v(path):
    """Parse a vehicle ('V-') IO-VNBD CSV (Racelogic VBOX @ 10 Hz): t, lat, lon, v (m/s), head (rad)."""
    v = pd.read_csv(path, skiprows=1, header=None, names=V_COLS, encoding="latin-1")
    return dict(
        t=v.tod_s.to_numpy(float),
        lat=v.lat.to_numpy(float),
        lon=v.lon.to_numpy(float),
        v=v.v_kmh.to_numpy(float) / 3.6,
        head=np.unwrap(np.radians(v.heading.to_numpy(float))),
    )


def lla_to_enu(lat, lon, lat0, lon0):
    """Lat/lon -> local ENU tangent plane [m] (spherical approximation, fine for <50 km)."""
    R = 6378137.0
    x = R * np.radians(np.asarray(lon) - lon0) * np.cos(np.radians(lat0))
    y = R * np.radians(np.asarray(lat) - lat0)
    return np.column_stack([x, y])


def enu_to_lla(xy, lat0, lon0):
    R = 6378137.0
    lat = lat0 + np.degrees(xy[:, 1] / R)
    lon = lon0 + np.degrees(xy[:, 0] / (R * np.cos(np.radians(lat0))))
    return lat, lon


# =========================================================================== time sync
def autosync(S, V, coarse_span=120.0, wide_span=40.0, verbose=False):
    """Constant clock offset between the phone and VBOX streams (dataset preparation step).

    Returns delta such that  V_rel_time(fix) = (fix_t - S.t[0]) + delta.
    Stage 1: 1 Hz speed-profile cross-correlation.  Stage 2: median fix-position error search
    (+/- wide_span).  Stage 3: 0.1 s polish.  No timezone assumptions.
    """
    from scipy.interpolate import interp1d
    s_rel = S["fix"]["t"] - S["t"][0]
    v_rel = V["t"] - V["t"][0]
    n1 = int(max(s_rel[-1], v_rel[-1])) + 1

    def bin1(rel_t, x):
        idx = np.clip(rel_t.astype(int), 0, n1 - 1)
        out = np.full(n1, np.nan)
        ser = pd.Series(x).groupby(idx).mean()
        out[ser.index] = ser.to_numpy()
        return pd.Series(out).ffill().fillna(0).to_numpy()

    bs, bv = bin1(s_rel, S["fix"]["spd"]), bin1(v_rel, V["v"])
    best_c, best_coarse = -2.0, 0
    for d in range(-int(coarse_span), int(coarse_span) + 1):
        if d >= 0:
            a2, b2 = bs[d:], bv[: n1 - d]
        else:
            a2, b2 = bs[: n1 + d], bv[-d:]
        if len(a2) < 60:
            continue
        c = np.corrcoef(a2, b2)[0, 1]
        if c > best_c:
            best_c, best_coarse = c, d
    fx = interp1d(v_rel, V["lat"], bounds_error=False)
    fy = interp1d(v_rel, V["lon"], bounds_error=False)
    R = 6371000.0

    def err(delta):
        tt = s_rel + delta
        dp = np.radians(fx(tt) - S["fix"]["lat"])
        dl = np.radians(fy(tt) - S["fix"]["lon"]) * np.cos(np.radians(S["fix"]["lat"]))
        return np.nanmedian(R * np.hypot(dp, dl))

    wide = best_coarse + np.arange(-wide_span, wide_span + 0.05, 0.5)
    errs = [err(d) for d in wide]
    d0 = wide[int(np.argmin(errs))]
    fine = d0 + np.arange(-2.0, 2.05, 0.1)
    errs = [err(d) for d in fine]
    k = int(np.argmin(errs))
    if verbose:
        print(f"[autosync] coarse={best_coarse} s corr={best_c:.3f} -> wide={d0:+.2f} "
              f"-> delta={fine[k]:+.2f} s  median fix err={errs[k]:.2f} m")
    return fine[k], errs[k]


def map_channels_to_v(S, V, delta, key):
    """Sample-and-hold map of a phone channel group (N,3) onto the VBOX 10 Hz grid."""
    sh = (S["t"] - S["t"][0]) + delta
    tv = V["t"] - V["t"][0]
    idx = np.searchsorted(tv, sh)
    ok = (idx > 0) & (idx < len(tv))
    ii = np.minimum(idx[ok] - 1, len(tv) - 1)
    out = np.full((len(tv), S[key].shape[1]), np.nan)
    out[ii] = S[key][ok]
    return pd.DataFrame(out).ffill().bfill().to_numpy()


def grid_channels(S, V, delta):
    """All phone channels on the 10 Hz master grid, computed ONCE per journey."""
    G = {k: map_channels_to_v(S, V, delta, k) for k in ("acc", "grav", "gyr", "mag")}
    G["lin"] = G["acc"] - G["grav"]
    return G


def calib_horizon(V, calib_s=900.0, calib_frac=0.4, min_s=240.0):
    """Length (V-relative seconds) of the GNSS-aided warm-up used for calibration.

    min(calib_s, calib_frac * duration), at least min_s. After this horizon every
    calibration constant is FROZEN - exactly what the phone does after its warm-up drive.
    """
    dur = V["t"][-1] - V["t"][0]
    return float(min(max(min(calib_s, calib_frac * dur), min_s), 0.6 * dur))


# =========================================================================== alignment engine
def course_from_fixes(fix_t, fix_xy, min_disp=15.0):
    """Course-over-ground (CW-from-north, rad) between successive fixes + validity + distance."""
    d = np.linalg.norm(np.diff(fix_xy, axis=0), axis=1)
    brg = np.arctan2(np.diff(fix_xy[:, 0]), np.diff(fix_xy[:, 1]))  # atan2(dE, dN)
    return brg, d >= min_disp, d


def _fix_series(S, V, delta, t_max):
    fix_t = (S["fix"]["t"] - S["t"][0]) + delta
    fix_xy = lla_to_enu(S["fix"]["lat"], S["fix"]["lon"], V["lat"][0], V["lon"][0])
    spd = S["fix"]["spd"]
    if t_max is not None:
        nf = int(np.searchsorted(fix_t, t_max))
        return fix_t[:nf], fix_xy[:nf], spd[:nf]
    return fix_t, fix_xy, spd


def fit_heading_mixer_gnss(S, V, delta, min_disp=25.0, t_max=None, G=None):
    """Phone->vehicle heading-rate mapping from the phone's OWN GNSS only (causal if t_max set).

    w_vert = c0 + c1*g1 + c2*g2 + c3*g3, any mount (axis permutation / rotation / sign flip all
    become coefficients). Unit-norm re-normalisation, ZUPT-style constant term, and a gyro
    scale-factor calibration against the GNSS course change.
    """
    G = G or grid_channels(S, V, delta)
    g = G["gyr"]
    tv = V["t"] - V["t"][0]
    ig = len(tv) if t_max is None else int(np.searchsorted(tv, t_max))
    fix_t, fix_xy, _ = _fix_series(S, V, delta, t_max)
    rows, targets = None, None
    for md, need in ((min_disp, 30), (15.0, 15)):
        brg, valid, d = course_from_fixes(fix_t, fix_xy, md)
        r_, t_ = [], []
        for k in range(1, len(brg) - 1):
            if not (valid[k - 1] and valid[k]):
                continue
            m0 = 0.5 * (fix_t[k - 1] + fix_t[k])
            m1 = 0.5 * (fix_t[k] + fix_t[k + 1])
            if m1 <= m0 + 1.0:
                continue
            i0, i1 = np.searchsorted(tv, m0), np.searchsorted(tv, m1)
            if i1 - i0 < 10:
                continue
            dpsi = wrap_pi(brg[k] - brg[k - 1])
            r_.append(np.r_[1.0, g[i0:i1].mean(axis=0)])
            t_.append(dpsi / (m1 - m0))
        rows, targets = np.asarray(r_), np.asarray(t_)
        if len(rows) >= need:
            break
    if len(rows) < 15:
        raise RuntimeError(f"not enough course pairs for GNSS mixer calibration ({len(rows)})")
    # ---- gravity prior: the yaw-rate axis is the (unit) gravity direction, up to the dataset's
    # axis labelling. Search all 48 signed axis permutations, keep the best-correlating one.
    ghat = G["grav"][:ig].mean(axis=0)
    ghat = ghat / (np.linalg.norm(ghat) + 1e-9)
    u0, best_c = None, 0.0
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1.0, -1.0), repeat=3):
            u = np.asarray(sg) * ghat[list(perm)]
            p = rows[:, 1:] @ u
            if np.std(p) < 1e-9:
                continue
            c = float(np.corrcoef(p, targets)[0, 1])
            if c > best_c:
                best_c, u0 = c, u
    use_prior = u0 is not None and best_c > 0.6

    def solve(Rw, Tw, lam):
        D = Rw
        if use_prior and lam > 0:
            reg = np.zeros((3, 4)); reg[:, 1:] = np.eye(3) * math.sqrt(lam)
            D = np.vstack([Rw, reg])
            T = np.r_[Tw, math.sqrt(lam) * u0]
        else:
            T = Tw
        return np.linalg.lstsq(D, T, rcond=None)[0]

    # scale of the ridge: lam is expressed relative to the data Gram diagonal
    gram = float(np.mean(np.sum(rows[:, 1:] ** 2, axis=0)) * len(rows))
    lam_grid = [0.0] if not use_prior else [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_l, best_cv = lam_grid[0], np.inf
    if len(lam_grid) > 1 and len(rows) >= 20:
        folds = np.arange(len(rows)) % 5
        for lm in lam_grid:
            e = 0.0
            for fo in range(5):
                tr, te = folds != fo, folds == fo
                cf = solve(rows[tr], targets[tr], lm * gram)
                e += float(np.sum((rows[te] @ cf - targets[te]) ** 2))
            if e < best_cv:
                best_cv, best_l = e, lm
    coef = solve(rows, targets, best_l * gram)
    r = rows @ coef - targets
    keep = np.abs(r) < 3 * np.std(r) + 1e-9
    if keep.sum() >= 15:
        coef = solve(rows[keep], targets[keep], best_l * gram)
    pred = rows @ coef
    anti = (np.sign(pred) != np.sign(targets)) & (np.abs(targets) > 0.02)
    keep2 = keep & ~anti
    if keep2.sum() >= 15 and anti.sum() > 0:
        coef = solve(rows[keep2], targets[keep2], best_l * gram)
        keep = keep2
    nrm = np.linalg.norm(coef[1:])
    if nrm > 1e-6:                      # orthonormal phone axes -> unit vertical vector
        coef = coef.copy()
        coef[1:] = coef[1:] / nrm
    lin_mag = np.linalg.norm(G["lin"], axis=1)
    still = lin_mag < 0.15
    still[ig:] = False                  # causal: stationary evidence only from the warm-up
    if still.sum() > 50:
        coef = coef.copy()
        coef[0] = -float(np.median((g @ coef[1:])[still]))
    w_vert = np.column_stack([np.ones(len(g)), g]) @ coef
    # gyro scale factor from integrated heading vs GNSS course change (on-device version of
    # factory scale-factor calibration)
    gm_, ta_ = [], []
    dt_v = float(np.median(np.diff(V["t"])))
    brg, valid, d = course_from_fixes(fix_t, fix_xy, min_disp)
    for k in range(1, len(brg) - 1):
        if not (valid[k - 1] and valid[k]):
            continue
        m0 = 0.5 * (fix_t[k - 1] + fix_t[k])
        m1 = 0.5 * (fix_t[k] + fix_t[k + 1])
        if m1 <= m0 + 1.0:
            continue
        i0, i1 = np.searchsorted(tv, m0), np.searchsorted(tv, m1)
        if i1 - i0 < 10:
            continue
        dc = wrap_pi(brg[k] - brg[k - 1])
        dm = w_vert[i0:i1].sum() * dt_v
        if abs(dm) > 0.05 and abs(dc) > 0.05:
            gm_.append(dm)
            ta_.append(dc)
    if len(gm_) >= 20:
        scale = float(np.clip(np.median(np.array(ta_) / np.array(gm_)), 0.6, 2.0))
        coef[1:] = coef[1:] * scale
        if still.sum() > 50:
            coef[0] = -float(np.median((g @ coef[1:])[still]))
        w_vert = np.column_stack([np.ones(len(g)), g]) @ coef
    else:
        scale = 1.0
    r_fin = rows[keep] @ coef - targets[keep]
    sigma_deg_s = float(np.std(r_fin) * 57.2958) if len(r_fin) > 5 else 8.0
    return dict(coef=coef, n_eq=int(len(rows)), n_keep=int(keep.sum()), w_vert=w_vert,
                dt=dt_v, sigma_deg_s=sigma_deg_s, scale=scale, t_max=t_max,
                prior_corr=float(best_c), prior_used=bool(use_prior), ridge=float(best_l))


def fit_accel_mixer_gnss(S, V, delta, min_speed=3.0, t_max=None, G=None):
    """Forward-accelerometer calibration from the phone's own GNSS speed differences (causal)."""
    G = G or grid_channels(S, V, delta)
    lin = G["lin"]
    tv = V["t"] - V["t"][0]
    fix_t, _, fspd = _fix_series(S, V, delta, t_max)
    rows, targets = [], []
    for k in range(1, len(fix_t)):
        dts = fix_t[k] - fix_t[k - 1]
        if not (2.0 < dts < 20.0) or min(fspd[k], fspd[k - 1]) < min_speed:
            continue
        i0, i1 = np.searchsorted(tv, fix_t[k - 1]), np.searchsorted(tv, fix_t[k])
        if i1 - i0 < 10:
            continue
        rows.append(np.r_[1.0, lin[i0:i1].mean(axis=0)])
        targets.append((fspd[k] - fspd[k - 1]) / dts)
    rows, targets = np.asarray(rows), np.asarray(targets)
    if len(rows) < 15:
        raise RuntimeError(f"not enough GNSS speed pairs for accel calibration ({len(rows)})")
    coef, *_ = np.linalg.lstsq(rows, targets, rcond=None)
    r = rows @ coef - targets
    keep = np.abs(r) < 3 * np.std(r) + 1e-9
    if keep.sum() >= 15:
        coef, *_ = np.linalg.lstsq(rows[keep], targets[keep], rcond=None)
        r = rows[keep] @ coef - targets[keep]
    # NO unit-norm renormalisation: forward accel (<=0.4 m/s2) is buried in vibration (~1.2 m/s2),
    # so the shrunken LS vector is the right amount of smoothing.
    tk = targets[keep] if keep.sum() >= 15 else targets
    ss = 1 - ((r ** 2).sum() / (((tk - tk.mean()) ** 2).sum() + 1e-9))
    a_fwd = np.column_stack([np.ones(len(lin)), lin]) @ coef
    return dict(coef=coef, a_fwd=a_fwd, r2=float(ss), n_eq=int(len(rows)),
                dt=float(np.median(np.diff(V["t"]))), t_max=t_max)


# =========================================================================== EKF
class EKF6:
    """Loosely-coupled GNSS+INS fusion for the 2-D bicycle model (non-holonomic by construction).

    State x = [E, N, psi (CW from north), v, b_g (gyro bias), b_a (fwd-accel bias)]
    Inputs : w (calibrated heading rate), a_fwd (calibrated forward acceleration)
    Aiding : GNSS position/speed/course, ZUPT, AI speed, road-map cross-track + road heading.
    """

    def __init__(self, x0, P0_diag, sig_pos=4.0, q=None):
        self.x = np.asarray(x0, float).copy()
        self.P = np.diag(np.asarray(P0_diag, float)).copy()
        self.sig_pos0 = sig_pos
        self.q = q or dict(pos=0.20, psi=0.03, v=0.35, b_g=5e-5, b_a=0.002)
        self.reject_streak = 0
        self.n_rej = 0
        self.n_acc = 0
        self.n_map = 0
        self.n_reanchor = 0
        self._last_rej = None
        self._hm = 0
        self.last_fix_t = -1e9

    def predict(self, w, a_fwd, dt, zupt=False):
        x, P, psi = self.x, self.P, self.x[2]
        frozen = zupt and x[3] < 2.0
        nx = x.copy()
        nx[0] += x[3] * math.sin(psi) * dt
        nx[1] += x[3] * math.cos(psi) * dt
        nx[2] += (0.0 if frozen else w) * dt - x[4] * dt
        if not frozen:
            nx[3] += (a_fwd - x[5]) * dt
        self.x = nx
        A = np.eye(6)
        A[0, 2] = x[3] * math.cos(psi) * dt
        A[0, 3] = math.sin(psi) * dt
        A[1, 2] = -x[3] * math.sin(psi) * dt
        A[1, 3] = math.cos(psi) * dt
        A[2, 4] = -dt
        A[3, 5] = -dt
        self.P = A @ P @ A.T
        q = self.q
        self.P[0, 0] += q["pos"] ** 2 * dt
        self.P[1, 1] += q["pos"] ** 2 * dt
        self.P[2, 2] += q["psi"] ** 2 * dt
        self.P[3, 3] += q["v"] ** 2 * dt
        self.P[4, 4] += q["b_g"] ** 2 * dt
        self.P[5, 5] += q["b_a"] ** 2 * dt
        if frozen:                       # evidence-gated ZUPT (self-rejects at driving speed)
            self._zupt(w, a_fwd)
        return self.x

    def _zupt(self, w, a_fwd):
        H = np.zeros((1, 6)); H[0, 3] = 1.0
        self._update(H, np.array([0.0 - self.x[3]]), np.array([[0.05 ** 2]]), gate_chi2=9.0)
        H = np.zeros((1, 6)); H[0, 4] = 1.0
        self._update(H, np.array([w - self.x[4]]), np.array([[0.004 ** 2]]), gate_chi2=9.0)
        H = np.zeros((1, 6)); H[0, 5] = 1.0
        self._update(H, np.array([a_fwd - self.x[5]]), np.array([[0.03 ** 2]]), gate_chi2=9.0)

    def _stabilize_covariance(self):
        """Keep P finite, symmetric, and positive-semidefinite after long runs."""
        P = 0.5 * (self.P + self.P.T)
        if not np.isfinite(P).all():
            raise FloatingPointError("EKF covariance contains NaN/Inf")

        # Joseph form should preserve PSD mathematically, but finite precision can create
        # tiny negative eigenvalues after many 10 Hz updates. Repair only when necessary.
        mine = float(np.linalg.eigvalsh(P)[0])
        if mine < 0.0:
            w, V = np.linalg.eigh(P)
            floor = max(1e-12, float(np.max(w)) * 1e-12)
            w = np.maximum(w, floor)
            P = (V * w) @ V.T
            P = 0.5 * (P + P.T)
        self.P = P

    @staticmethod
    def _solve_spd(S, b):
        """Solve an innovation system with a small adaptive diagonal jitter if needed."""
        S = 0.5 * (np.asarray(S, float) + np.asarray(S, float).T)
        b = np.asarray(b, float)
        if not np.isfinite(S).all() or not np.isfinite(b).all():
            raise FloatingPointError("EKF innovation contains NaN/Inf")
        eye = np.eye(S.shape[0])
        scale = max(1.0, float(np.max(np.abs(np.diag(S)))))
        jitter = 1e-12 * scale
        for _ in range(7):
            try:
                return np.linalg.solve(S + jitter * eye, b)
            except np.linalg.LinAlgError:
                jitter *= 10.0
        # Last-resort fallback. This should only be reached for badly conditioned data.
        return np.linalg.pinv(S, rcond=1e-12) @ b

    def _update(self, H, r, R, gate_chi2=None):
        """Joseph-form update with covariance/innovation safeguards; returns accepted bool."""
        H = np.asarray(H, float)
        r = np.asarray(r, float)
        R = np.asarray(R, float)
        self._stabilize_covariance()
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)

        sr = self._solve_spd(S, r)
        if gate_chi2 is not None and float(r @ sr) > gate_chi2:
            return False

        PHt = self.P @ H.T
        K = self._solve_spd(S, PHt.T).T
        self.x = self.x + K @ r
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self._stabilize_covariance()
        return True

    # ---------------- GNSS ----------------
    def update_pos(self, z_xy, t=None, sigma=None, gate=11.83):
        if sigma is None:
            sigma = self.sig_pos0
        H = np.zeros((2, 6)); H[0, 0] = 1; H[1, 1] = 1
        r = z_xy - H @ self.x
        self._stabilize_covariance()
        S = H @ self.P @ H.T + (sigma ** 2) * np.eye(2)
        if float(r @ self._solve_spd(S, r)) > gate:        # chi2(2 dof) 99.7 %
            self.n_rej += 1
            self.reject_streak += 1
            # Two consecutive rejected fixes that agree with EACH OTHER (distance consistent with the
            # vehicle's speed) mean the filter is wrong, not the fixes -> re-anchor immediately.
            lr = self._last_rej
            self._last_rej = (t, np.array(z_xy, float))
            if lr is not None and t is not None and lr[0] is not None and 0.5 < t - lr[0] < 30.0:
                span = float(np.hypot(*(np.asarray(z_xy) - lr[1])))
                if span <= max(35.0, 1.6 * max(self.x[3], 3.0) * (t - lr[0])):
                    self.x[0], self.x[1] = z_xy
                    self.P[0:2, :] = 0.0
                    self.P[:, 0:2] = 0.0
                    self.P[0, 0] = (sigma * 1.5) ** 2
                    self.P[1, 1] = (sigma * 1.5) ** 2
                    self._inflate(2, 1.5)
                    self.reject_streak = 0
                    self._last_rej = None
                    self.n_reanchor += 1
                    self.last_fix_t = t
                    return False
            if self.reject_streak >= 4:                    # soft reset: inflate std (keeps P PSD)
                self._inflate(0, 2.0)
                self._inflate(1, 2.0)
                self._inflate(2, 1.22)
                self.reject_streak = 0
            if self.n_rej % 8 == 0:                        # hard re-anchor onto the fix
                self.x[0], self.x[1] = z_xy
                self.P[0:2, :] = 0.0
                self.P[:, 0:2] = 0.0
                self.P[0, 0] = (sigma * 1.5) ** 2
                self.P[1, 1] = (sigma * 1.5) ** 2
            return False
        self._update(H, r, (sigma ** 2) * np.eye(2))
        self.n_acc += 1
        self.reject_streak = 0
        self._last_rej = None
        if t is not None:
            self.last_fix_t = t
        return True

    def _inflate(self, i, f):
        """Scale the standard deviation of state i by f (D P D with D=diag(1..f..1): stays PSD)."""
        self.P[i, :] *= f
        self.P[:, i] *= f

    def update_speed(self, v_meas, sigma=1.0):
        H = np.zeros((1, 6)); H[0, 3] = 1.0
        return self._update(H, np.array([v_meas - self.x[3]]), np.array([[sigma ** 2]]))

    def update_course(self, psi_meas, sigma=15.0, gate_rad=1.0):
        """v1 behaviour (kept as the 'legacy' baseline): course treated as heading NOW, sigma 15 deg."""
        H = np.zeros((1, 6)); H[0, 2] = 1.0
        r = wrap_pi(psi_meas - self.x[2])
        if abs(r) > gate_rad:
            return False
        return self._update(H, np.array([r]), np.array([[np.radians(sigma) ** 2]]))

    def update_heading_innov(self, r, sigma_rad, gate_rad=0.7):
        """Heading update from an innovation computed elsewhere (delayed course aiding, road bearing)."""
        if abs(r) > gate_rad:
            return False
        H = np.zeros((1, 6)); H[0, 2] = 1.0
        return self._update(H, np.array([r]), np.array([[sigma_rad ** 2]]), gate_chi2=9.0)

    # ---------------- road map ----------------
    def update_map(self, snap_xy, tangent, sig_xt=6.0, sig_head_deg=10.0, gate_xt=40.0,
                   use_heading=True, min_speed=2.5):
        """Cross-track (+ road-bearing) measurement from the HMM-matched road segment.

        Only the component NORMAL to the road is corrected (along-track is left to the
        inertial/speed solution): that is the non-holonomic 'stay on the road' constraint.
        At every junction turn the new road's cross-track constraint pins the old along-track
        coordinate, which is how map matching also removes along-track drift.
        """
        t = np.asarray(tangent, float)
        t = t / (np.linalg.norm(t) + 1e-12)
        n = np.array([-t[1], t[0]])
        r_xt = float(n @ (np.asarray(snap_xy, float) - self.x[:2]))
        if abs(r_xt) > gate_xt:
            return False
        H = np.zeros((1, 6)); H[0, 0], H[0, 1] = n[0], n[1]
        ok = self._update(H, np.array([r_xt]), np.array([[sig_xt ** 2]]), gate_chi2=11.0)
        if ok:
            self.n_map += 1
            self._hm += 1
            if use_heading and self.x[3] > min_speed and self._hm % 4 == 1:     # road bearing every ~4 s
                beta = math.atan2(t[0], t[1])
                r = wrap_pi(beta - self.x[2])
                if abs(r) > math.pi / 2:
                    r = wrap_pi(beta + math.pi - self.x[2])
                self.update_heading_innov(r, math.radians(sig_head_deg), gate_rad=0.6)
        return ok

    @property
    def pos(self):
        return self.x[0], self.x[1]


# =========================================================================== road network
class RoadMap:
    """Undirected road graph in a local ENU frame with exact polyline projection + routing."""

    def __init__(self, edges, sample_m=5.0):
        """edges: iterable of (u, v, xy) - xy (n,2) polyline ENU metres, oriented u -> v."""
        self.eu, self.ev, self.exy, self.ecum, self.elen = [], [], [], [], []
        self.adj = {}
        pts, pe, ps = [], [], []
        for (u, v, xy) in edges:
            xy = np.asarray(xy, float)
            if len(xy) < 2:
                continue
            seg = np.diff(xy, axis=0)
            sl = np.hypot(seg[:, 0], seg[:, 1])
            if sl.sum() < 1e-3:
                continue
            e = len(self.eu)
            cum = np.r_[0.0, np.cumsum(sl)]
            self.eu.append(u); self.ev.append(v); self.exy.append(xy)
            self.ecum.append(cum); self.elen.append(float(cum[-1]))
            self.adj.setdefault(u, []).append((v, float(cum[-1])))
            self.adj.setdefault(v, []).append((u, float(cum[-1])))
            for i in range(len(seg)):
                n = max(int(sl[i] / sample_m) + 1, 1)
                t = (np.arange(n) / n)[:, None]
                pts.append(xy[i] + seg[i] * t)
                pe.append(np.full(n, e))
                ps.append(np.full(n, i))
            pts.append(xy[-1][None, :]); pe.append(np.array([e])); ps.append(np.array([len(seg) - 1]))
        if not pts:
            raise ValueError("RoadMap needs at least one edge")
        self.pts = np.concatenate(pts)
        self.pe = np.concatenate(pe).astype(int)
        self.ps = np.concatenate(ps).astype(int)
        self.tree = cKDTree(self.pts)
        self._sp = {}

    # ----- construction from OSMnx -----
    @classmethod
    def from_osmnx(cls, G, lat0, lon0, sample_m=5.0):
        """G: osmnx MultiDiGraph (unprojected, lat/lon). Two-way roads appear once."""
        seen, edges = set(), []
        for u, v, k, data in G.edges(keys=True, data=True):
            key = (min(u, v), max(u, v), round(float(data.get("length", 0.0)), 1))
            if key in seen:
                continue
            seen.add(key)
            geom = data.get("geometry")
            if geom is None:
                xs = [G.nodes[u]["x"], G.nodes[v]["x"]]
                ys = [G.nodes[u]["y"], G.nodes[v]["y"]]
            else:
                xs, ys = geom.xy
            xy = lla_to_enu(np.asarray(ys), np.asarray(xs), lat0, lon0)
            pu = lla_to_enu(np.array([G.nodes[u]["y"]]), np.array([G.nodes[u]["x"]]), lat0, lon0)[0]
            if np.hypot(*(xy[0] - pu)) > np.hypot(*(xy[-1] - pu)):   # orient u -> v
                xy = xy[::-1]
            edges.append((u, v, xy))
        return cls(edges, sample_m)

    # ----- geometry -----
    def _proj_seg(self, e, i, p):
        xy = self.exy[e]
        a, b = xy[i], xy[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float(np.clip(((p - a) @ ab) / L2, 0.0, 1.0))
        q = a + t * ab
        sl = math.sqrt(L2)
        tan = ab / sl if sl > 1e-9 else np.array([0.0, 1.0])
        return dict(e=e, off=float(self.ecum[e][i] + t * sl), xy=q, tan=tan,
                    dist=float(np.hypot(*(p - q))))

    def project(self, e, i, p):
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(self.exy[e]) - 1:
                c = self._proj_seg(e, j, p)
                if best is None or c["dist"] < best["dist"]:
                    best = c
        return best

    def candidates(self, p, radius, K=6):
        p = np.asarray(p, float)
        k = min(4 * K, len(self.pts))
        d, ii = self.tree.query(p, k=k, distance_upper_bound=radius)
        d, ii = np.atleast_1d(d), np.atleast_1d(ii)
        best = {}
        for dist, j in zip(d, ii):
            if not np.isfinite(dist) or j >= len(self.pts):
                break
            e = int(self.pe[j])
            if e in best:
                continue
            best[e] = self.project(e, int(self.ps[j]), p)
            if len(best) >= K:
                break
        return list(best.values())

    # ----- routing -----
    def _sp_from(self, node, cutoff):
        hit = self._sp.get(node)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist, heap = {node: 0.0}, [(0.0, node)]
        done = set()
        while heap:
            d, n = heapq.heappop(heap)
            if n in done:
                continue
            done.add(n)
            for m, w in self.adj.get(n, ()):
                nd = d + w
                if nd <= cutoff and nd < dist.get(m, 1e18):
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
        self._sp[node] = (cutoff, dist)
        return dist

    def route_dist(self, c1, c2, cutoff=1500.0):
        e1, e2 = c1["e"], c2["e"]
        if e1 == e2:
            return abs(c1["off"] - c2["off"])
        best = math.inf
        for n1, d1 in ((self.eu[e1], c1["off"]), (self.ev[e1], self.elen[e1] - c1["off"])):
            sp = self._sp_from(n1, cutoff)
            for n2, d2 in ((self.eu[e2], c2["off"]), (self.ev[e2], self.elen[e2] - c2["off"])):
                dd = sp.get(n2)
                if dd is not None:
                    best = min(best, d1 + dd + d2)
        return best


class OnlineMapMatcher:
    """Forward (filtering) HMM map matcher, Newson-Krumm style with ROUTE-distance transitions.

    emission  : Gaussian in the distance from the dead-reckoned point to each candidate road
                (sigma = the filter's own position uncertainty -> widens during a blackout)
    transition: exp(-| route_distance - great_circle_distance | / beta)
    Fully causal: step() only uses past observations, so it is valid for real-time use.
    """

    def __init__(self, rmap, K=6, beta0=6.0, beta_k=0.2, radius_min=35.0, max_route=1500.0):
        self.rm, self.K = rmap, K
        self.beta0, self.beta_k = beta0, beta_k
        self.radius_min, self.max_route = radius_min, max_route
        self.reset()

    def reset(self):
        self.prev = None
        self.logp = None
        self.prev_xy = None

    def step(self, xy, sigma, heading=None, sig_head=None):
        """One filtering step. heading (rad, CW from north) + sig_head (rad) add a heading-consistency
        term to the emission, so a road CROSSING the driven direction cannot win at a junction."""
        xy = np.asarray(xy, float)
        cands = self.rm.candidates(xy, radius=max(self.radius_min, 3.0 * sigma), K=self.K)
        if not cands:
            self.reset()
            return None
        em = np.array([-0.5 * (c["dist"] / sigma) ** 2 for c in cands])
        if heading is not None and sig_head:
            for k, c in enumerate(cands):
                beta = math.atan2(c["tan"][0], c["tan"][1])
                d = abs(wrap_pi(beta - heading))
                d = min(d, math.pi - d)                       # roads are undirected
                em[k] += -0.5 * (d / sig_head) ** 2
        if self.prev is None:
            lp = em
        else:
            gc = float(np.hypot(*(xy - self.prev_xy)))
            beta = self.beta0 + self.beta_k * gc
            T = np.full((len(self.prev), len(cands)), -30.0)
            cut = min(gc + 400.0, self.max_route)
            for i, pc in enumerate(self.prev):
                for j, c in enumerate(cands):
                    dr = self.rm.route_dist(pc, c, cutoff=self.max_route)
                    if math.isfinite(dr) and dr <= cut + 400.0:
                        T[i, j] = -abs(dr - gc) / beta
            tot = self.logp[:, None] + T
            m = tot.max(axis=0)
            lp = m + np.log(np.exp(tot - m).sum(axis=0)) + em
        lse = lp.max() + np.log(np.exp(lp - lp.max()).sum())
        lp = lp - lse
        j = int(np.argmax(lp))
        self.prev, self.logp, self.prev_xy = cands, lp, xy
        c = cands[j]
        return dict(xy=c["xy"], tan=c["tan"], dist=c["dist"], conf=float(np.exp(lp[j])), edge=c["e"])


# =========================================================================== fusion runner
DEFAULT_COURSE = dict(mode="delayed", min_disp=25.0, sig_floor_deg=2.0, curve_k=0.25,
                      max_dpsi_deg=60.0, gate_rad=0.7, legacy_sigma_deg=15.0)
DEFAULT_MAP = dict(every=10, after_s=3.0, sig_xt=12.0, sig_head_deg=10.0, conf_min=0.9,
                   gate_xt=30.0, use_heading=False, sig_obs_min=8.0, sig_obs_max=80.0,
                   heading_aware=True, head_sig_deg=30.0)


def _find_init(fix_t, fix_xy, vfix, avail, t_lo, min_init_disp):
    """Causal EKF initialisation: heading = course of the PAST fix pair (j-1, j), steady driving."""
    course, cvalid, d = course_from_fixes(fix_t, fix_xy, min_init_disp)
    for j in range(2, len(fix_t)):
        if fix_t[j] < t_lo or not (avail[j] and avail[j - 1] and avail[j - 2]):
            continue
        if (cvalid[j - 1] and cvalid[j - 2] and vfix[j] > 4.0 and d[j - 1] >= 40.0
                and abs(wrap_pi(course[j - 1] - course[j - 2])) < 0.15):
            return j, float(course[j - 1])
    for j in range(1, len(fix_t)):                       # relaxed fallback
        if fix_t[j] >= t_lo and avail[j] and avail[j - 1] and cvalid[j - 1]:
            return j, float(course[j - 1])
    raise RuntimeError("No valid course pair found for EKF init")


def run_ekf(V, w_vert, a_fwd, fix_t_rel, fix_xy, fix_spd, *, v_ai=None, sigma_ai=2.5,
            ai_every=5, outage_mask=None, sig_pos=4.0, sig_pos_arr=None, sig_fix_spd=1.0,
            min_init_disp=15.0, zupt=None, q=None, i_start=0, i_end=None,
            course_cfg=None, mapper=None, map_cfg=None, trace=None):
    """GNSS+INS fusion on the 10 Hz master grid, optionally on a slice [i_start, i_end).

    v_ai / sigma_ai : AI speed (N,) [NaN = none] and its sigma (scalar or (N,))
    zupt            : (N,) bool stationary flags from the AI ZUPT head
    outage_mask     : (F,) bool, True = fix unavailable (blackout)
    course_cfg      : dict(mode='delayed'|'legacy'|'off', ...)   see DEFAULT_COURSE
    mapper          : OnlineMapMatcher or None;  map_cfg see DEFAULT_MAP
    """
    n = len(V["t"]) if i_end is None else int(i_end)
    dt = float(np.median(np.diff(V["t"])))
    t0 = V["t"][0]
    tv = V["t"] - t0
    fix_t = np.asarray(fix_t_rel, float)
    fix_xy = np.asarray(fix_xy, float)
    fix_spd = np.asarray(fix_spd, float)
    nF = len(fix_t)
    outage_mask = np.zeros(nF, bool) if outage_mask is None else np.asarray(outage_mask, bool)
    avail = ~outage_mask
    cc = dict(DEFAULT_COURSE); cc.update(course_cfg or {})
    mc = dict(DEFAULT_MAP); mc.update(map_cfg or {})
    if sig_pos_arr is not None:
        sig_pos_arr = np.asarray(sig_pos_arr, float).ravel()
        if len(sig_pos_arr) != nF:
            raise ValueError("sig_pos_arr must have one value per fix")
    xy_true = lla_to_enu(V["lat"], V["lon"], V["lat"][0], V["lon"][0])
    ifix = np.clip(np.searchsorted(tv, fix_t), 0, len(tv) - 1)

    j0, psi0 = _find_init(fix_t, fix_xy, fix_spd, avail, tv[min(i_start, len(tv) - 1)], min_init_disp)
    i0 = int(np.clip(ifix[j0], i_start, n - 1))
    kf = EKF6([fix_xy[j0, 0], fix_xy[j0, 1], psi0, fix_spd[j0], 0.0, 0.0],
              [(sig_pos * 1.5) ** 2, (sig_pos * 1.5) ** 2, np.radians(3.0) ** 2, 1.0,
               0.01 ** 2, 0.05 ** 2], sig_pos=sig_pos, q=q)
    kf.last_fix_t = fix_t[j0]

    out = dict(x=np.full(n, np.nan), y=np.full(n, np.nan), psi=np.full(n, np.nan),
               v=np.full(n, np.nan), b=np.full(n, np.nan), mode=np.zeros(n, int),
               acc=np.zeros(nF, bool), i0=i0, j0=j0, i_end=n)
    psi_hist = np.full(n, np.nan)
    fix_ptr, last_avail = j0 + 1, j0
    dt_fix_nom = float(np.median(np.diff(fix_t))) if nF > 2 else 9.0
    for i in range(i0, n):
        t_i = tv[i]
        z = bool(zupt[i]) if zupt is not None else False
        kf.predict(w_vert[i], a_fwd[i], dt, zupt=z)
        psi_hist[i] = kf.x[2]
        if v_ai is not None and (i - i0) % ai_every == 0 and np.isfinite(v_ai[i]):
            sa = sigma_ai[i] if np.ndim(sigma_ai) else sigma_ai
            kf.update_speed(float(v_ai[i]), sigma=float(sa))
        while fix_ptr < nF and fix_t[fix_ptr] <= t_i:
            f = fix_ptr
            fix_ptr += 1
            if outage_mask[f]:
                continue
            sg = sig_pos_arr[f] if sig_pos_arr is not None else sig_pos
            before = np.array(kf.pos)
            accepted = kf.update_pos(fix_xy[f], t=fix_t[f], sigma=sg)
            out["acc"][f] = accepted
            if accepted:
                kf.update_speed(float(fix_spd[f]), sigma=sig_fix_spd)
                if mapper is not None and np.hypot(*(np.array(kf.pos) - before)) > 15.0:
                    mapper.reset()
                if f - 1 == last_avail and cc["mode"] != "off":
                    dxy = fix_xy[f] - fix_xy[f - 1]
                    d = float(np.hypot(*dxy))
                    if cc["mode"] == "legacy":
                        if d >= 15.0:
                            kf.update_course(math.atan2(dxy[0], dxy[1]), sigma=cc["legacy_sigma_deg"])
                    elif d >= cc["min_disp"]:
                        ia, ib = int(ifix[f - 1]), int(ifix[f])
                        if ia >= i0 and ib > ia and np.isfinite(psi_hist[ia]) and np.isfinite(psi_hist[ib]):
                            im = (ia + ib) // 2
                            dpsi = abs(wrap_pi(psi_hist[ib] - psi_hist[ia]))
                            if dpsi < math.radians(cc["max_dpsi_deg"]):
                                course = math.atan2(dxy[0], dxy[1])
                                sg2 = (sig_pos_arr[f - 1] if sig_pos_arr is not None else sig_pos)
                                s_geo = math.atan(math.hypot(sg, sg2) / d)
                                s_all = math.sqrt(s_geo ** 2 + (cc["curve_k"] * dpsi) ** 2
                                                  + math.radians(cc["sig_floor_deg"]) ** 2)
                                r = wrap_pi(course - psi_hist[im])
                                kf.update_heading_innov(r, s_all, gate_rad=cc["gate_rad"])
                last_avail = f
        if (mapper is not None and (i - i0) % mc["every"] == 0 and not z and kf.x[3] > 2.0
                and (t_i - kf.last_fix_t) >= mc["after_s"]):
            sig_obs = float(np.clip(math.sqrt(max(abs(kf.P[0, 0]), abs(kf.P[1, 1]))) + 6.0,
                                    mc["sig_obs_min"], mc["sig_obs_max"]))
            sh = None
            if mc.get("heading_aware", True) and kf.x[3] > 3.0:
                sh = math.sqrt(math.radians(mc.get("head_sig_deg", 30.0)) ** 2 + abs(kf.P[2, 2]))
            m = mapper.step(np.array(kf.pos), sig_obs, heading=(kf.x[2] if sh else None), sig_head=sh)
            if trace is not None:
                trace.append(dict(i=i, est=np.array(kf.pos), m=m, sig=sig_obs, psi=kf.x[2]))
            if m is not None and m["conf"] >= mc["conf_min"]:
                kf.update_map(m["xy"], m["tan"], sig_xt=mc["sig_xt"], sig_head_deg=mc["sig_head_deg"],
                              gate_xt=mc["gate_xt"], use_heading=mc["use_heading"])
        out["x"][i], out["y"][i] = kf.pos
        out["psi"][i], out["v"][i], out["b"][i] = kf.x[2], kf.x[3], kf.x[4]
        out["mode"][i] = int((t_i - kf.last_fix_t) < 1.35 * dt_fix_nom)   # 1 = GNSS-aided, 0 = DR
    out["err"] = np.hypot(out["x"] - xy_true[:n, 0], out["y"] - xy_true[:n, 1])
    out["xy_true"] = xy_true
    out["n_rej"], out["n_acc"], out["n_map"] = kf.n_rej, kf.n_acc, kf.n_map
    return out


# =========================================================================== benchmark helpers
def _runs(mask):
    """(start, end_exclusive) of True runs in a boolean array."""
    m = np.r_[False, mask, False]
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def schedule_windows(V, i_lo, i_hi, dur_s=60.0, spacing_s=150.0, min_path_m=150.0):
    """Blackout windows of dur_s every ~spacing_s inside [i_lo, i_hi) - NOT filtered for 'easy' driving.

    Only requirement: the vehicle covers >= min_path_m (otherwise drift% is meaningless).
    """
    dt = float(np.median(np.diff(V["t"])))
    L, step = int(dur_s / dt), int(spacing_s / dt)
    wins, a = [], int(i_lo)
    while a + L < min(i_hi, len(V["v"]) - 1):
        path = float(np.sum(V["v"][a:a + L]) * dt)
        if path >= min_path_m:
            wins.append((a, a + L))
            a += step
        else:
            a += 20
    return wins


def moving_fraction(V, a, b, thr=1.5):
    return float(np.mean(V["v"][a:b] > thr))


def scenario_windows(V, i_lo, i_hi):
    """The two PS scenarios: >=1 km at >=55 km/h ('tunnel at 60 km/h') and 50 m at <=30 km/h."""
    dt = float(np.median(np.diff(V["t"])))
    v = V["v"]
    hi = lo = None
    for (a, b) in _runs(v >= 55 / 3.6):
        a = max(a, i_lo); b = min(b, i_hi)
        if b - a < 10:
            continue
        cum = np.cumsum(v[a:b]) * dt
        if cum[-1] >= 1000.0:
            hi = (a, a + int(np.searchsorted(cum, 1000.0)) + 1)
            break
    for (a, b) in _runs((v > 1.0) & (v <= 30 / 3.6)):
        a = max(a, i_lo); b = min(b, i_hi)
        if b - a < 5:
            continue
        cum = np.cumsum(v[a:b]) * dt
        if cum[-1] >= 50.0:
            lo = (a, a + int(np.searchsorted(cum, 50.0)) + 1)
            break
    return hi, lo


def merge_windows(base, extra, guard=200):
    """Drop base windows that overlap (within `guard` samples) any 'extra' window; append extras."""
    keep = [w for w in base if all(w[1] + guard < e[0] or w[0] > e[1] + guard for e in extra)]
    return sorted(keep + list(extra))


def window_outage_mask(fix_t_rel, V, windows):
    tv = V["t"] - V["t"][0]
    m = np.zeros(len(fix_t_rel), bool)
    for (a, b) in windows:
        m |= (fix_t_rel >= tv[a]) & (fix_t_rel < tv[min(b, len(tv) - 1)])
    return m


def window_metrics(xy_est, xy_true, a, b):
    """End-of-window error, path length, RMS and max error inside the window."""
    b = min(b, len(xy_est) - 1)
    de = float(np.hypot(*(xy_est[b] - xy_true[b])))
    path = float(np.sum(np.linalg.norm(np.diff(xy_true[a:b + 1], axis=0), axis=1)))
    e = np.linalg.norm(xy_est[a:b + 1] - xy_true[a:b + 1], axis=1)
    return dict(end=de, path=path, drift=de / max(path, 1e-9), rms=float(np.sqrt(np.nanmean(e ** 2))),
                max=float(np.nanmax(e)))
