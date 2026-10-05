"""
idr_ai.py - learning layer of DISHA / IDR v2

  SpeedNetA   GNSS-ANCHORED speed network (siamese CNN+GRU encoder). Instead of regressing an
              absolute speed from vibration (information ceiling r~0.6 on 10 Hz phone data) it
              predicts  speed(now) = f( last GNSS speed, age of that fix, IMU now, IMU at the fix ).
              Trained with SIMULATED GNSS dropouts of 0-90 s, i.e. for exactly the blackout use case.
              The zero-initialised output head starts as 'hold last GNSS speed' and learns only the
              deviations it can really see in the inertial data. Normalisation is baked in, so the
              exported ONNX takes RAW features.
  RAdapter    small MLP predicting the phone-GNSS position error scale per fix.
  prepare_journey / fuse / tune   plumbing: causal calibration -> features -> EKF with all aiding.
"""
import copy
import math
import time

import numpy as np
import pandas as pd

import idr_core as ic

W_WIN, STRIDE, SKIP = 50, 5, 600          # 5 s window, 0.5 s stride, skip first 60 s
N_CH = 13
AGE_BINS = [0, 10, 30, 60, 90]            # s - validation / sigma table


# =========================================================================== features
def window_features(G, a_fwd, dt):
    """(N,13) per-sample features on the 10 Hz grid (device-frame vectors + invariants)."""
    lin, gyr, mag = G["lin"], G["gyr"], G["mag"]
    dmag = np.linalg.norm(np.gradient(mag, axis=0) / dt, axis=1)
    return np.column_stack([lin, gyr, mag, np.linalg.norm(lin, axis=1), np.linalg.norm(gyr, axis=1),
                            a_fwd, dmag]).astype(np.float32)


def make_windows(feat, grav, v=None, skip=SKIP, w=W_WIN, stride=STRIDE):
    idx = np.arange(skip + w, len(feat) - 1, stride)
    X = np.stack([feat[i - w:i] for i in idx]).astype(np.float32)
    gh = grav[idx - w // 2]
    gh = (gh / (np.linalg.norm(gh, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
    out = dict(idx=idx, X=X, ghat=gh)
    if v is not None:
        out["y"] = v[idx].astype(np.float32)
    return out


# =========================================================================== journey prep
def prepare_journey(name, S, V, calib_s=900.0, calib_frac=0.4, delta=None):
    """Everything that is computed ONCE per journey, strictly causally."""
    if delta is None:
        delta, sync_err = ic.autosync(S, V)
    else:
        sync_err = float("nan")
    dt = float(np.median(np.diff(V["t"])))
    G = ic.grid_channels(S, V, delta)
    tcal = ic.calib_horizon(V, calib_s, calib_frac)
    mix = ic.fit_heading_mixer_gnss(S, V, delta, t_max=tcal, G=G)
    amix = ic.fit_accel_mixer_gnss(S, V, delta, t_max=tcal, G=G)
    fix_t = (S["fix"]["t"] - S["t"][0]) + delta
    fix_xy = ic.lla_to_enu(S["fix"]["lat"], S["fix"]["lon"], V["lat"][0], V["lon"][0])
    tv = V["t"] - V["t"][0]
    j = dict(name=name, S=S, V=V, delta=delta, sync_err=sync_err, dt=dt, G=G, tcal=tcal, i_cal=int(tcal / dt),
             mix=mix, amix=amix, fix_t=fix_t, fix_xy=fix_xy, fix_spd=S["fix"]["spd"],
             fix_acc=S["fix"]["acc"], ifix=np.clip(np.searchsorted(tv, fix_t), 0, len(tv) - 1))
    j["feat"] = window_features(G, amix["a_fwd"], dt)
    j["win"] = make_windows(j["feat"], G["grav"], V["v"])
    return j


def tilt_deg(j):
    g = j["G"]["grav"].mean(axis=0)
    g = g / np.linalg.norm(g)
    return float(np.degrees(np.arccos(np.clip(abs(g[2]), 0, 1))))


# =========================================================================== SpeedNetA
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_model(gm, gsd, ch=N_CH):
    torch, nn, F = _torch()

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(ch, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 96, 3, stride=2, padding=1), nn.BatchNorm1d(96), nn.GELU())
            self.rnn = nn.GRU(96, 64, batch_first=True)
            self.proj = nn.Sequential(nn.Linear(64, 64), nn.GELU())

        def forward(self, x):                       # (B, W, C) normalised
            h = self.net(x.transpose(1, 2))
            _, hn = self.rnn(h.transpose(1, 2))
            return self.proj(hn[-1])

    class SpeedNetA(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("gm", torch.as_tensor(np.asarray(gm, np.float32).reshape(1, 1, -1)))
            self.register_buffer("gsd", torch.as_tensor(np.asarray(gsd, np.float32).reshape(1, 1, -1)))
            self.enc = Encoder()
            self.z_out = nn.Linear(64, 1)
            self.head = nn.Sequential(nn.Linear(64 * 3 + 3, 96), nn.GELU(), nn.Dropout(0.05),
                                      nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

        def norm(self, x):
            return (x - self.gm) / self.gsd

        def encode(self, x_raw):
            e = self.enc(self.norm(x_raw))
            return e, self.z_out(e).squeeze(-1)

        def speed_from(self, e, ea, anch):          # anch = [v_anchor m/s, age s, valid flag]
            va, tau, flag = anch[:, 0:1], anch[:, 1:2], anch[:, 2:3]
            f = torch.cat([e, ea, e - ea, va / 20.0, tau / 60.0, flag], dim=1)
            u = va + self.head(f)
            return F.softplus(u, beta=10.0).squeeze(-1)

        def forward(self, x_raw, xa_raw, anch):
            B = x_raw.shape[0]
            e_all, z_all = self.encode(torch.cat([x_raw, xa_raw], dim=0))
            return self.speed_from(e_all[:B], e_all[B:], anch), z_all[:B]

    return SpeedNetA()


def _rot_about(v, g, th):
    """Rodrigues rotation of vectors v (B,W,3) about unit axis g (B,3) by angle th (B,)."""
    torch, _, _ = _torch()
    c = torch.cos(th)[:, None, None]
    s = torch.sin(th)[:, None, None]
    gg = g[:, None, :]
    gxv = torch.cross(gg.expand_as(v), v, dim=2)
    gdv = (gg * v).sum(dim=2, keepdim=True)
    return v * c + gxv * s + gg * gdv * (1 - c)


def train_speednet(Xtr, ytr, gtr, jtr, Xva, yva, gva, jva, device="cpu", epochs=24, batch=384,
                   seed=0, age_max_steps=180, noise_v=0.4, verbose=True):
    """Xtr: (N,W,C) RAW windows; jtr: journey-start index of each window (anchors stay inside a journey)."""
    torch, nn, F = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gm = Xtr.reshape(-1, Xtr.shape[2]).mean(axis=0)
    gsd = Xtr.reshape(-1, Xtr.shape[2]).std(axis=0) + 1e-3
    model = build_model(gm, gsd).to(device)
    Xt = torch.as_tensor(Xtr, device=device)
    yt = torch.as_tensor(ytr, device=device)
    gt = torch.as_tensor(gtr, device=device)
    Xv = torch.as_tensor(Xva, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    steps_per_ep = int(math.ceil(len(Xt) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=epochs * steps_per_ep)
    N = len(Xt)
    ages = np.r_[rng.integers(0, 24, 40000), rng.integers(24, age_max_steps + 1, 60000)]  # 0-12 s: 40 %, 12-90 s: 60 %

    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(N)
        tot = 0.0
        for k in range(0, N, batch):
            ids = perm[k:k + batch]
            age = rng.choice(ages, len(ids))
            anc = np.maximum(ids - age, jtr[ids])
            tau = ((ids - anc) * 0.5).astype(np.float32)
            va = np.clip(ytr[anc] + rng.normal(0, noise_v, len(ids)).astype(np.float32), 0, None)
            x = Xt[ids]
            xa = Xt[anc]
            if True:                                   # yaw-about-gravity augmentation (mount invariance)
                th = torch.as_tensor(rng.uniform(0, 2 * np.pi, len(ids)) * (rng.random(len(ids)) < 0.5),
                                     dtype=torch.float32, device=device)
                tha = th.clone()
                g1, g2 = gt[ids], gt[anc]
                x = x.clone(); xa = xa.clone()
                for c0 in (0, 3, 6):
                    x[:, :, c0:c0 + 3] = _rot_about(x[:, :, c0:c0 + 3], g1, th)
                    xa[:, :, c0:c0 + 3] = _rot_about(xa[:, :, c0:c0 + 3], g2, th)
            anch = torch.as_tensor(np.stack([va, tau, np.ones_like(va)], 1), dtype=torch.float32, device=device)
            v_pred, z_logit = model(x, xa, anch)
            ytrue = yt[ids]
            loss = F.huber_loss(v_pred, ytrue, delta=1.0) + 0.3 * F.binary_cross_entropy_with_logits(
                z_logit, (ytrue < 0.3).float())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(ids)
        rep = evaluate_speednet(model, Xv, yva, jva, device, seed=123)
        score = float(np.sqrt(np.mean([rep["rmse"][b] ** 2 for b in range(len(AGE_BINS) - 1)])))
        hist.append((tot / N, score, rep["zacc"]))
        if verbose:
            print(f"epoch {ep:2d}: loss {tot/N:6.3f} | val RMSE by age {[round(r, 2) for r in rep['rmse']]} "
                  f"| hold-last {[round(r, 2) for r in rep['rmse_hold']]} | ZUPT acc {rep['zacc']:.3f}")
        if score < best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, np.array(hist), gm, gsd


def evaluate_speednet(model, Xva, yva, jva, device="cpu", seed=0, reps=2, noise_v=0.4):
    """Validation RMSE per anchor-age bin, vs the 'hold last GNSS speed' baseline."""
    torch, _, _ = _torch()
    rng = np.random.default_rng(seed)
    model.eval()
    Xv = torch.as_tensor(Xva, device=device)
    with torch.no_grad():
        E, Z = [], []
        for k in range(0, len(Xv), 2048):
            e, z = model.encode(Xv[k:k + 2048])
            E.append(e); Z.append(z)
        E = torch.cat(E); Z = torch.cat(Z)
    zacc = float(((torch.sigmoid(Z).cpu().numpy() > 0.5) == (yva < 0.3)).mean())
    rmse, rmse_hold, nb = [], [], []
    ids_all = np.arange(len(Xva))
    for b in range(len(AGE_BINS) - 1):
        lo, hi = int(AGE_BINS[b] / 0.5), int(AGE_BINS[b + 1] / 0.5)
        se, seh, n = 0.0, 0.0, 0
        for _ in range(reps):
            ids = ids_all[::2]
            age = rng.integers(max(lo, 1) if b else 0, hi, len(ids))
            anc = np.maximum(ids - age, jva[ids])
            tau = ((ids - anc) * 0.5).astype(np.float32)
            va = np.clip(yva[anc] + rng.normal(0, noise_v, len(ids)).astype(np.float32), 0, None)
            anch = torch.as_tensor(np.stack([va, tau, np.ones_like(va)], 1), dtype=torch.float32, device=device)
            with torch.no_grad():
                vp = model.speed_from(E[ids], E[anc], anch).cpu().numpy()
            se += float(((vp - yva[ids]) ** 2).sum())
            seh += float(((va - yva[ids]) ** 2).sum())
            n += len(ids)
        rmse.append(math.sqrt(se / n)); rmse_hold.append(math.sqrt(seh / n)); nb.append(n)
    return dict(rmse=rmse, rmse_hold=rmse_hold, zacc=zacc)


def sigma_table(rep, margin=1.25, floor=0.5):
    """Per-age-bin measurement sigma for the EKF from validation RMSE."""
    centers = [0.5 * (AGE_BINS[i] + AGE_BINS[i + 1]) for i in range(len(AGE_BINS) - 1)]
    return np.array(centers), np.maximum(np.array(rep["rmse"]) * margin, floor)


def sigma_from_age(tau, table):
    c, s = table
    return np.interp(tau, c, s)


# =========================================================================== speed track
def encode_windows(model, X, device="cpu", bs=2048):
    torch, _, _ = _torch()
    model.eval()
    E, Z = [], []
    with torch.no_grad():
        for k in range(0, len(X), bs):
            e, z = model.encode(torch.as_tensor(X[k:k + bs], device=device))
            E.append(e); Z.append(torch.sigmoid(z).cpu().numpy())
    return torch.cat(E), np.concatenate(Z)


def dwell(z, hold=10):
    """Hysteresis: state flips only after `hold` consecutive samples of the new value."""
    out = np.zeros(len(z), bool)
    state, run = False, 0
    for i, zz in enumerate(z):
        if zz == state:
            run = 0
        else:
            run += 1
            if run >= hold:
                state, run = bool(zz), 0
        out[i] = state
    return out


def to_grid(idx, vals, n, fill=np.nan):
    """Hold the value of the latest window (ending at idx) on the 10 Hz grid."""
    pos = np.searchsorted(idx, np.arange(n), side="right") - 1
    out = np.full(n, fill, float)
    ok = pos >= 0
    out[ok] = np.asarray(vals, float)[pos[ok]]
    return out


def speed_track(model, j, outage_mask, device="cpu", cache=None):
    """GNSS-anchored AI speed on the 10 Hz grid for a given set of available fixes.

    Returns dict(v_ai, tau, zprob, zupt). For every window the anchor is the last AVAILABLE fix
    (so inside a blackout the anchor age grows exactly as it would on the phone)."""
    torch, _, _ = _torch()
    win = j["win"]
    idx = win["idx"]
    n = len(j["V"]["v"])
    if cache is not None and "E" in cache:
        E, zp = cache["E"], cache["z"]
    else:
        E, zp = encode_windows(model, win["X"], device)
        if cache is not None:
            cache["E"], cache["z"] = E, zp
    avail = np.where(~np.asarray(outage_mask, bool))[0]
    ifx = j["ifix"][avail]
    k_last = np.searchsorted(ifx, idx, side="right") - 1          # last available fix at/before window end
    ok = k_last >= 0
    v = np.full(len(idx), np.nan)
    tau = np.full(len(idx), np.nan)
    if ok.any():
        f = avail[k_last[ok]]
        anc = np.clip(np.searchsorted(idx, j["ifix"][f]), 0, len(idx) - 1)
        anc = np.minimum(anc, np.where(ok)[0])                    # anchor window never after current
        age = (idx[ok] - j["ifix"][f]) * 0.1
        va = np.clip(j["fix_spd"][f], 0, None)
        anch = torch.as_tensor(np.stack([va, age, np.ones_like(va)], 1), dtype=torch.float32, device=device)
        ids = np.where(ok)[0]
        out = []
        with torch.no_grad():
            for k in range(0, len(ids), 4096):
                sl = slice(k, k + 4096)
                out.append(model.speed_from(E[ids[sl]], E[anc[sl]], anch[sl]).cpu().numpy())
        v[ok] = np.concatenate(out)
        tau[ok] = age
    z_state = dwell(to_grid(idx, zp > 0.5, n, 0.0) > 0.5, hold=10)
    v_grid = to_grid(idx, v, n)
    tau_grid = to_grid(idx, tau, n)
    v_grid[z_state] = 0.0
    return dict(v_ai=v_grid, tau=tau_grid, zprob=to_grid(idx, zp, n, 0.0), zupt=z_state)


def add_sigma(spd, table):
    """Attach the age-dependent AI-speed sigma (m/s) to a speed_track() result."""
    sg = sigma_from_age(np.nan_to_num(spd["tau"], nan=90.0), table)
    spd["sigma"] = np.where(np.isfinite(spd["tau"]), sg, 3.0)
    return spd


# =========================================================================== R-adapter
def gnss_context(j):
    """Per-fix context features for the position-sigma adapter (phone-observable only)."""
    G = j["G"]
    lin, gyr = G["lin"], G["gyr"]
    vib = pd.Series(np.linalg.norm(lin, axis=1)).rolling(50, min_periods=5).std().bfill().to_numpy()
    wm = pd.Series(np.linalg.norm(gyr, axis=1)).rolling(50, min_periods=5).std().bfill().to_numpy()
    ifx = j["ifix"]
    dtf = np.r_[9.0, np.diff(j["fix_t"])]
    return np.column_stack([j["fix_acc"], dtf, j["fix_spd"], vib[ifx], wm[ifx]]).astype(np.float64)


def fix_errors(j):
    """Truth-based label (training only): horizontal error of each phone fix vs VBOX."""
    V = j["V"]
    xy = ic.lla_to_enu(V["lat"], V["lon"], V["lat"][0], V["lon"][0])
    tv = V["t"] - V["t"][0]
    ex = np.interp(j["fix_t"], tv, xy[:, 0])
    ey = np.interp(j["fix_t"], tv, xy[:, 1])
    return np.hypot(j["fix_xy"][:, 0] - ex, j["fix_xy"][:, 1] - ey)


def fit_radapter(train_js, device="cpu", epochs=250, seed=0):
    torch, nn, F = _torch()
    torch.manual_seed(seed)
    F_, E_ = [], []
    for j in train_js:
        f, e = gnss_context(j), fix_errors(j)
        ok = np.isfinite(f).all(1) & np.isfinite(e) & (e < 200) & (e > 0.05)
        F_.append(f[ok]); E_.append(e[ok])
    F_, E_ = np.concatenate(F_), np.concatenate(E_)
    mu, sd = F_.mean(0), F_.std(0) + 1e-9
    net = nn.Sequential(nn.Linear(F_.shape[1], 16), nn.GELU(), nn.Linear(16, 1)).to(device)
    Xf = torch.as_tensor((F_ - mu) / sd, dtype=torch.float32, device=device)
    yl = torch.log(torch.as_tensor(E_, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xf), device=device)
        for k in range(0, len(perm), 256):
            b = perm[k:k + 256]
            loss = F.mse_loss(net(Xf[b]).squeeze(-1), yl[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return dict(net=net, mu=mu, sd=sd, n=len(Xf))


def radapter_sigma(ra, j, device="cpu", mult=1.25, floor=4.0, ceil=40.0):
    torch, _, _ = _torch()
    f = gnss_context(j)
    ok = np.isfinite(f).all(1)
    sig = np.full(len(f), 15.0)
    with torch.no_grad():
        p = ra["net"](torch.as_tensor((f[ok] - ra["mu"]) / ra["sd"], dtype=torch.float32, device=device)).squeeze(-1)
    sig[ok] = np.exp(p.cpu().numpy())
    return np.clip(sig * mult, floor, ceil)


# =========================================================================== fusion wrapper
DEFAULT_P = dict(q_pos=0.20, q_psi_deg=None, q_v=0.35, q_bg=5e-5, q_ba=0.002, sig_ai_scale=1.4,
                 ai_every=5, sig_floor_deg=2.0, curve_k=0.25, course_mode="delayed",
                 map_sig_xt=12.0, map_sig_head=10.0, map_conf_min=0.9, map_after_s=3.0, map_every=10,
                 map_gate=30.0, map_use_heading=False)


def fuse(j, P, outage_mask, *, spd=None, use_ai_speed=True, use_zupt=True, use_radapter=None,
         mapper=None, i_start=None, i_end=None, zero_afwd=False, course_mode=None, trace=None):
    """One causal GNSS+INS(+AI+map) run. `spd` = speed_track() output (or None for no AI)."""
    P = {**DEFAULT_P, **P}
    qpsi = P["q_psi_deg"] if P["q_psi_deg"] is not None else max(j["mix"].get("sigma_deg_s", 8.0) * 3.0, 8.0)
    q = dict(pos=P["q_pos"], psi=math.radians(qpsi), v=P["q_v"], b_g=P["q_bg"], b_a=P["q_ba"])
    v_ai, sig_ai, zupt = None, 2.5, None
    if spd is not None and use_ai_speed:
        v_ai = spd["v_ai"]
        sig_ai = P["sig_ai_scale"] * spd["sigma"]
    if spd is not None and use_zupt:
        zupt = spd["zupt"]
    sig_arr = j.get("sig_pos_arr") if (use_radapter is None or use_radapter) else None
    a_fwd = np.zeros_like(j["amix"]["a_fwd"]) if zero_afwd else j["amix"]["a_fwd"]
    i0 = j["i_cal"] if i_start is None else i_start
    return ic.run_ekf(
        j["V"], j["mix"]["w_vert"], a_fwd, j["fix_t"], j["fix_xy"], j["fix_spd"], v_ai=v_ai, sigma_ai=sig_ai,
        ai_every=int(P["ai_every"]), outage_mask=outage_mask, sig_pos_arr=sig_arr, zupt=zupt, q=q,
        i_start=i0, i_end=i_end,
        course_cfg=dict(mode=course_mode or P["course_mode"], sig_floor_deg=P["sig_floor_deg"], curve_k=P["curve_k"]),
        mapper=mapper, trace=trace,
        map_cfg=dict(sig_xt=P["map_sig_xt"], sig_head_deg=P["map_sig_head"], conf_min=P["map_conf_min"],
                     after_s=P["map_after_s"], every=int(P["map_every"]), gate_xt=P["map_gate"],
                     use_heading=bool(P["map_use_heading"])))


def eval_windows(res, wins):
    est = np.column_stack([res["x"], res["y"]])
    return [ic.window_metrics(est, res["xy_true"], a, b) for (a, b) in wins]


def objective(ms, cap=0.5):
    if not ms:
        return 1.0
    d = np.array([min(m["drift"], cap) for m in ms])
    return float(0.7 * d.mean() + 0.3 * np.percentile(d, 90))


def tune(evaluate, P0, space, n_rand=24, n_local=16, seed=0, verbose=True):
    """Random search + local perturbation (CEM-lite). space: name -> (lo, hi, 'lin'|'log')."""
    rng = np.random.default_rng(seed)

    def sample(center=None, shrink=1.0):
        P = dict(P0)
        for k, (lo, hi, kind) in space.items():
            if center is None:
                u = rng.uniform(0, 1)
            else:
                c = center[k]
                cu = (math.log(c / lo) / math.log(hi / lo)) if kind == "log" else (c - lo) / (hi - lo)
                u = float(np.clip(cu + rng.normal(0, 0.18 * shrink), 0, 1))
            P[k] = lo * (hi / lo) ** u if kind == "log" else lo + u * (hi - lo)
        return P

    best_P, best_s = dict(P0), evaluate(P0)
    base_s = best_s
    t0 = time.time()
    if verbose:
        print(f"  start objective {best_s:.4f}")
    for it in range(n_rand + n_local):
        P = sample() if it < n_rand else sample(best_P, shrink=1.0 - 0.6 * (it - n_rand) / max(n_local, 1))
        try:
            s = evaluate(P)
        except Exception as e:                       # a bad corner must not kill the search
            if verbose:
                print("  candidate failed:", type(e).__name__, str(e)[:80])
            continue
        if s < best_s:
            best_P, best_s = P, s
            if verbose:
                print(f"  [{it:2d}] objective {s:.4f}  ({time.time()-t0:.0f}s)")
    return best_P, base_s, best_s


# =========================================================================== export
def export_onnx_safe(model, path, example, input_names, output_names, opset=17):
    """Export to ONE self-contained .onnx file and PROVE it matches PyTorch.

    Why this exists: recent PyTorch versions use the dynamo exporter by default, which writes the
    weights into a sidecar '<name>.onnx.data' file. Copying only the .onnx (e.g. to GitHub or the
    phone) leaves a graph whose weights are missing -> 'wrong / corrupt weights'. We (1) try the
    classic exporter, (2) fall back to dynamo, and in BOTH cases (3) re-load with external data,
    (4) re-save with every weight embedded, (5) delete sidecar files, (6) run onnx.checker and
    (7) compare against PyTorch with onnxruntime on the example inputs."""
    import os
    import onnx
    import onnxruntime as ort
    torch, _, _ = _torch()
    model = model.eval().cpu()
    tmp = path + ".tmp.onnx"
    errs = []
    for kind in ("legacy", "dynamo"):
        try:
            if kind == "legacy":
                torch.onnx.export(model, example, tmp, input_names=input_names, output_names=output_names,
                                  opset_version=opset, dynamo=False)
            else:
                torch.onnx.export(model, example, tmp, input_names=input_names, output_names=output_names,
                                  opset_version=opset)
            break
        except Exception as e:                             # noqa
            errs.append(f"{kind}: {type(e).__name__}: {str(e)[:120]}")
            for f in (tmp, tmp + ".data"):
                if os.path.exists(f):
                    os.remove(f)
    else:
        raise RuntimeError("ONNX export failed with both exporters: " + " | ".join(errs))
    m = onnx.load(tmp, load_external_data=True)
    onnx.save_model(m, path, save_as_external_data=False)
    for f in (tmp, tmp + ".data", path + ".data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = [t.numpy() for t in model(*example)]
    out = sess.run(None, {n: e.numpy() for n, e in zip(input_names, example)})
    diff = [float(np.abs(o - r).max()) for o, r in zip(out, ref)]
    return dict(path=path, kind=kind, bytes=os.path.getsize(path), max_abs_diff=diff, errors=errs)
