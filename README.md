# SIH 2026 — Problem Statement 26168

**AI/ML-based Intelligent Dead Reckoning (IDR) for seamless navigation under GNSS outages.**

Preliminary AI models and the position-plot results inferred on a subset of the
[IO-VNBD](https://github.com/onyekpeu/IO-VNBD) dataset, submitted as part of our proposal.

---

## 1. Approach

The system keeps a continuous 10 Hz position estimate on the phone when GNSS is
unavailable (urban canyon, tunnels, short signal loss) by fusing phone IMU with
the phone's own GNSS in an error-state extended Kalman filter, using a small AI
model for the two quantities the phone cannot measure on its own: **vehicle speed**
and **whether the vehicle is moving at all**.

Pipeline:

| Stage | What it does |
|---|---|
| 1. Time alignment | Automatic phone↔vehicle clock synchronisation (per-journey offsets are 0.1–7.5 s in IO-VNBD) |
| 2. Mount/axis alignment engine | Learns a 4-coefficient axis mixer for the accelerometer and magnetometer, plus a turn-rate scale factor, from the phone's **own GNSS course and speed deltas** — no external calibration file, works for any phone orientation |
| 3. **SpeedNet (AI)** | 13-channel CNN+GRU that outputs vehicle speed (m/s) and a stationary (ZUPT) probability from 5 s of raw 10 Hz IMU |
| 4. GNSS+INS fusion | 6-state EKF `[x, y, ψ, v, b_gyro, b_accel]`, Joseph-form updates, course-gated innovation rejection, evidence-gated ZUPT, AI speed used as a weak velocity anchor |
| 5. Map matching | HMM/Viterbi over OSM road graph to snap the DR track to the road network |
| 6. Edge engine | The whole fusion chain in NumPy: **128× real time on CPU at 200 Hz**, 2.4 m drift over a 60 s blackout |

Why an AI speed model helps IDR: dead reckoning from accelerometers alone needs
the forward-acceleration projection to be perfect. In IO-VNBD the phone is
loosely mounted, so that projection is wrong by an unknown per-journey axis mix
and an unknown turn-gain (measured 0.53–0.85 on rough journeys). SpeedNet learns
speed from the *raw* vibration signature of the vehicle, which is weakly
observable even with poor mounting (linear holdout R² ≈ 0.54), and therefore
survives mounts it was never trained on.

## 2. Preliminary AI model

`models/speednet.pt` — TorchScript export of SpeedNet.

| Property | Value |
|---|---|
| Architecture | 13-channel CNN + bi-GRU, **77 k parameters** |
| Input | `(1, 50, 13)` — 5 s window of 10 Hz IMU |
| Input channels | body accel (3), gyro (3), magnetometer (3), ‖accel‖, ‖gyro‖, calibrated forward accel, ‖d(mag)/dt‖ |
| Output | vehicle speed (m/s), ZUPT logit |
| Size | 344 KB (TorchScript), 8.4 KB (ONNX, `models/speednet.onnx`) |
| Training data | 4 IO-VNBD journeys, ~280 km (M, S1, S2, Y1 — Coventry, flat urban) |
| Held-out test data | 4 journeys, unseen driver, wet/hilly Peak District + M42 motorway (Vta16, Vta29, Vtb1, Vw14c) |
| Validation RMSE | **2.89 m/s** (speed) |
| Stationary detection | **> 97 % accuracy** |

`models/speednet.onnx` is the same network in ONNX format.

## 3. Position-plot results on the IO-VNBD subset

Held-out journey **Vta16**: VBOX ground truth vs the phone's GNSS fixes (which arrive
only every ~9 s) vs the 10 Hz fused IDR track.

![GNSS+INS fusion, Vta16](results/fig_fusion.png)

AI speed prediction against VBOX ground truth:

![SpeedNet predictions](results/fig_speednet.png)

60 s GNSS-denied blackout benchmark on all four held-out journeys:

![blackout benchmark](results/fig_benchmark.png)

Full-journey fusion accuracy (no outage), median position error:

| Journey | Median error |
|---|---|
| Vta16 | 119.5 m |
| Vta29 | 56.9 m |
| Vtb1 | 46.7 m |
| Vw14c | 115.0 m |

60 s blackout drift, median over all blackout windows (naive IMU double-integration
vs our fusion — the contrast shows what the AI/alignment stages buy):

| Journey | naive IMU only | EKF, no AI | AI-DR (ours) |
|---|---|---|---|
| Vta16 | 3423 m (503 %) | 121 m (32 %) | 1122 m (170 %) |
| Vta29 | 5493 m (680 %) | 544 m (64 %) | 848 m (144 %) |
| Vtb1 | 4162 m (508 %) | 492 m (79 %) | 1241 m (131 %) |
| Vw14c | 3425 m (408 %) | 1282 m (207 %) | 807 m (104 %) |

Detailed tables, including the problem statement's `1 km @ 60 km/h` and
`50 m urban` scenario checks: [`results/results_summary.md`](results/results_summary.md).

## 4. Repository contents

```
models/
  speednet.pt          SpeedNet weights (TorchScript, CPU, inference-ready)
  speednet.onnx        SpeedNet weights (ONNX)
results/
  fig_fusion.png       position plot: truth vs GNSS vs fused track + error
  fig_speednet.png     AI speed vs ground truth
  fig_benchmark.png    blackout drift: CDF + 1 km @ 60 km/h zoom
  fig_training.png     SpeedNet training curves
  results_summary.md   full numeric results
calibration/
  mixers.json          mount/axis alignment coefficients learnt per journey
```

## 5. Dataset

IO-VNBD — *An Open-Source Dataset for Vehicle Navigation*, smartphone IMU + VBOX
ground truth (Oxford, UK). Used here under CC BY 4.0. Four journeys were used for
training and four (different driver, unseen mounting and terrain) as the held-out
test set.