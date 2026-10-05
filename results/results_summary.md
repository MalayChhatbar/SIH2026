# SIH 26168 - DISHA IDR v2 results (IO-VNBD, held-out journeys, calibration frozen before blackouts)

Data mode: **iovnbd**

| Method | 60 s windows | median drift % | mean % | p90 % | windows < 10 % | median end error (m) |
|---|---|---|---|---|---|---|
| M0 naive integrate |a| | 28 | 611.48 | 640.14 | 981.02 | 0 % | 4184.3 |
| M1 classical EKF (no AI) | 28 | 130.65 | 431.86 | 1712.86 | 7 % | 979.3 |
| M1b hold last GNSS speed | 28 | 54.31 | 86.06 | 170.71 | 4 % | 414.8 |
| M2 + AI speed/ZUPT/R-adapter | 28 | 76.53 | 100.15 | 221.94 | 4 % | 597.1 |
| M3 + adaptive delayed course | 28 | 111.15 | 128.44 | 246.14 | 0 % | 767.0 |
| M4 + map-aided EKF (default) | 28 | 31.26 | 80.70 | 217.72 | 21 % | 176.5 |
| M5 DISHA v2 (selected+tuned) | 28 | 76.53 | 100.15 | 221.94 | 4 % | 597.1 |

## Problem-statement scenarios (M5)

| Journey | Scenario | Distance | End error | Target | Result |
|---|---|---|---|---|---|
| Vta16 | 50 m city | 50 m | 5.3 m | < 5 m | FAIL |
| Vta29 | 50 m city | 50 m | 124.5 m | < 5 m | FAIL |
| Vta29 | 1 km @ >=55 km/h | 1003 m | 1723.8 m | < 100 m | FAIL |
| Vtb1 | 1 km @ >=55 km/h | 999 m | 421.9 m | < 100 m | FAIL |
| Vtb1 | 50 m city | 51 m | 94.1 m | < 5 m | FAIL |
| Vw14c | 1 km @ >=55 km/h | 1002 m | 645.7 m | < 100 m | FAIL |
| Vw14c | 50 m city | 51 m | 9.1 m | < 5 m | FAIL |

AI speed (SpeedNetA) validation RMSE by anchor age [0, 10, 30, 60, 90]: [1.56, 2.16, 2.48, 2.57] m/s (hold-last-speed: [2.79, 5.33, 6.4, 6.69]); ZUPT accuracy 0.979
ONNX: single file verified vs PyTorch, max diff 5.7e-06; CPU latency 0.20 ms; 200 Hz engine 26,401 steps/s.
Selected configuration: course=legacy, map-aiding=False, road-bearing=False.