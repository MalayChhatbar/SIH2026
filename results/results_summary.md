# SIH 26168 - IDR MVP results summary (IO-VNBD, held-out journeys)

| Journey | Method | 60 s blackout drift (median) | drift % (median) | p90 drift % |
|---|---|---|---|---|
| Vta16 | naive integrate |a| (IMU only) | 3423.4 m | 502.54 % | 603.67 % |
| Vta16 | EKF, no AI modules | 121.3 m | 31.89 % | 46.90 % |
| Vta16 | AI-DR (IDR) | 1122.4 m | 170.49 % | 217.32 % |
| Vta16 | AI-DR + map matching | 1070.2 m | 173.24 % | - |
| Vta29 | naive integrate |a| (IMU only) | 5492.9 m | 680.43 % | 1072.59 % |
| Vta29 | EKF, no AI modules | 543.8 m | 64.32 % | 146.80 % |
| Vta29 | AI-DR (IDR) | 848.3 m | 143.70 % | 209.73 % |
| Vta29 | AI-DR + map matching | 813.2 m | 146.65 % | - |
| Vtb1 | naive integrate |a| (IMU only) | 4161.5 m | 507.77 % | 718.73 % |
| Vtb1 | EKF, no AI modules | 491.8 m | 79.18 % | 164.48 % |
| Vtb1 | AI-DR (IDR) | 1241.4 m | 131.39 % | 202.16 % |
| Vtb1 | AI-DR + map matching | 1377.8 m | 127.94 % | - |
| Vw14c | naive integrate |a| (IMU only) | 3425.3 m | 408.23 % | 476.32 % |
| Vw14c | EKF, no AI modules | 1282.3 m | 207.23 % | 313.55 % |
| Vw14c | AI-DR (IDR) | 806.8 m | 103.86 % | 141.87 % |
| Vw14c | AI-DR + map matching | 870.1 m | 117.67 % | - |

## Problem-statement scenario checks

| Journey | Scenario | Distance | AI-DR drift | Target | Result |
|---|---|---|---|---|---|
| Vta16 | 50m-city | 50 m | 248.2 m | < 5 m | FAIL |
| Vta29 | 1km@60kmph | 1003 m | 978.1 m | < 100 m | FAIL |
| Vta29 | 50m-city | 50 m | 22.5 m | < 5 m | FAIL |
| Vw14c | 1km@60kmph | 1002 m | 486.1 m | < 100 m | FAIL |
| Vw14c | 50m-city | 50 m | 136.3 m | < 5 m | FAIL |

Full-journey fusion error (no blackout): Vta16 median 119.50 m, Vta29 median 56.92 m, Vtb1 median 46.70 m, Vw14c median 115.01 m
AI speed model (SpeedNet): val RMSE 2.89 m/s, ZUPT accuracy > 97 %
