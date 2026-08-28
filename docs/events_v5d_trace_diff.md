# v5d 台本 — 兆候の before/after（種類×取引種別）

`tools/build_events_v5d.py` が機械生成する。取得46件（順番・月・区画・名義・kind・note）は v5c と完全に同一で、変わったのは traces だけである。

| 種類 | 取引種別 | v5c | v5d | 差 |
|---|---|---|---|---|
| registry | sale | 33 | 0 | -33 |
| construction | sale | 8 | 0 | -8 |
| construction | lease | 1 | 0 | -1 |
| sign_change | sale | 6 | 0 | -6 |
| sign_change | lease | 7 | 0 | -7 |
| tenant_swap | lease | 8 | 8 | +0 |
| moving_out | sale | 5 | 4 | -1 |
| moving_out | lease | 2 | 0 | -2 |
| survey | sale | 8 | 8 | +0 |
| survey | lease | 1 | 0 | -1 |
| strangers | sale | 6 | 6 | +0 |
| strangers | lease | 1 | 0 | -1 |
| broker_known | sale | 5 | 5 | +0 |
| broker_known | lease | 2 | 2 | +0 |
| **合計** | — | **93** | **33** | **-60** |

取得件数 46（v5c と同一）／兆候ゼロの取得 18件
