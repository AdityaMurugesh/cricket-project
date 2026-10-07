# run10 results

110 evaluated checkpoints.

## Headline per budget

| budget | best final policy, legal in-zone (det) | at angle | best checkpoint | at angle | scripted ceiling | at angle |
|---|---|---|---|---|---|---|
| 0.75x | 34.2 km/h | 265 | 34.0 km/h | 265 | 34.8 km/h | 267.5 |
| 1.00x | 33.2 km/h | 265 | 41.9 km/h | 270 | 43.3 km/h | 275.0 |
| 1.50x | 54.9 km/h | 275 | 55.8 km/h | 275 | 56.5 km/h | 280.0 |

## Found the zone, then lost it

21 of 110 runs reached >=30% legal-and-in-zone during training and ended below 10% (stochastic, training-time episodes).

| run | peak good% | at step | final good% | final det |
|---|---|---|---|---|
| a230_s075_e0003_0 | 30 | 0.8M | 0 | 19.2 km/h, lands 3.8 m |
| a240_s075_e0003_0 | 52 | 0.8M | 0 | 19.4 km/h, lands 4.0 m |
| a240_s075_e0003_2 | 39 | 0.7M | 0 | 19.3 km/h, lands 4.0 m |
| a250_s075_e0003_0 | 69 | 0.7M | 9 | 26.4 km/h, lands 6.0 m |
| a250_s075_e0003_1 | 99 | 2.1M | 0 | 24.7 km/h, lands 5.4 m |
| a250_s075_e0003_2 | 61 | 0.7M | 0 | 22.9 km/h, lands 4.9 m |
| a260_s075_e0003_1 | 98 | 1.5M | 0 | 25.2 km/h, lands 5.0 m |
| a230_s100_e0003_1 | 45 | 0.7M | 1 | 23.1 km/h, lands 5.2 m |
| a230_s100_e0003_2 | 49 | 1.6M | 0 | 25.3 km/h, lands 6.0 m |
| a240_s100_e0003_1 | 46 | 0.8M | 2 | 23.8 km/h, lands 5.5 m |
| a240_s100_e0003_2 | 39 | 0.7M | 0 | 23.0 km/h, lands 5.2 m |
| a250_s100_e0003_2 | 73 | 3.1M | 7 | 25.1 km/h, lands 5.6 m |
| a230_s150_e0003_0 | 72 | 2.1M | 2 | no release |
| a230_s150_e0003_1 | 64 | 2.9M | 0 | no release |
| a230_s150_e0003_2 | 57 | 1.6M | 0 | 21.8 km/h, lands 4.7 m |
| a240_s150_e0003_0 | 77 | 2.5M | 0 | 29.7 km/h, lands 7.9 m |
| a240_s150_e0003_1 | 70 | 3.2M | 0 | 26.0 km/h, lands 6.4 m |
| a240_s150_e0003_2 | 44 | 1.6M | 0 | 31.1 km/h, lands 8.6 m |
| a265_s150_e0003_2 | 90 | 3.1M | 2 | 34.4 km/h, lands 6.3 m |
| a270_s150_e0003_0 | 100 | 4.8M | 2 | 44.1 km/h, lands 8.0 m |
| a270_s150_e0003_2 | 100 | 3.2M | 5 | 38.9 km/h, lands 6.5 m |

## Full grid

ent_coef=0.003. Deterministic policy per seed: speed (km/h) if the delivery was legal AND in the 6-8 m zone (bold), else why not. Best ckpt = fastest legal in-zone deterministic delivery among the periodic checkpoints (runs that saved them). good% = stochastic, 200 episodes, final policy.

| budget | angle | scripted ceiling | seed 0 | seed 1 | seed 2 | best ckpt | good% s0 / s1 / s2 |
|---|---|---|---|---|---|---|---|
| 0.75x | 230 | 29.6 | 19.2 (illegal) | 19.6 (lands 4.0 m) | 18.9 (illegal) | -- | 0 / 0 / 0 |
| 0.75x | 240 | 30.1 | 19.4 (lands 4.0 m) | **28.3** | 19.3 (lands 4.0 m) | 26.3 | 0 / 42 / 0 |
| 0.75x | 250 | 32.3 | **26.4** | 24.7 (lands 5.4 m) | 22.9 (lands 4.9 m) | 29.3 | 48 / 0 / 0 |
| 0.75x | 260 | 33.7 | **33.7** | 25.2 (lands 5.0 m) | 28.7 (lands 5.8 m) | 33.5 | 100 / 0 / 2 |
| 0.75x | 265 | 34.4 | **34.2** | **32.2** | 30.4 (lands 5.7 m) | 34.0 | 100 / 60 / 12 |
| 0.75x | 270 | unreachable | 33.4 (lands 5.5 m) | no release | no release | -- | 0 / 0 / 0 |
| 0.75x | 275 | unreachable | no release | no release | no release | -- | 0 / 0 / 0 |
| 0.75x | 280 | unreachable | no release | no release | 34.6 (lands 4.4 m) | -- | 0 / 0 / 0 |
| 0.75x | 285 | unreachable | no release | no release | no release | -- | 0 / 0 / 0 |
| 0.75x | 290 | unreachable | no release | no release | no release | -- | 0 / 0 / 0 |
| 0.75x | 300 | unreachable | no release | no release | no release | -- | 0 / 0 / 0 |
| 1.00x | 230 | 29.6 | 23.7 (illegal) | 23.1 (lands 5.2 m) | 25.3 (illegal) | 27.2 | 0 / 0 / 0 |
| 1.00x | 240 | 30.1 | no release | 23.8 (lands 5.5 m) | 23.0 (illegal) | 25.2 | 0 / 2 / 0 |
| 1.00x | 250 | 32.3 | 29.8 (illegal) | **26.6** | 25.1 (illegal) | 29.0 | 26 / 60 / 2 |
| 1.00x | 260 | 37.1 | **31.7** | **30.6** | 28.0 (lands 5.7 m) | 31.7 | 80 / 83 / 18 |
| 1.00x | 265 | 42.0 | no release | **33.0** | **33.2** | 40.8 | 0 / 100 / 100 |
| 1.00x | 270 | 42.7 | no release | no release | 35.3 (lands 5.8 m) | 41.9 | 0 / 0 / 5 |
| 1.00x | 275 | 43.3 | no release | no release | no release | -- | 0 / 0 / 0 |
| 1.00x | 280 | unreachable | no release | no release | no release | -- | 0 / 0 / 0 |
| 1.00x | 285 | unreachable | no release | no release | no release | -- | 0 / 0 / 0 |
| 1.00x | 290 | unreachable | 35.9 (illegal) | no release | 45.0 (lands 3.8 m) | -- | 0 / 0 / 0 |
| 1.00x | 300 | unreachable | 45.6 (lands 2.8 m) | no release | no release | -- | 0 / 0 / 0 |
| 1.50x | 230 | 29.6 | no release | no release | 21.8 (lands 4.7 m) | 28.8 | 0 / 0 / 0 |
| 1.50x | 240 | 30.1 | 29.7 (illegal) | 26.0 (illegal) | 31.1 (illegal) | 27.3 | 0 / 0 / 0 |
| 1.50x | 250 | 32.3 | **28.9** | **27.6** | **28.5** | 30.7 | 98 / 90 / 89 |
| 1.50x | 260 | 36.8 | 38.4 (lands 8.5 m) | **30.3** | 27.7 (lands 5.6 m) | 34.2 | 64 / 69 / 16 |
| 1.50x | 265 | 41.5 | **33.0** | **34.8** | 34.4 (illegal) | 38.4 | 84 / 99 / 6 |
| 1.50x | 270 | 49.5 | 44.1 (illegal) | **38.8** | 38.9 (illegal) | 48.4 | 0 / 82 / 1 |
| 1.50x | 275 | 55.5 | no release | no release | **54.9** | 55.8 | 0 / 0 / 100 |
| 1.50x | 280 | 56.5 | 51.9 (illegal) | no release | no release | -- | 0 / 0 / 0 |
| 1.50x | 285 | unreachable | no release | 57.5 (lands 5.1 m) | no release | -- | 0 / 0 / 0 |
| 1.50x | 290 | unreachable | no release | no release | no release | -- | 0 / 0 / 0 |
| 1.50x | 300 | unreachable | no release | 59.2 (lands 3.1 m) | no release | -- | 0 / 0 / 0 |

## Control column, ent_coef=0.01 (seed 0 only)

ent_coef=0.01. Deterministic policy per seed: speed (km/h) if the delivery was legal AND in the 6-8 m zone (bold), else why not. Best ckpt = fastest legal in-zone deterministic delivery among the periodic checkpoints (runs that saved them). good% = stochastic, 200 episodes, final policy.

| budget | angle | scripted ceiling | seed 0 | best ckpt | good% s0 |
|---|---|---|---|---|---|
| 1.00x | 230 | 29.6 | 22.6 (lands 5.0 m) | 26.2 | 3 |
| 1.00x | 240 | 30.1 | no release | -- | 0 |
| 1.00x | 250 | 32.3 | **27.9** | 30.9 | 49 |
| 1.00x | 260 | 37.1 | no release | -- | 0 |
| 1.00x | 265 | 42.0 | no release | -- | 0 |
| 1.00x | 270 | 42.7 | **41.1** | 42.7 | 100 |
| 1.00x | 275 | 43.3 | no release | -- | 0 |
| 1.00x | 280 | unreachable | no release | -- | 0 |
| 1.00x | 285 | unreachable | no release | -- | 0 |
| 1.00x | 290 | unreachable | no release | -- | 0 |
| 1.00x | 300 | unreachable | no release | -- | 0 |
