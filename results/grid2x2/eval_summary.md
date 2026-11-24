# Evaluation (2x2 grid, 5 seeds per scenario)


## Mean waiting time (s/veh)

| scenario | ppo | fixed_time | longest_queue | max_pressure | PPO vs fixed-time |
|---|---|---|---|---|---|
| low | 6.95 | 15.38 | 6.92 | 9.63 | -54.8% |
| normal | 10.10 | 18.71 | 11.50 | 12.80 | -46.0% |
| unbalanced | 9.06 | 37.75 | 11.24 | 17.95 | -76.0% |
| rush_hour | 10.09 | 32.60 | 12.14 | 18.98 | -69.0% |
| accident | 17.13 | 33.01 | 19.83 | 22.33 | -48.1% |
| event | 10.05 | 35.91 | 12.18 | 20.49 | -72.0% |

## Mean travel time (s)

| scenario | ppo | fixed_time | longest_queue | max_pressure | PPO vs fixed-time |
|---|---|---|---|---|---|
| low | 33.15 | 41.88 | 33.19 | 36.02 | -20.9% |
| normal | 37.08 | 46.31 | 38.72 | 40.15 | -19.9% |
| unbalanced | 36.00 | 68.11 | 38.71 | 46.24 | -47.1% |
| rush_hour | 37.21 | 62.13 | 39.54 | 47.30 | -40.1% |
| accident | 44.70 | 61.49 | 47.65 | 50.50 | -27.3% |
| event | 36.92 | 64.76 | 39.60 | 49.04 | -43.0% |

## Mean number of waiting vehicles

| scenario | ppo | fixed_time | longest_queue | max_pressure | PPO vs fixed-time |
|---|---|---|---|---|---|
| low | 2.77 | 6.14 | 2.76 | 3.84 | -54.9% |
| normal | 11.28 | 20.91 | 12.86 | 14.25 | -46.0% |
| unbalanced | 10.16 | 42.69 | 12.50 | 19.98 | -76.2% |
| rush_hour | 9.32 | 30.07 | 11.24 | 17.48 | -69.0% |
| accident | 19.09 | 36.80 | 22.03 | 25.01 | -48.1% |
| event | 8.10 | 28.85 | 9.75 | 16.61 | -71.9% |

## Vehicles served

| scenario | ppo | fixed_time | longest_queue | max_pressure | PPO vs fixed-time |
|---|---|---|---|---|---|
| low | 1421.60 | 1421.20 | 1422.60 | 1419.00 | +0.0% |
| normal | 3976.00 | 3968.00 | 3979.00 | 3964.60 | +0.2% |
| unbalanced | 3991.60 | 3977.40 | 3957.80 | 3949.80 | +0.4% |
| rush_hour | 3301.40 | 3295.00 | 3306.60 | 3291.20 | +0.2% |
| accident | 3970.40 | 3955.60 | 3960.00 | 3975.60 | +0.4% |
| event | 2880.20 | 2857.20 | 2857.80 | 2889.80 | +0.8% |

## Stops per vehicle

| scenario | ppo | fixed_time | longest_queue | max_pressure | PPO vs fixed-time |
|---|---|---|---|---|---|
| low | 1.22 | 1.68 | 1.66 | 1.72 | -27.3% |
| normal | 2.84 | 3.72 | 3.35 | 3.48 | -23.7% |
| unbalanced | 2.69 | 9.46 | 3.72 | 5.45 | -71.6% |
| rush_hour | 2.99 | 7.60 | 3.81 | 5.53 | -60.7% |
| accident | 3.77 | 5.00 | 4.55 | 4.94 | -24.5% |
| event | 2.60 | 6.00 | 3.82 | 5.73 | -56.6% |

## CO2 emitted (kg)

| scenario | ppo | fixed_time | longest_queue | max_pressure | PPO vs fixed-time |
|---|---|---|---|---|---|
| low | 60.74 | 68.79 | 65.96 | 67.43 | -11.7% |
| normal | 228.16 | 265.72 | 247.13 | 251.72 | -14.1% |
| unbalanced | 223.24 | 482.99 | 258.60 | 322.79 | -53.8% |
| rush_hour | 193.48 | 338.89 | 218.17 | 270.70 | -42.9% |
| accident | 265.47 | 319.85 | 292.93 | 310.56 | -17.0% |
| event | 159.02 | 256.81 | 188.93 | 244.05 | -38.1% |
