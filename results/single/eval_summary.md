# Evaluation (1x1 grid, 5 seeds per scenario)


## Mean waiting time (s/veh)

| scenario | ppo | fixed_time | longest_queue | max_pressure | random | PPO vs fixed-time |
|---|---|---|---|---|---|---|
| low | 3.51 | 8.75 | 3.46 | 4.91 | 17.47 | -59.9% |
| normal | 5.38 | 10.82 | 5.79 | 6.65 | 27.49 | -50.2% |
| unbalanced | 4.23 | 25.35 | 5.28 | 10.61 | 116.96 | -83.3% |
| rush_hour | 4.92 | 21.06 | 5.54 | 13.36 | 64.52 | -76.6% |
| accident | 19.39 | 35.26 | 17.42 | 18.28 | 47.62 | -45.0% |
| event | 5.04 | 28.62 | 6.43 | 13.49 | 47.85 | -82.4% |

## Mean travel time (s)

| scenario | ppo | fixed_time | longest_queue | max_pressure | random | PPO vs fixed-time |
|---|---|---|---|---|---|---|
| low | 16.68 | 22.10 | 16.62 | 18.15 | 31.04 | -24.5% |
| normal | 18.99 | 24.94 | 19.47 | 20.41 | 42.04 | -23.9% |
| unbalanced | 17.70 | 41.79 | 19.09 | 25.17 | 135.66 | -57.6% |
| rush_hour | 18.46 | 36.63 | 19.25 | 28.22 | 81.97 | -49.6% |
| accident | 34.08 | 50.69 | 32.05 | 33.02 | 64.19 | -32.8% |
| event | 18.57 | 44.00 | 20.36 | 28.27 | 63.85 | -57.8% |

## Mean number of waiting vehicles

| scenario | ppo | fixed_time | longest_queue | max_pressure | random | PPO vs fixed-time |
|---|---|---|---|---|---|---|
| low | 0.71 | 1.75 | 0.69 | 0.98 | 3.49 | -59.6% |
| normal | 3.01 | 6.04 | 3.25 | 3.71 | 15.38 | -50.2% |
| unbalanced | 2.34 | 14.24 | 2.95 | 5.86 | 64.80 | -83.5% |
| rush_hour | 2.28 | 9.64 | 2.54 | 6.15 | 29.64 | -76.4% |
| accident | 10.84 | 19.66 | 9.79 | 10.23 | 26.67 | -44.8% |
| event | 2.02 | 11.53 | 2.60 | 5.42 | 19.28 | -82.4% |

## Vehicles served

| scenario | ppo | fixed_time | longest_queue | max_pressure | random | PPO vs fixed-time |
|---|---|---|---|---|---|---|
| low | 719.00 | 714.80 | 711.80 | 717.60 | 712.40 | +0.6% |
| normal | 2002.40 | 1999.00 | 2007.60 | 1995.60 | 1976.00 | +0.2% |
| unbalanced | 1983.60 | 1984.40 | 1997.20 | 1974.20 | 1920.60 | -0.0% |
| rush_hour | 1661.20 | 1636.60 | 1644.20 | 1646.60 | 1648.00 | +1.5% |
| accident | 2000.00 | 1991.60 | 2011.40 | 2006.00 | 1989.00 | +0.4% |
| event | 1437.60 | 1442.00 | 1442.40 | 1436.80 | 1441.80 | -0.3% |

## Stops per vehicle

| scenario | ppo | fixed_time | longest_queue | max_pressure | random | PPO vs fixed-time |
|---|---|---|---|---|---|---|
| low | 0.66 | 1.03 | 0.82 | 0.90 | 1.53 | -35.6% |
| normal | 1.53 | 2.47 | 1.71 | 1.83 | 4.84 | -38.0% |
| unbalanced | 1.20 | 7.29 | 1.94 | 3.39 | 13.93 | -83.5% |
| rush_hour | 1.39 | 5.36 | 1.80 | 4.01 | 9.03 | -74.0% |
| accident | 3.56 | 4.79 | 3.49 | 3.66 | 6.74 | -25.8% |
| event | 1.37 | 4.94 | 2.23 | 3.88 | 5.82 | -72.2% |

## CO2 emitted (kg)

| scenario | ppo | fixed_time | longest_queue | max_pressure | random | PPO vs fixed-time |
|---|---|---|---|---|---|---|
| low | 15.69 | 18.58 | 16.46 | 17.26 | 22.88 | -15.5% |
| normal | 59.50 | 77.78 | 62.84 | 64.86 | 124.86 | -23.5% |
| unbalanced | 52.89 | 167.03 | 66.38 | 92.53 | 313.19 | -68.3% |
| rush_hour | 47.17 | 107.72 | 52.57 | 86.70 | 174.03 | -56.2% |
| accident | 99.85 | 126.89 | 98.39 | 101.33 | 165.78 | -21.3% |
| event | 40.66 | 91.61 | 51.80 | 74.11 | 108.01 | -55.6% |
