# The Project Explained (read this before the presentation)

This document explains **everything** in the project: what problem we solve, how each file works,
why each design choice was made, how to run it, what the results mean, and which questions the
teacher is likely to ask (with answers).

---

## 1. The problem in one paragraph

Traffic lights in most cities run **fixed-time plans**: every phase gets the same green time,
whatever the traffic. When traffic is unbalanced (a peak hour, an accident, a stadium emptying),
cars wait at red lights in front of empty crossroads. We want an **agent that observes the
intersection and decides, in real time, which direction gets the green light and for how long**,
so that the total waiting time goes down. We learn this decision rule with **reinforcement learning
(RL)**: the agent tries actions in a simulator, gets a reward (less waiting = better), and gradually
improves its policy.

---

## 2. Reinforcement learning, quickly

* An **agent** observes a **state** `s`, picks an **action** `a`, and the **environment** returns a
  **reward** `r` and a new state `s'`.
* The goal is to learn a **policy** `π(a|s)` that maximises the expected sum of discounted rewards
  `Σ γ^t r_t`.
* **PPO (Proximal Policy Optimization)** is a *policy-gradient, actor-critic* algorithm:
  * the **actor** is a neural network that outputs the probabilities of each action;
  * the **critic** is a neural network that estimates the value `V(s)` (how good a state is);
  * the **advantage** `A = (what actually happened) − V(s)` says whether an action was better or
    worse than expected. We compute it with **GAE** (Generalised Advantage Estimation), which mixes
    short-term and long-term estimates (parameter λ);
  * PPO increases the probability of actions with positive advantage, but **clips** the update
    (`ratio` limited to `[1−ε, 1+ε]`, ε = 0.2) so the policy never changes too much in one step.
    That is why PPO is stable and popular.
* **Why PPO and not DDPG?** DDPG is designed for *continuous* actions. Our actions are discrete
  (choose a phase + choose a duration from a list), and PPO handles discrete actions natively. It is
  also more stable and easier to tune. The brief allowed either one.
* **Epsilon-greedy:** the brief asks for an epsilon-greedy exploration strategy at the beginning.
  With probability ε the agent plays a *random* action instead of following its policy. ε starts
  at 0.3 and decreases linearly to 0 over the first 30 % of training (`PPOAgent.epsilon`). PPO
  already explores through its stochastic policy; epsilon-greedy adds extra exploration early on.
  *Detail you can mention:* the random actions are stored with their probability under the
  current policy, which introduces a small off-policy bias. PPO's clipping keeps it harmless, and the
  bias disappears once ε reaches 0.

---

## 3. Why a custom simulator instead of SUMO?

The brief says "SUMO **or** a custom environment". We chose a custom simulator because:

1. **No installation problems**: SUMO + TraCI are heavy to install and slow to couple with Python.
   Our simulator is pure Python and runs anywhere.
2. **Speed**: one hour of simulated traffic takes ≈ 0.05 s for one intersection, so we can train
   on hundreds of episodes in a few minutes on a laptop CPU.
3. **Full control & understanding**: we know exactly how queues, saturation flow and yellow
   time are modelled, and we can easily create scenarios (accident, event…).

The simulator still reproduces the important physics: queues building up at red lights,
**saturation flow** (one car crosses every 2 s on green), **lost time** during yellow,
**spill-back** (if the next road is full, cars cannot cross), and random turning.

**Limitation to admit:** it is less realistic than SUMO (no lane changes, no acceleration
profiles, one lane per approach, permissive left turns). The environment interface is generic,
so SUMO could replace the simulator later (see §10).

---

## 4. How the simulator works (`traffic_rl/envs/simulator.py`)

* The network is a **grid** of `rows × cols` intersections (1×1 = single intersection).
* Each intersection has **4 incoming lanes**: from North, East, South, West.
* Each lane has **25 cells of 7.5 m** (≈ 190 m). One cell holds one car. Cell 0 is the stop line.
* **Time step = 1 second.** Each second:
  1. **Signals update** (`_update_signals`): green timers count down. A yellow phase (3 s) is inserted
     when the phase changes.
  2. **Vehicles move** (`_move_lane`): a car advances up to `v_max = 2` cells/s (≈ 54 km/h) if the
     cells in front are free (a simplified **Nagel–Schreckenberg cellular automaton**). The first car
     can cross if: the light is green, 2 s have passed since the last crossing (saturation headway),
     and the downstream lane has room. After crossing it goes straight (70 %), turns right (15 %)
     or left (15 %), into the next intersection's incoming lane or out of the network.
  3. **Arrivals** (`_spawn`): on border lanes, a new car appears with probability λ(t) (a Poisson
     process in discrete time). If the entrance is full, it waits in a buffer, and that waiting is
     also counted.
* A car that does not move during a second is **waiting**: this feeds the waiting-time, queue and
  stops metrics.
* **CO₂ model** (simplified, realistic orders of magnitude for a petrol car): 0.19 g/s idling,
  1.2 g/s moving, +8 g each time a car restarts after a stop.

**Metrics** (`TrafficSimulator.metrics`):

| Metric | Meaning |
|---|---|
| `mean_waiting_time` | total vehicle-seconds of waiting / number of vehicles (s/veh) |
| `mean_travel_time` | average time from entering to leaving the network (s) |
| `mean_queue_length` | average number of waiting vehicles over the hour |
| `throughput` | number of vehicles that left the network |
| `stops_per_vehicle` | how many times each car had to stop |
| `co2_kg` | total CO₂ emitted |

---

## 5. Scenarios (`traffic_rl/envs/scenarios.py`)

| Name | Description | Used for training? |
|---|---|---|
| `low` | night traffic, 0.05 veh/s per entry | no (test) |
| `normal` | balanced, 0.14 veh/s per entry | yes |
| `unbalanced` | 0.22 veh/s North-South, 0.06 East-West | yes |
| `rush_hour` | Gaussian peak on North-South up to 0.26 veh/s in the middle of the hour | yes |
| `accident` | normal demand, but between minutes 20 and 40 the North approach of the first intersection has its capacity **divided by 3** | yes |
| `event` | end of a sports event: a strong, short surge from the West | **no: generalisation test** |

At each training episode a scenario is drawn at random, so the agent learns a **robust** policy
and does not specialise on one demand profile. `low` and `event` are never seen during training,
so the results on them measure **generalisation**.

---

## 6. The RL environment (`traffic_rl/envs/traffic_env.py`)

### State (55 numbers per intersection)
For each of the 4 incoming lanes:
* number of waiting vehicles (normalised by lane length): *"nombre de véhicules sur chaque voie"*;
* mean waiting time of the cars on the lane (/100 s): *"temps d'attente moyen"*;
* a 10-cell binary vector showing which cells near the stop line are occupied: *"position des
  véhicules dans la zone d'intersection"*.

Plus: occupancy of the 4 downstream lanes (to avoid sending cars into a full road), the current
phase (one-hot), and the time since the green started.

These are exactly the three state components required by the brief, plus a little context.

### Action (12 discrete actions)
`action = phase × 6 + k`. It chooses **which axis gets the green** (North-South or East-West) and the
**green duration** from {5, 10, 15, 20, 25, 30} s. This covers both actions in the brief:
*"changer l'état des feux"* and *"ajuster la durée des feux verts"*. Choosing the current phase
**extends** the green. Choosing the other phase triggers a **3 s yellow** first, so switching has a
cost and the agent must learn not to switch too often.

### Decision timing (semi-MDP)
The agent is not asked every second, only **when its green expires**. This is realistic (a
controller changes the plan at the end of a phase) and makes learning easier (≈ 200 decisions per
hour instead of 3600).

### Reward
Accumulated between two decisions:

```
r = − 0.01 × Σ_seconds (number of waiting vehicles)     # negative reward per second of waiting
    + 0.1  × (queue at decision − queue now)            # positive if the queue decreased
    − co2_coef × CO2 emitted                            # optional ecological term (0 by default)
```

This matches the brief: *"récompense négative pour chaque seconde d'attente cumulée"* +
*"récompense positive si le nombre de véhicules en file diminue"*. The CO₂ term implements the
"considérations écologiques" variant (`--co2-coef`).

### Multi-intersection
In a grid, **every intersection is an agent**. They decide **asynchronously** (each when its own
green ends). They all share **the same neural network** (*parameter sharing*): each one uses only
its local observation, so the same policy can run on a grid of any size. Downstream occupancy in
the state gives indirect coordination between neighbours.

### Gymnasium compatibility
`SingleIntersectionEnv` wraps the single intersection into a standard `gymnasium.Env`
(`Box` observation, `Discrete(12)` action). A test checks it with Gymnasium's official
`check_env`, so it can also be used with Stable-Baselines3 or RLlib.

---

## 7. The PPO agent (`traffic_rl/agents/ppo.py` and `traffic_rl/training.py`)

* **Networks**: actor and critic are separate MLPs, 55 → 128 → 128 → (12 or 1), tanh, orthogonal
  initialisation.
* **Rollout collection** (`train_ppo`): play full episodes (1 simulated hour). For each agent, store
  (state, action, log-probability, value, reward). Once ≥ 2048 transitions are collected, compute
  GAE per trajectory and run an update.
* **Truncation**: an episode stops because of the 1-hour time limit, not because the task is
  "finished". So we **bootstrap** the last value `V(s_T)` instead of assuming 0. This is a
  frequent mistake that we avoided.
* **Update** (`PPOAgent.update`): 8 epochs, minibatches of 256, normalised advantages, loss =
  clipped policy loss + 0.5 × value loss − 0.01 × entropy, gradient clipping at 0.5, Adam with
  lr = 3e-4.
* **Checkpointing**: `ppo_best.pt` is the model with the best 20-episode moving average of waiting
  time. `ppo_final.pt` is the last one.

Hyper-parameters (defaults in `PPOConfig`):

| Parameter | Value | Role |
|---|---|---|
| lr | 3e-4 | learning rate |
| γ (gamma) | 0.98 | discount per decision (≈ 15 s), horizon of a few minutes |
| λ (GAE) | 0.95 | bias/variance trade-off of the advantage |
| clip ε | 0.2 | maximum policy change per update |
| entropy coef | 0.01 | keeps exploration alive |
| rollout | 2048 transitions | batch size per update |
| epsilon-greedy | 0.3 → 0 over 30 % of training | early exploration |

---

## 8. Baselines (`traffic_rl/agents/baselines.py`)

A result means nothing without comparison. We compare PPO with:

* **Fixed-time**: alternate phases, 25 s green each. This is what most cities do.
* **Longest-queue (actuated)**: serve the axis with the most waiting cars, with green ≈ 2 s per car
  in the longest queue. A strong hand-written heuristic.
* **Max-pressure** (Varaiya 2013): serve the phase with the largest (upstream − downstream) queue
  difference, 10 s green. A classic algorithm from the literature.
* **Random**: a sanity check (lower bound).

---

## 9. Results and how to read them

Training: 600 episodes (single intersection, ≈ 5 min CPU) and 400 episodes (2×2 network, ≈ 12 min).
Evaluation: 5 seeds × 6 scenarios × 1 hour, greedy policy (no exploration).

**Training curve** (`results/single/training_curve.png`): the mean waiting time drops from
≈ 70 s (random-like behaviour at the start) to ≈ 5–8 s. The red dashed line is ε, which decays to 0
at episode 180. The curve is noisy because each episode uses a different random scenario.

**Main table: mean waiting time (s/veh)**

| Scenario | Fixed | Longest-queue | Max-pressure | PPO | PPO vs fixed |
|---|---|---|---|---|---|
| *Single intersection* |
| low | 8.75 | 3.46 | 4.91 | 3.51 | −60 % |
| normal | 10.82 | 5.79 | 6.65 | **5.38** | −50 % |
| unbalanced | 25.35 | 5.28 | 10.61 | **4.23** | −83 % |
| rush_hour | 21.06 | 5.54 | 13.36 | **4.92** | −77 % |
| accident | 35.26 | **17.42** | 18.28 | 19.39 | −45 % |
| event | 28.62 | 6.43 | 13.49 | **5.04** | −82 % |
| *2×2 network* |
| low | 15.38 | 6.92 | 9.63 | 6.95 | −55 % |
| normal | 18.71 | 11.50 | 12.80 | **10.10** | −46 % |
| unbalanced | 37.75 | 11.24 | 17.95 | **9.06** | −76 % |
| rush_hour | 32.60 | 12.14 | 18.98 | **10.09** | −69 % |
| accident | 33.01 | 19.83 | 22.33 | **17.13** | −48 % |
| event | 35.91 | 12.18 | 20.49 | **10.05** | −72 % |

**What to say about it:**
1. PPO divides waiting time by **2 to 6** compared with fixed-time plans, in every scenario.
2. PPO is **better than all baselines** in 4/6 scenarios on one intersection and 5/6 on the
   network, and practically tied in the others (low traffic: everyone does well).
3. **Generalisation**: on `event`, which was never seen during training, PPO is still the best
   (−82 % / −72 %).
4. **Honest weakness**: during the accident on a single intersection, the longest-queue heuristic is
   slightly better (17.4 s vs 19.4 s). The accident makes one lane very slow, and a simple
   "serve the longest queue" rule happens to fit that well. On the network, PPO wins the accident
   scenario.
5. **Network effect**: on the 2×2 grid, the gap between PPO and the heuristics is bigger. The
   downstream-occupancy features help avoid sending cars into full roads.
6. **Ecology**: fewer stops means less CO₂: −15 % to −68 % vs fixed-time (single intersection),
   −12 % to −54 % (network), without even putting CO₂ in the reward.

Other figures in `results/single/` and `results/grid2x2/`:
* `eval_*.png`: grouped bar chart per metric (waiting, travel time, queue, throughput, stops, CO₂);
* `queue_rush_hour.png` / `queue_accident.png`: number of waiting cars over time. During the rush
  hour, fixed-time reaches 30 waiting cars while PPO stays under ~8. **This is the best figure for
  the slides.**
* `snapshot_rush_hour.png`: picture of the intersection (green/red lights, cars, red = stopped).

**Hyper-parameter tuning** (`results/tuning/`): 12 random configurations × 150 episodes.
The best one (lr 3e-4, γ 0.98, clip 0.3, entropy 0.03, ε-start 0.3) scored 9.6 s, close to our
defaults. Lessons:
* **PPO is robust here**: 9 of the 12 configurations land between 9.6 and 12.5 s, and our default
  settings sit in that band;
* three runs were unstable (19–29 s). Two of them had no epsilon-greedy (trials 10, 11), but three
  other runs without epsilon-greedy were fine (11.2–11.8 s). With 150 episodes and one seed per
  trial, the search **can't isolate** the effect of a single hyper-parameter. Say so honestly if
  asked: a proper study would need several seeds per configuration.

---

## 10. Limitations and possible extensions

| Limitation | Possible extension |
|---|---|
| Simplified simulator (1 lane, no acceleration model) | Plug SUMO through TraCI. The environment API (`reset`/`step`, metrics) stays the same |
| 2 phases (no protected left turn) | Add 4-phase plans: just extend `PHASES` |
| Only cars | **Multimodal variant**: add pedestrians (a pedestrian phase) and buses (priority weight in the reward) |
| Independent agents with shared weights | Explicit coordination: neighbours' observations, graph neural networks, MAPPO with a centralised critic |
| CO₂ only measured | Train with `--co2-coef > 0` to optimise emissions directly |
| 1 training seed | Several seeds + confidence intervals |

Real applications: smart cities (adaptive systems like SCOOT/SCATS, now RL pilots), crisis
management (evacuations, sports events: our `event` scenario).

---

## 11. Likely questions from the jury (and answers)

**Q: Why PPO rather than DQN?**
DQN would work with our discrete actions, but the brief suggested PPO/DDPG. PPO is on-policy,
stable thanks to clipping, handles discrete actions natively, and combines well with parameter
sharing for multi-agent control.

**Q: What does the clipping do?**
It limits the ratio π_new/π_old to [0.8, 1.2]. If an update would move the policy further, the
gradient is cut, which prevents catastrophic policy collapses.

**Q: Why do you need epsilon-greedy if PPO is stochastic?**
At the start the policy is almost uniform anyway, but it can quickly become over-confident in
bad actions (e.g. always 30 s). Forcing ε % random actions early guarantees all durations and
phases are tried. It is also what the brief asked for. (Our short tuning runs don't prove it is
necessary: some runs without it also did well.)

**Q: How do you avoid the agent switching lights all the time?**
Switching costs 3 s of yellow during which nobody crosses. This lost time increases waiting,
so the reward itself penalises useless switching. The agent learned to extend the green when the
queue is still discharging.

**Q: Is the result statistically meaningful?**
Every number is an average over 5 different random seeds (different arrival sequences) and one hour of
traffic each, with the same seeds for every controller, so the comparison is fair. The gaps vs
fixed-time (50–80 %) are much larger than seed-to-seed variability (`eval_metrics.csv` has the
per-seed values).

**Q: How does it scale to a city?**
Thanks to parameter sharing, the same network controls any number of intersections (input size is
per intersection). The 2×2 grid uses the exact same code as the single intersection. A 5×5
grid only needs `--rows 5 --cols 5`.

**Q: Why is the reward split in two terms?**
The waiting term is the real objective. The queue-difference term gives a faster, denser signal
("did my last decision reduce the queue?"), which speeds up learning.

**Q: What is GAE?**
A weighted average of n-step advantage estimates: `A_t = Σ (γλ)^l δ_{t+l}` with
`δ_t = r_t + γV(s_{t+1}) − V(s_t)`. λ = 0 is low variance but biased, λ = 1 is unbiased but noisy.
0.95 is the standard compromise.

---

## 12. Commands cheat-sheet

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 14 tests
python scripts/train.py --episodes 600 --out results/single
python scripts/evaluate.py --model results/single/ppo_best.pt --out results/single
python scripts/train.py --rows 2 --cols 2 --episodes 400 --out results/grid2x2
python scripts/evaluate.py --model results/grid2x2/ppo_best.pt --rows 2 --cols 2 --out results/grid2x2 --no-random
python scripts/tune.py --trials 12 --episodes 150
python scripts/train.py --co2-coef 0.001 --out results/eco   # ecological variant
```

## 13. File map

| File | What is inside |
|---|---|
| `traffic_rl/envs/simulator.py` | vehicles, lanes, signals, movement rules, metrics, CO₂ |
| `traffic_rl/envs/scenarios.py` | the 6 demand scenarios + accident incident |
| `traffic_rl/envs/traffic_env.py` | state / action / reward, multi-agent stepping, Gymnasium wrapper |
| `traffic_rl/agents/ppo.py` | actor-critic, epsilon-greedy, GAE, PPO update, save/load |
| `traffic_rl/agents/baselines.py` | fixed-time, longest-queue, max-pressure, random |
| `traffic_rl/training.py` | training loop, evaluation, CSV writing |
| `traffic_rl/utils/plotting.py` | all figures |
| `scripts/*.py` | command-line entry points |
| `tests/test_env.py` | API compliance, vehicle conservation, red light blocks, PPO save/load… |
| `results/` | trained models (`ppo_best.pt`), logs, tables, figures |
