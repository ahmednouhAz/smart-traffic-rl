# Prompt for generating the presentation

> **How to use:** copy everything below the line into Claude. For the best result, also attach
> these figures from the repository (drag & drop):
> `results/single/queue_rush_hour.png`, `results/single/training_curve.png`,
> `results/single/eval_mean_waiting_time.png`, `results/grid2x2/eval_mean_waiting_time.png`,
> `results/single/snapshot_rush_hour.png`, `results/grid2x2/snapshot_rush_hour.png`,
> `results/single/eval_co2_kg.png`.
> The slides are requested in **French** (the course language). Replace "French" with "English" in
> the prompt if you need English slides.

---

You are an expert in reinforcement learning and in academic presentation design. Create a
**professional slide deck (14–16 slides) in French** for a 15-minute oral defence of an engineering
school project. Audience: the teacher (Dr. Olfa Ferchichi) and classmates of the 5th-year Data
Science engineering program, module "Renforcement Learning et Optimisation". The tone should be
clear, technical but accessible, and visual. Use little text per slide: short bullet points, big
numbers, diagrams, and the attached figures. Add **speaker notes** to every slide (what to say,
≈ 1 minute each).

Use a clean, modern design: a white or very light background, one accent colour (blue #2a6fdb, the
colour of PPO in our charts), green/red for traffic lights. Use consistent icons. Do not overload
the slides.

## Project facts (use only these facts and numbers; do not invent results)

**Title:** "Feux de circulation intelligents par apprentissage par renforcement (PPO)". Projet 5,
Optimisation d'un réseau de transport intelligent.

**Problem:** fixed-time traffic lights ignore real traffic → cars wait at red lights while the
crossroad is empty, causing congestion, lost time and CO₂. Goal: an agent that adjusts the lights
dynamically to minimise the overall waiting time.

**Environment (custom simulator, Python, no SUMO needed):**
- grid of intersections (1×1 and 2×2 tested); each intersection has 4 incoming lanes (N, E, S, W)
  of 25 cells × 7.5 m; time step 1 s;
- cellular automaton (simplified Nagel–Schreckenberg): cars advance up to 2 cells/s (~54 km/h);
  saturation flow 1 car / 2 s on green; 3 s yellow when switching; spill-back; random turns
  (70 % straight, 15 % right, 15 % left); Poisson arrivals;
- simplified CO₂ model: 0.19 g/s idling, 1.2 g/s moving, +8 g per restart;
- why custom instead of SUMO: portable, very fast (1 simulated hour ≈ 0.05 s), full control over
  scenarios; the API is generic so SUMO could be plugged in later.

**Scenarios:** low (night), normal (0.14 veh/s per entry), unbalanced (heavy N-S), rush_hour
(Gaussian peak on N-S), accident (capacity of one approach ÷3 between minutes 20 and 40), event
(surge from the West after a sports event). Training uses a random mix of normal / unbalanced /
rush_hour / accident. **low and event are never seen in training (generalisation test).**

**RL formulation (semi-MDP, the agent decides when its green expires):**
- State (55 features per intersection): for each lane, the number of waiting cars, the mean
  waiting time, and the occupancy of the 10 cells nearest the stop line (vehicle positions); plus
  downstream lane occupancy, the current phase and the elapsed green time.
- Action (12 discrete): which axis is green (N-S or E-W) × green duration {5, 10, 15, 20, 25, 30} s.
  Keeping the phase extends the green; switching inserts a 3 s yellow.
- Reward: −0.01 × (vehicle-seconds of waiting) + 0.1 × (queue before − queue after)
  [+ optional CO₂ penalty for the ecological variant].

**Algorithm: PPO implemented from scratch in PyTorch**
- actor & critic MLP 2×128 tanh; GAE (γ = 0.98, λ = 0.95); clipped objective ε = 0.2; 8 epochs,
  minibatch 256, rollout 2048 transitions, entropy 0.01, lr 3e-4;
- **epsilon-greedy exploration** 0.3 → 0 over the first 30 % of training;
- time-limit truncation handled with value bootstrapping;
- **multi-intersection: parameter sharing**, so each intersection is an agent that uses the same
  network with local observations, acting asynchronously; it scales to any grid size;
- why PPO over DDPG: discrete actions, stability thanks to clipping.

**Baselines:** fixed-time (25 s each phase), longest-queue (actuated heuristic), max-pressure
(Varaiya 2013), random.

**Training:** single intersection, 600 episodes of 1 simulated hour (≈ 5 min on a laptop CPU);
2×2 network, 400 episodes (≈ 12 min). Mean waiting time during training: ≈ 70 s → ≈ 5–8 s.

**Results: mean waiting time (s/vehicle), 5 seeds × 1 h per scenario**

Single intersection:
| Scenario | Fixed | Longest-queue | Max-pressure | PPO | PPO vs fixed |
|---|---|---|---|---|---|
| low | 8.75 | 3.46 | 4.91 | 3.51 | −60 % |
| normal | 10.82 | 5.79 | 6.65 | 5.38 | −50 % |
| unbalanced | 25.35 | 5.28 | 10.61 | 4.23 | −83 % |
| rush_hour | 21.06 | 5.54 | 13.36 | 4.92 | −77 % |
| accident | 35.26 | 17.42 | 18.28 | 19.39 | −45 % |
| event (unseen) | 28.62 | 6.43 | 13.49 | 5.04 | −82 % |

2×2 network:
| Scenario | Fixed | Longest-queue | Max-pressure | PPO | PPO vs fixed |
|---|---|---|---|---|---|
| low | 15.38 | 6.92 | 9.63 | 6.95 | −55 % |
| normal | 18.71 | 11.50 | 12.80 | 10.10 | −46 % |
| unbalanced | 37.75 | 11.24 | 17.95 | 9.06 | −76 % |
| rush_hour | 32.60 | 12.14 | 18.98 | 10.09 | −69 % |
| accident | 33.01 | 19.83 | 22.33 | 17.13 | −48 % |
| event (unseen) | 35.91 | 12.18 | 20.49 | 10.05 | −72 % |

Other results:
- mean travel time, single intersection, rush hour: 36.6 s (fixed) → 18.5 s (PPO); network: 62.1 s → 37.2 s;
- CO₂: −15 % to −68 % vs fixed-time (single), −12 % to −54 % (network), without CO₂ in the reward;
- stops per vehicle, rush hour: 5.36 → 1.39 (single);
- rush-hour figure: fixed-time reaches ~30 waiting cars at the peak, PPO stays under ~8;
- PPO beats all baselines in 4/6 scenarios (single) and 5/6 (network); it ties the longest-queue
  heuristic in low traffic; **honest weakness**: in the single-intersection accident, the
  longest-queue heuristic is slightly better (17.4 s vs 19.4 s);
- hyper-parameter random search (12 configs × 150 episodes): best = lr 3e-4, γ 0.98, clip 0.3,
  entropy 0.03, ε-start 0.3 (9.6 s); runs **without** exploration (no ε-greedy, no entropy) were the
  worst (19–29 s), which confirms the value of early exploration.

**Limitations & perspectives:** simplified simulator (→ SUMO/TraCI), 2 phases only (→ 4 phases with
protected left turns), multimodal variant (pedestrians, buses with priority), explicit coordination
(MAPPO with centralised critic, graph neural networks), CO₂ directly in the reward (option already
implemented: `--co2-coef`), more training seeds. Applications: smart cities, crisis management
(evacuations, sports events).

**Engineering quality:** Python package, Gymnasium-compatible environment (checked with
`check_env`), 14 unit tests (vehicle conservation, red light blocks traffic, PPO save/load…),
reproducible scripts (train / evaluate / tune), GitHub repository with incremental commits.

## Suggested structure

1. Title slide (project, module, team members' names as placeholders `[Nom 1]`, `[Nom 2]`,
   date, teacher)
2. Context & problem (fixed-time lights vs real traffic; smart-city stakes)
3. Objectives (reduce waiting time, improve flow, robustness to scenarios, scale to networks)
4. Why reinforcement learning? (the agent–environment loop as a diagram applied to a crossroad)
5. The simulator (diagram of an intersection with cells, lights, rules; why not SUMO)
6. Scenarios (6 icons/cards; highlight training vs unseen)
7. MDP formulation: State / Action / Reward (three columns)
8. PPO in one slide (actor–critic diagram, clipping, GAE, epsilon-greedy schedule)
9. From one intersection to a network (parameter sharing, asynchronous agents)
10. Training (training_curve figure + key message)
11. Results: single intersection (bar chart + big number "−77 % waiting at rush hour")
12. Results over time (queue_rush_hour figure: the "wow" slide)
13. Results: 2×2 network + generalisation to the unseen "event" scenario
14. Ecological impact & hyper-parameter tuning insights
15. Limitations & perspectives
16. Conclusion + thank you / questions

For each results slide, put **one key number in large font** (e.g. "−77 %") and the chart beside it.
For diagrams you cannot take from the attached figures, draw them with simple shapes
(boxes and arrows). At the end, add one **backup slide** with the full results tables.
