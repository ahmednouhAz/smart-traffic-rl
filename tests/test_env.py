import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_rl.agents.baselines import BASELINES  # noqa: E402
from traffic_rl.agents.ppo import PPOAgent, PPOConfig, PPOController  # noqa: E402
from traffic_rl.envs import ALL_SCENARIOS, EnvConfig, SingleIntersectionEnv, TrafficNetworkEnv  # noqa: E402
from traffic_rl.envs.simulator import TrafficSimulator  # noqa: E402
from traffic_rl.training import run_episode, train_ppo  # noqa: E402


def test_gymnasium_api_compliance():
    from gymnasium.utils.env_checker import check_env

    env = SingleIntersectionEnv(EnvConfig(episode_length=300))
    check_env(env, skip_render_check=True)


def test_observation_shape_and_bounds():
    env = TrafficNetworkEnv(EnvConfig(rows=2, cols=2, episode_length=600), seed=0)
    obs = env.reset(seed=0)
    assert len(obs) == 4
    done = False
    while not done:
        for o in obs.values():
            assert o.shape == (env.observation_dim,)
            assert np.all(o >= 0) and np.all(o <= 3.0)
        obs, rewards, done, _ = env.step({a: 0 for a in obs})
        assert set(rewards) == set(obs)


def test_action_encoding_roundtrip():
    env = TrafficNetworkEnv(EnvConfig())
    for a in range(env.n_actions):
        phase, dur = env.decode_action(a)
        assert env.encode_action(phase, dur) == a


def test_vehicle_conservation():
    sim = TrafficSimulator(rows=2, cols=2, seed=1)
    for _ in range(1200):
        sim.step(lambda t, i, o: 0.2)
        if sim.t % 20 == 0:
            for inter in sim.intersections:
                sim.set_signal(inter, (sim.t // 20) % 2, 20)
    assert sim.spawned == sim.completed + sim.vehicles_in_network()


def test_no_two_vehicles_in_same_cell():
    sim = TrafficSimulator(rows=2, cols=2, seed=2)
    for _ in range(900):
        sim.step(lambda t, i, o: 0.3)
        for lane in sim.lanes.values():
            pos = [v.pos for v in lane.vehicles]
            assert len(pos) == len(set(pos))
            assert pos == sorted(pos)


def test_red_light_blocks_traffic():
    sim = TrafficSimulator(seed=0)
    sim.set_signal((0, 0), 0, 10_000)  # North-South green forever
    for _ in range(600):
        sim.step(lambda t, i, o: 0.1 if o in ("E", "W") else 0.0)
    assert sim.completed == 0
    assert sim.total_queue() > 0


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_scenarios_run(scenario):
    env = TrafficNetworkEnv(EnvConfig(episode_length=600), seed=0)
    metrics, history = run_episode(env, BASELINES["fixed_time"](), seed=0, scenario=scenario)
    assert len(history) == 600
    assert metrics["spawned"] > 0


def test_ppo_smoke_and_save_load(tmp_path):
    cfg = EnvConfig(episode_length=600)
    agent, log = train_ppo(cfg, PPOConfig(total_episodes=3, rollout_steps=64, minibatch_size=32), verbose=False)
    assert len(log) == 3
    path = tmp_path / "ppo.pt"
    agent.save(path)
    loaded = PPOAgent.load(path)
    env = TrafficNetworkEnv(cfg, seed=0)
    obs = env.reset(seed=0)
    a1 = PPOController(agent).act(env, obs)
    a2 = PPOController(loaded).act(env, obs)
    assert a1 == a2


def test_epsilon_schedule():
    agent = PPOAgent(10, 4, PPOConfig(total_episodes=100, eps_start=0.3, eps_decay_fraction=0.3))
    assert agent.epsilon(0) == pytest.approx(0.3)
    assert agent.epsilon(15) == pytest.approx(0.15)
    assert agent.epsilon(30) == pytest.approx(0.0)
    assert agent.epsilon(99) == pytest.approx(0.0)
