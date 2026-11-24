"""Proximal Policy Optimization (PPO-Clip) for traffic-signal control.

Implementation from scratch with PyTorch:

* actor-critic MLP (separate policy and value networks, tanh activations);
* Generalised Advantage Estimation (GAE-lambda);
* clipped surrogate objective + value loss + entropy bonus;
* **epsilon-greedy exploration** at the beginning of training: with probability
  ``epsilon`` the agent plays a uniformly random action instead of sampling its
  policy; ``epsilon`` decays linearly to 0 (as asked in the project brief);
* **parameter sharing** for networks: every intersection is an agent but all
  agents use the same neural network, so the policy scales to any grid size.

Rollouts are collected per agent (agents decide asynchronously), then
advantages are computed per agent trajectory and everything is merged into a
single batch for the update.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 8
    minibatch_size: int = 256
    rollout_steps: int = 2048  # transitions collected before each update
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    eps_start: float = 0.3  # epsilon-greedy exploration at the start
    eps_end: float = 0.0
    eps_decay_fraction: float = 0.3  # fraction of training used to decay epsilon
    total_episodes: int = 300
    seed: int = 0


def _layer(in_dim: int, out_dim: int, gain: float = np.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, n_actions, 0.01)
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, 1, 1.0)
        )

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class Trajectory:
    """Transitions of one agent during one episode."""

    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logps: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, config: Optional[PPOConfig] = None) -> None:
        self.cfg = config or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.net = ActorCritic(obs_dim, n_actions, self.cfg.hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)

    # ---------------------------------------------------------------- acting
    def epsilon(self, episode: int) -> float:
        decay = max(1, int(self.cfg.eps_decay_fraction * self.cfg.total_episodes))
        frac = min(1.0, episode / decay)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, obs_batch: np.ndarray, epsilon: float = 0.0, greedy: bool = False):
        """Return actions, log-probabilities and state values for a batch of observations."""
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        dist = self.net.dist(obs_t)
        if greedy:
            actions = dist.probs.argmax(-1)
        else:
            actions = dist.sample()
            if epsilon > 0:
                explore = torch.as_tensor(self.rng.random(len(actions)) < epsilon)
                random_a = torch.as_tensor(self.rng.integers(self.n_actions, size=len(actions)))
                actions = torch.where(explore, random_a, actions)
        return actions.numpy(), dist.log_prob(actions).numpy(), self.net.value(obs_t).numpy()

    @torch.no_grad()
    def value(self, obs_batch: np.ndarray) -> np.ndarray:
        return self.net.value(torch.as_tensor(obs_batch, dtype=torch.float32)).numpy()

    # -------------------------------------------------------------- learning
    def compute_gae(self, traj: Trajectory, last_value: float):
        """GAE over one trajectory. The episode ends on a time limit, so we bootstrap with V(s_T)."""
        n = len(traj)
        adv = np.zeros(n, dtype=np.float32)
        values = np.append(np.asarray(traj.values[:n], dtype=np.float32), last_value)
        gae = 0.0
        for t in reversed(range(n)):
            delta = traj.rewards[t] + self.cfg.gamma * values[t + 1] - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * gae
            adv[t] = gae
        return adv, adv + values[:-1]

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long)
        old_logp = torch.as_tensor(batch["logps"], dtype=torch.float32)
        adv = torch.as_tensor(batch["advantages"], dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n_updates = 0
        for _ in range(cfg.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                dist = self.net.dist(obs[mb])
                logp = dist.log_prob(actions[mb])
                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (self.net.value(obs[mb]) - returns[mb]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy"] += entropy.item()
                    stats["approx_kl"] += (old_logp[mb] - logp).mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
                n_updates += 1
        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "config": asdict(self.cfg),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PPOAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], PPOConfig(**ckpt["config"]))
        agent.net.load_state_dict(ckpt["state_dict"])
        return agent


class PPOController:
    """Adapter so a trained PPO agent can be evaluated like the baselines."""

    name = "ppo"

    def __init__(self, agent: PPOAgent, greedy: bool = True) -> None:
        self.agent, self.greedy = agent, greedy

    def reset(self) -> None:
        pass

    def act(self, env, obs: Dict) -> Dict:
        keys = list(obs)
        actions, _, _ = self.agent.act(np.stack([obs[k] for k in keys]), greedy=self.greedy)
        return {k: int(a) for k, a in zip(keys, actions)}
