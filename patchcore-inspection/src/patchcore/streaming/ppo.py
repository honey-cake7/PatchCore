"""Self-contained PPO for memory-bank maintenance (no gym / stable-baselines3).

The 53-dim observation / 6-dim continuous action problem does not warrant a
heavy RL framework, and the cluster pins torch 2.0.1 with tight transformers
constraints; a compact PPO avoids all dependency risk. The environment applies
``tanh`` to the raw action inside ``decode_action`` as part of its deterministic
dynamics, so the policy is a plain diagonal Gaussian over the raw action and
ordinary Gaussian log-probs are exact — no tanh-Jacobian correction required.
"""
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn

from patchcore.streaming.env import OBS_DIM


ACTION_DIM = 6


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACTION_DIM, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        h = self.actor(obs)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std).clamp(1e-3, 5.0)
        dist = torch.distributions.Normal(mean, std)
        value = self.critic(obs).squeeze(-1)
        return dist, value


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    epochs: int = 4
    minibatches: int = 8
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_env_steps: int = 200_000
    device: str = "cpu"


class PPOTrainer:
    def __init__(self, env_fns: List[Callable], cfg: PPOConfig = None):
        self.cfg = cfg or PPOConfig()
        self.envs = [fn() for fn in env_fns]
        self.n_env = len(self.envs)
        self.device = torch.device(self.cfg.device)
        self.ac = ActorCritic().to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.cfg.lr)
        self._obs = np.stack([e.reset() for e in self.envs])
        self._ret_rms_std = 1.0  # running return std for reward scaling

    def _act(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist, value = self.ac(obs)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    def collect(self):
        cfg = self.cfg
        T, N = cfg.rollout_steps, self.n_env
        obs_buf = np.zeros((T, N, OBS_DIM), np.float32)
        act_buf = np.zeros((T, N, ACTION_DIM), np.float32)
        logp_buf = np.zeros((T, N), np.float32)
        rew_buf = np.zeros((T, N), np.float32)
        val_buf = np.zeros((T, N), np.float32)
        done_buf = np.zeros((T, N), np.float32)

        for t in range(T):
            action, logp, value = self._act(self._obs)
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = (
                self._obs, action, logp, value
            )
            next_obs = np.zeros_like(self._obs)
            for i, env in enumerate(self.envs):
                o, r, d, _ = env.step(action[i])
                rew_buf[t, i], done_buf[t, i] = r, float(d)
                next_obs[i] = env.reset() if d else o
            self._obs = next_obs

        with torch.no_grad():
            _, last_val = self.ac(
                torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            )
        last_val = last_val.cpu().numpy()

        # scale rewards by running return std (stabilizes advantages)
        flat_r = rew_buf.reshape(-1)
        self._ret_rms_std = 0.9 * self._ret_rms_std + 0.1 * (flat_r.std() + 1e-6)
        rew_buf = rew_buf / self._ret_rms_std

        adv = np.zeros((T, N), np.float32)
        last_gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            next_v = last_val if t == T - 1 else val_buf[t + 1]
            next_nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * next_v * next_nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + val_buf
        return {
            "obs": obs_buf.reshape(-1, OBS_DIM),
            "act": act_buf.reshape(-1, ACTION_DIM),
            "logp": logp_buf.reshape(-1),
            "adv": adv.reshape(-1),
            "ret": ret.reshape(-1),
            "mean_reward": float(flat_r.mean()),
        }

    def update(self, batch):
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"], device=self.device)
        act = torch.as_tensor(batch["act"], device=self.device)
        old_logp = torch.as_tensor(batch["logp"], device=self.device)
        adv = torch.as_tensor(batch["adv"], device=self.device)
        ret = torch.as_tensor(batch["ret"], device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        idx = np.arange(n)
        mb = max(1, n // cfg.minibatches)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                b = idx[start:start + mb]
                dist, value = self.ac(obs[b])
                logp = dist.log_prob(act[b]).sum(-1)
                ratio = torch.exp(logp - old_logp[b])
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[b]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((value - ret[b]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()

    def train(self, log_every: int = 1):
        cfg = self.cfg
        steps_per_iter = cfg.rollout_steps * self.n_env
        n_iters = max(1, cfg.total_env_steps // steps_per_iter)
        history = []
        for it in range(n_iters):
            batch = self.collect()
            self.update(batch)
            history.append(batch["mean_reward"])
            if (it + 1) % log_every == 0:
                print(f"[ppo] iter {it+1}/{n_iters} mean_reward={batch['mean_reward']:.4f}")
        return history

    def save(self, path: str):
        torch.save(self.ac.state_dict(), path)

    def load(self, path: str):
        self.ac.load_state_dict(torch.load(path, map_location=self.device))
