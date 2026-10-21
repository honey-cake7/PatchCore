"""Self-contained PPO for memory-bank maintenance (no gym / stable-baselines3).

The 53-dim observation / 6-dim continuous action problem does not warrant a
heavy RL framework, and the cluster pins torch 2.0.1 with tight transformers
constraints; a compact PPO avoids all dependency risk. The environment applies
``tanh`` to the raw action inside ``decode_action`` as part of its deterministic
dynamics, so the policy is a plain diagonal Gaussian over the raw action and
ordinary Gaussian log-probs are exact — no tanh-Jacobian correction required.
"""
import copy
import dataclasses
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from patchcore.streaming.env import OBS_DIM, RunningNorm


ACTION_DIM = 6

CHECKPOINT_FORMAT_VERSION = 2


class ActorCritic(nn.Module):
    """Policy + value net.

    ``dist="gaussian"`` (historical): diagonal Gaussian over the RAW action,
    which the env squashes with tanh. Nothing bounds the mean, and on the real
    runs it drifted to |mu| ~ 20-220: every sample then decodes to the same
    corner of the action space and exploration is dead (bin/diagnose_ppo_policy).
    ``log_std_min`` floors the std; the trainer's ``mu_l2`` penalises |mu|.

    ``dist="beta"``: Beta(alpha, beta) per dim on (0, 1), mapped to the env's
    raw action by atanh(2x - 1) so the env's tanh recovers 2x - 1 in (-1, 1).
    Bounded by construction (the mean cannot run away) and its entropy lives
    in action space. alpha, beta = 1 + softplus(.) (unimodal), capped at
    ``conc_max`` — the cap is the std floor. Initialised at alpha = beta = 5
    (std ~0.15 on (0,1)): a moderate start rather than uniform-random
    eviction weights, which produced catastrophic episodes on the fixture.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACTION_DIM, hidden: int = 128,
                 dist: str = "gaussian", log_std_min: Optional[float] = None,
                 conc_max: float = 50.0):
        super().__init__()
        self.dist_kind = dist
        self.log_std_min = log_std_min
        self.conc_max = conc_max
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        if dist == "beta":
            self.conc_head = nn.Linear(hidden, 2 * act_dim)
            with torch.no_grad():
                self.conc_head.weight.mul_(0.01)
                self.conc_head.bias.fill_(float(np.log(np.expm1(4.0))))  # 1+softplus = 5
        elif dist == "gaussian":
            self.mean_head = nn.Linear(hidden, act_dim)
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        else:
            raise ValueError(f"unknown policy dist {dist!r}")
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        h = self.actor(obs)
        if self.dist_kind == "beta":
            conc = (1.0 + nn.functional.softplus(self.conc_head(h))).clamp(max=self.conc_max)
            a, b = conc.chunk(2, dim=-1)
            dist = torch.distributions.Beta(a, b)
        else:
            mean = self.mean_head(h)
            log_std = self.log_std
            if self.log_std_min is not None:
                log_std = log_std.clamp(min=self.log_std_min)
            std = torch.exp(log_std).clamp(1e-3, 5.0)
            dist = torch.distributions.Normal(mean, std)
        value = self.critic(obs).squeeze(-1)
        return dist, value

    # Beta samples live on (0, 1); keep them off the boundary where log_prob
    # and atanh are infinite.
    _EPS = 1e-6

    def sample(self, dist):
        x = dist.sample()
        return x.clamp(self._EPS, 1 - self._EPS) if self.dist_kind == "beta" else x

    def mean_action(self, dist):
        """Deterministic (evaluation) action, in the policy's own space."""
        return dist.mean

    def to_env(self, x):
        """Policy-space action -> the raw action the env tanh-squashes."""
        if self.dist_kind == "beta":
            return torch.atanh((2 * x - 1).clamp(-1 + 2 * self._EPS, 1 - 2 * self._EPS))
        return x


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    epochs: int = 4
    minibatches: int = 64
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    lr_end: Optional[float] = None       # linear anneal target (None = constant)
    ent_coef: float = 1e-3
    ent_coef_end: Optional[float] = None  # linear anneal target (None = constant)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_env_steps: int = 200_000
    device: str = "cpu"
    # "ewma": divide rewards by a running EWMA of per-step reward std (legacy).
    # "fixed": estimate the scale once from the first rollout, then keep it —
    #          the critic's target scale stays stationary across iterations.
    # "none": no scaling (reward terms are already O(1) via estimate_scales).
    reward_scale: str = "ewma"
    # "gae": standard critic + GAE-lambda advantages.
    # "grpo": critic-free group-relative advantages — discounted return-to-go
    #         minus its mean across the vectorized envs at the same timestep.
    #         Valid here because all envs replay the same cached stream in
    #         lockstep, so the group mean is a matched baseline.
    adv_mode: str = "gae"
    # grpo baseline at each timestep: "mean" = group mean including the env
    # itself (original), "loo" = leave-one-out mean of the other envs. Note
    # loo advantages are exactly N/(N-1) x mean advantages, so after update()'s
    # per-batch advantage normalization the two give identical updates; kept
    # selectable because the spec names it and it documents intent.
    adv_baseline: str = "mean"
    hidden: int = 128                    # actor/critic MLP width
    # "clip": standard PPO hard clip (zero gradient once the ratio leaves the
    #         trust region).
    # "gppo": gradient-preserving clip — same forward loss, but the gradient of
    #         clipped samples flows at the boundary value instead of being zeroed
    #         (straight-through: clamp(r.detach()) * exp(logp - logp.detach())).
    clip_mode: str = "clip"
    clip_high: Optional[float] = None  # upper epsilon (clip-higher); None = clip
    # Keep the best policy seen during training instead of the last one.
    # Scored by rolling mean_reward over best_window iters — one window
    # ≈ one full replay of the stream (episode ~2301 steps / 256-step
    # rollouts ≈ 9), which cancels the episode-phase oscillation in the
    # per-iter reward. Runs shorter than best_window keep the final policy.
    save_best: bool = True
    best_window: int = 9
    # Which policy to keep. "rolling": best rolling mean of the TRAINING
    # reward (sampled actions; noisy, and on the real runs it chose between
    # saturated corners by chance). "eval": every eval_every iters run the
    # trainer's eval_fn — a deterministic episode on a held-out set no
    # training episode uses — and keep the best of those.
    select: str = "rolling"
    eval_every: int = 0                  # 0 -> best_window
    # Policy class (see ActorCritic): "gaussian" (historical) | "beta".
    policy_dist: str = "gaussian"
    log_std_min: Optional[float] = None  # gaussian std floor (None = 1e-3 only)
    conc_max: float = 50.0               # beta concentration cap (= std floor)
    mu_l2: float = 0.0                   # gaussian: L2 penalty on the raw mean
    # Stop the update once approx-KL(old || new) on a minibatch exceeds
    # 1.5 x target_kl. Checked per minibatch: with 64 minibatches an epoch is
    # already 64 Adam steps, so a per-epoch check fires far too late.
    target_kl: Optional[float] = None


class PPOTrainer:
    def __init__(self, env_fns: List[Callable], cfg: PPOConfig = None,
                 obs_norm: Optional[RunningNorm] = None, lockstep: bool = False,
                 eval_fn: Optional[Callable] = None):
        self.cfg = cfg or PPOConfig()
        # eval_fn(actor_critic, obs_norm) -> score (higher better); required
        # for cfg.select == "eval"
        self.eval_fn = eval_fn
        self.envs = [fn() for fn in env_fns]
        self.n_env = len(self.envs)
        # lockstep=True: all envs replay the SAME stream (cache path), so
        # their per-step k-NN queries have identical shapes and collect() can
        # fuse them into one batched cdist per phase instead of n_env separate
        # dispatch+sync round-trips. Trajectory-equal to sequential stepping.
        self.lockstep = lockstep
        self._probe_stack = None
        # One normalizer shared by every env (either injected — e.g. prefit on
        # a full episode — or fresh). Per-env normalizers cannot be saved with
        # the policy, which previously made eval see differently-normalized
        # observations than training (train/eval mismatch).
        self.obs_norm = obs_norm if obs_norm is not None else RunningNorm(OBS_DIM)
        for env in self.envs:
            env._norm = self.obs_norm
        self.device = torch.device(self.cfg.device)
        self.ac = ActorCritic(hidden=self.cfg.hidden, dist=self.cfg.policy_dist,
                              log_std_min=self.cfg.log_std_min,
                              conc_max=self.cfg.conc_max).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.cfg.lr)
        self._obs = np.stack([e.reset() for e in self.envs])
        self._ret_rms_std = 1.0  # running return std for reward scaling
        self._fixed_scale_set = False

    def _act(self, obs_np):
        """Returns (policy-space action [buffered for log-probs], env action,
        log-prob, value)."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist, value = self.ac(obs)
            action = self.ac.sample(dist)
            logp = dist.log_prob(action).sum(-1)
            env_action = self.ac.to_env(action)
        return (action.cpu().numpy(), env_action.cpu().numpy(),
                logp.cpu().numpy(), value.cpu().numpy())

    def _lockstep_step(self, action, next_obs):
        """One synchronized step of all envs with fused cross-env k-NNs.

        Phases: decode+mutate per env -> ONE batched cdist per query family
        (reward window, probe, member NN-2, next-image novelty, intra-batch)
        -> per-env numpy assembly. Trajectory-equal to sequential env.step():
        distances are identical and every rng draw stays inside the env at the
        same sequence position. Fills ``next_obs`` in place unless done (all
        envs finish together); returns (rewards [N], done, batched-knn secs).
        """
        from patchcore.streaming.bank import (
            batched_bank_knn, batched_intra_nn2, batched_member_nn2,
        )

        envs = self.envs
        cfg0 = envs[0].reward_cfg
        # ι mode: each env computes ι itself (incremental S_B, inside
        # lockstep_reward). Only the window NN-1 the observation needs is
        # batched here; the proxy's probe and member-NN-2 terms are skipped.
        iota_mode = envs[0].reward_mode == "iota"
        need_r = cfg0.beta != 0 and not iota_mode
        bknn = 0.0

        holds = [env.lockstep_begin(action[i]) for i, env in enumerate(envs)]
        banks = [env.bank for env in envs]

        t0 = time.perf_counter()
        wd, _ = batched_bank_knn(banks, torch.stack(holds), k=1)
        probe_rows = None
        if (not iota_mode and len(envs[0]._proxy.probe)
                and (cfg0.probe_coef != 0 or cfg0.gamma != 0)):
            if (self._probe_stack is None
                    or self._probe_stack.shape[0] != self.n_env):
                # Per-env probes (identical objects when envs share init
                # state, but never assume it); fixed for the envs' lifetime.
                self._probe_stack = torch.stack([
                    torch.from_numpy(env._proxy.probe).to(banks[0].device)
                    for env in envs
                ])
            pd, _ = batched_bank_knn(banks, self._probe_stack, k=1)
            probe_rows = pd[:, :, 0]
        if need_r:
            batched_member_nn2(banks)  # injects each bank's per-step cache
        bknn += time.perf_counter() - t0

        dones = [
            env.lockstep_reward(
                wd[i, :, 0],
                None if probe_rows is None else probe_rows[i],
            )
            for i, env in enumerate(envs)
        ]
        assert all(d == dones[0] for d in dones), "lockstep envs diverged"
        rewards = np.asarray([env._pend_result[0] for env in envs], np.float32)
        if dones[0]:
            # Same rng consumption as the sequential path: the terminal obs is
            # computed (and discarded by the caller) once per episode.
            for env in envs:
                env._observe_terminal()
            return rewards, True, bknn

        adm = [env.lockstep_load() for env in envs]
        t0 = time.perf_counter()
        aq = torch.stack(adm)
        if aq.shape[1]:
            nd, _ = batched_bank_knn(banks, aq, k=1)
        else:  # held-out image: empty admissible batch for every env (shared H)
            nd = np.zeros((len(envs), 0, 1), dtype=np.float32)
        intra = batched_intra_nn2(aq)
        if not need_r:
            # The observation's redundancy features need member NN-2 that the
            # (skipped) reward term no longer provides. Mirror the sequential
            # member_redundancy(sample=1024) branch exactly, rng included.
            obs_sample = 1024
            if len(banks[0]) > obs_sample:
                slot_rows = [env.lockstep_member_draw(obs_sample) for env in envs]
                mq = torch.stack([
                    b._base[torch.from_numpy(s).to(b._device)]
                    for b, s in zip(banks, slot_rows)
                ])
                md, _ = batched_bank_knn(banks, mq, k=2)
                for i, b in enumerate(banks):
                    b._nn2_sample_cache = md[i, :, 1].astype(np.float32)
                    b._nn2_sample_step = b._step
            else:
                batched_member_nn2(banks)
        bknn += time.perf_counter() - t0

        for i, env in enumerate(envs):
            next_obs[i] = env.lockstep_obs(nd[i, :, 0], intra[i])
        return rewards, False, bknn

    def collect(self):
        cfg = self.cfg
        T, N = cfg.rollout_steps, self.n_env
        obs_buf = np.zeros((T, N, OBS_DIM), np.float32)
        act_buf = np.zeros((T, N, ACTION_DIM), np.float32)
        logp_buf = np.zeros((T, N), np.float32)
        rew_buf = np.zeros((T, N), np.float32)
        val_buf = np.zeros((T, N), np.float32)
        done_buf = np.zeros((T, N), np.float32)

        act_s = reset_s = bknn_s = 0.0
        n_resets = 0
        sat_steps = 0.0      # fraction of action dims with |tanh(a)| > 0.995
        admits = 0.0
        for t in range(T):
            t0 = time.perf_counter()
            action, env_action, logp, value = self._act(self._obs)
            act_s += time.perf_counter() - t0
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = (
                self._obs, action, logp, value
            )
            sat_steps += float(np.mean(np.abs(np.tanh(env_action)) > 0.995))
            next_obs = np.zeros_like(self._obs)
            if self.lockstep:
                rews, dones, step_bknn = self._lockstep_step(env_action, next_obs)
                bknn_s += step_bknn
                admits += sum(env._pend_result[1]["n_admit"] for env in self.envs)
                rew_buf[t] = rews
                done_buf[t] = float(dones)
                if dones:
                    for i, env in enumerate(self.envs):
                        t0 = time.perf_counter()
                        next_obs[i] = env.reset()
                        reset_s += time.perf_counter() - t0
                        n_resets += 1
            else:
                for i, env in enumerate(self.envs):
                    o, r, d, info = env.step(env_action[i])
                    admits += info["n_admit"]
                    rew_buf[t, i], done_buf[t, i] = r, float(d)
                    if d:
                        t0 = time.perf_counter()
                        next_obs[i] = env.reset()
                        reset_s += time.perf_counter() - t0
                        n_resets += 1
                    else:
                        next_obs[i] = o
            self._obs = next_obs

        with torch.no_grad():
            _, last_val = self.ac(
                torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            )
        last_val = last_val.cpu().numpy()

        flat_r = rew_buf.reshape(-1)
        if cfg.reward_scale == "ewma":
            self._ret_rms_std = 0.9 * self._ret_rms_std + 0.1 * (flat_r.std() + 1e-6)
            rew_scaled = rew_buf / self._ret_rms_std
        elif cfg.reward_scale == "fixed":
            if not self._fixed_scale_set:
                self._ret_rms_std = float(flat_r.std() + 1e-6)
                self._fixed_scale_set = True
            rew_scaled = rew_buf / self._ret_rms_std
        else:
            rew_scaled = rew_buf

        if cfg.adv_mode == "grpo":
            # Critic-free discounted return-to-go; group-relative baseline.
            G = np.zeros((T, N), np.float32)
            running = np.zeros(N, np.float32)
            for t in reversed(range(T)):
                running = rew_scaled[t] + cfg.gamma * running * (1.0 - done_buf[t])
                G[t] = running
            if cfg.adv_baseline == "loo" and N > 1:
                base = (G.sum(axis=1, keepdims=True) - G) / (N - 1)
            else:
                base = G.mean(axis=1, keepdims=True)
            adv = G - base
            ret = G
        else:
            adv = np.zeros((T, N), np.float32)
            last_gae = np.zeros(N, np.float32)
            for t in reversed(range(T)):
                next_v = last_val if t == T - 1 else val_buf[t + 1]
                next_nonterminal = 1.0 - done_buf[t]
                delta = rew_scaled[t] + cfg.gamma * next_v * next_nonterminal - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                adv[t] = last_gae
            ret = adv + val_buf
        # Drain the envs' wall-clock attribution for this rollout (summed
        # across envs; reset so each iteration reports only its own cost).
        perf = {}
        for env in self.envs:
            for key, val in getattr(env, "perf", {}).items():
                perf[key] = perf.get(key, 0.0) + val
                env.perf[key] = 0.0
        perf["act"] = act_s
        perf[f"reset({n_resets}x)"] = reset_s
        if self.lockstep:
            perf["bknn"] = bknn_s
        return {
            "obs": obs_buf.reshape(-1, OBS_DIM),
            "act": act_buf.reshape(-1, ACTION_DIM),
            "logp": logp_buf.reshape(-1),
            "adv": adv.reshape(-1),
            "ret": ret.reshape(-1),
            "mean_reward": float(flat_r.mean()),
            "perf": perf,
            "sat_frac": sat_steps / T,
            "admits_per_step": admits / (T * N),
        }

    def update(self, batch, ent_coef: Optional[float] = None):
        cfg = self.cfg
        if ent_coef is None:
            ent_coef = cfg.ent_coef
        obs = torch.as_tensor(batch["obs"], device=self.device)
        act = torch.as_tensor(batch["act"], device=self.device)
        old_logp = torch.as_tensor(batch["logp"], device=self.device)
        adv = torch.as_tensor(batch["adv"], device=self.device)
        ret = torch.as_tensor(batch["ret"], device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        idx = np.arange(n)
        mb = max(1, n // cfg.minibatches)
        steps, stopped = 0, False
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                b = idx[start:start + mb]
                dist, value = self.ac(obs[b])
                logp = dist.log_prob(act[b]).sum(-1)
                if cfg.target_kl is not None:
                    with torch.no_grad():
                        log_ratio = logp - old_logp[b]
                        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    if approx_kl > 1.5 * cfg.target_kl:
                        stopped = True
                        break
                ratio = torch.exp(logp - old_logp[b])
                lo = 1.0 - cfg.clip
                hi = 1.0 + (cfg.clip_high if cfg.clip_high is not None else cfg.clip)
                surr1 = ratio * adv[b]
                if cfg.clip_mode == "gppo":
                    # forward value == clamp(ratio); backward grad flows at the
                    # boundary magnitude instead of being zeroed. exp(logp - sg(logp))
                    # is the straight-through factor (== ratio / sg(ratio)) computed
                    # in log-space: ratio itself underflows to 0 for far-off-policy
                    # samples and 0/0 would poison the weights with NaNs.
                    clipped = torch.clamp(ratio.detach(), lo, hi) * torch.exp(logp - logp.detach())
                else:
                    clipped = torch.clamp(ratio, lo, hi)
                surr2 = clipped * adv[b]
                pg_loss = -torch.min(surr1, surr2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg_loss - ent_coef * ent
                if cfg.mu_l2 and self.ac.dist_kind == "gaussian":
                    loss = loss + cfg.mu_l2 * (dist.mean ** 2).mean()
                if cfg.adv_mode != "grpo":  # grpo has no value target
                    v_loss = ((value - ret[b]) ** 2).mean()
                    loss = loss + cfg.vf_coef * v_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()
                steps += 1
            if stopped:
                break
        return steps, stopped

    def train(self, log_every: int = 1):
        cfg = self.cfg
        steps_per_iter = cfg.rollout_steps * self.n_env
        n_iters = max(1, cfg.total_env_steps // steps_per_iter)
        history = []
        best_score, best_iter, best_state = -float("inf"), -1, None
        use_eval = cfg.save_best and cfg.select == "eval"
        if use_eval and self.eval_fn is None:
            raise ValueError("PPOConfig.select='eval' needs PPOTrainer(eval_fn=...)")
        eval_every = cfg.eval_every or cfg.best_window
        last_sat = 0.0
        for it in range(n_iters):
            frac = it / max(n_iters - 1, 1)
            if cfg.lr_end is not None:
                lr = cfg.lr + frac * (cfg.lr_end - cfg.lr)
                for g in self.opt.param_groups:
                    g["lr"] = lr
            ent_coef = cfg.ent_coef
            if cfg.ent_coef_end is not None:
                ent_coef = cfg.ent_coef + frac * (cfg.ent_coef_end - cfg.ent_coef)
            t0 = time.time()
            batch = self.collect()
            collect_s = time.time() - t0
            history.append(batch["mean_reward"])
            # mean_reward measures the pre-update policy, so snapshot before
            # update() — the weights that produced the score, not their successor.
            last_sat = batch["sat_frac"]
            if cfg.save_best and not use_eval and len(history) >= cfg.best_window:
                score = sum(history[-cfg.best_window:]) / cfg.best_window
                if score > best_score:
                    best_score, best_iter = score, it + 1
                    best_state = copy.deepcopy(self.ac.state_dict())
            t0 = time.time()
            n_steps, kl_stop = self.update(batch, ent_coef=ent_coef)
            update_s = time.time() - t0
            eval_str = ""
            if use_eval and ((it + 1) % eval_every == 0 or it + 1 == n_iters):
                # scores the POST-update policy, so snapshot after update()
                score = float(self.eval_fn(self.ac, self.obs_norm))
                eval_str = f" eval={score:.4f}"
                if score > best_score:
                    best_score, best_iter = score, it + 1
                    best_state = copy.deepcopy(self.ac.state_dict())
            if (it + 1) % log_every == 0:
                perf = batch.get("perf", {})
                other = collect_s - sum(perf.values())
                detail = " ".join(f"{k}={v:.1f}" for k, v in perf.items())
                print(f"[ppo] iter {it+1}/{n_iters} mean_reward={batch['mean_reward']:.4f} "
                      f"sat={batch['sat_frac']:.0%} admits/step={batch['admits_per_step']:.2f} "
                      f"grad_steps={n_steps}{' (KL stop)' if kl_stop else ''}{eval_str} "
                      f"collect={collect_s:.1f}s ({detail} other={other:.1f}) "
                      f"update={update_s:.1f}s")
        if best_state is not None:
            self.ac.load_state_dict(best_state)
            how = (f"deterministic eval={best_score:.4f}" if use_eval else
                   f"rolling[{cfg.best_window}] mean_reward={best_score:.4f}")
            print(f"[ppo] restored best policy from iter {best_iter} ({how})")
        if last_sat > 0.5:
            print(f"[ppo] WARNING: {last_sat:.0%} of action dims are saturated "
                  "(|tanh(a)| > 0.995) in the last rollout — every sample decodes "
                  "to the same decision and exploration is dead. Use "
                  "policy_dist='beta' (or log_std_min + mu_l2) and target_kl.")
        return history

    def save(self, path: str):
        env0 = self.envs[0]
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "ac": self.ac.state_dict(),
                "obs_norm": self.obs_norm.state_dict(),
                "obs_dim": OBS_DIM,
                "act_dim": ACTION_DIM,
                "action_mode": env0.action_cfg.mode,
                "reward_cfg": dataclasses.asdict(env0.reward_cfg),
                "reward_mode": env0.reward_mode,
                "heldout_frac": env0.heldout_frac,
                "admit_cap_abs": env0.admit_cap_abs,
                "hidden": self.cfg.hidden,
                "policy_dist": self.cfg.policy_dist,
                "log_std_min": self.cfg.log_std_min,
                "conc_max": self.cfg.conc_max,
                "action_cfg": dataclasses.asdict(env0.action_cfg),
            },
            path,
        )

    def load(self, path: str):
        ac, norm, _ = load_checkpoint(path, device=self.device)
        self.ac.load_state_dict(ac.state_dict())
        if norm is not None:
            self.obs_norm = norm
            for env in self.envs:
                env._norm = self.obs_norm


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[ActorCritic, Optional[RunningNorm], dict]:
    """Load a policy checkpoint; returns (actor_critic, obs_norm, meta).

    Handles both format v2 (dict with normalizer + metadata) and legacy
    checkpoints that are a raw ``state_dict`` — those return ``obs_norm=None``,
    meaning eval will re-fit normalization from its own stream (the old,
    train/eval-mismatched behavior).
    """
    # Our own artifact (contains numpy arrays for the normalizer); torch>=2.6
    # defaults weights_only=True which rejects them.
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and obj.get("format_version", 0) >= 2:
        ac = ActorCritic(obj.get("obs_dim", OBS_DIM), obj.get("act_dim", ACTION_DIM),
                         hidden=obj.get("hidden", 128),
                         dist=obj.get("policy_dist", "gaussian"),
                         log_std_min=obj.get("log_std_min"),
                         conc_max=obj.get("conc_max", 50.0))
        ac.load_state_dict(obj["ac"])
        ac.to(torch.device(device))
        norm = RunningNorm.from_state_dict(obj["obs_norm"]) if obj.get("obs_norm") else None
        # reward_mode / heldout_frac / admit_cap_abs / hidden only exist in
        # checkpoints written after the ι rebuild; absent keys = legacy proxy.
        meta = {k: obj[k] for k in ("action_mode", "reward_cfg", "reward_mode",
                                    "heldout_frac", "admit_cap_abs", "hidden",
                                    "policy_dist", "action_cfg")
                if k in obj}
        return ac, norm, meta
    print("[ppo] WARNING: legacy checkpoint (no obs normalizer saved) — "
          "eval will re-fit observation normalization from its own stream.")
    ac = ActorCritic()
    ac.load_state_dict(obj)
    ac.to(torch.device(device))
    return ac, None, {}


def action_cfg_from_meta(meta: Optional[dict], default_mode: str = "continuous6"):
    """The ActionConfig a checkpoint trained under (mode, admit floor, caps);
    older checkpoints stored only the mode, which implies the old defaults."""
    from patchcore.streaming.env import ActionConfig

    meta = meta or {}
    if meta.get("action_cfg"):
        return ActionConfig(**meta["action_cfg"])
    return ActionConfig(mode=meta.get("action_mode") or default_mode)
