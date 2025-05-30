import sys
from functools import partial

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import xminigrid
from einops import rearrange
from flax import linen as nn
from flax import nnx
from flax.struct import PyTreeNode
from oetils import JaxTqdm
from xminigrid.core.constants import NUM_ACTIONS, TILES_REGISTRY, Colors, Tiles

from traces.tmaze import TMaze


class MinigridEncoding:
    def __init__(self, num_actions, tiles, view_size):
        self.num_actions = num_actions
        self.tiles = tiles
        self.view_size = view_size
        self.obs_dims = self.view_size, self.view_size, len(self.tiles)
        self.act_dims = self.num_actions,

    @partial(jax.jit, static_argnums=0)
    def encode(self, obs, act):
        # One-hot encoding of tiles in observation wrt. `TILES`
        def enc(tile):
            matches = (tile == self.tiles).all(axis=-1)
            matches = eqx.error_if(matches, ~matches.any(), 'Invalid observation')
            return jnp.eye(len(self.tiles))[matches.argmax()]
        act = jax.lax.select(act >= 0, jnp.eye(self.num_actions)[act], jnp.zeros(self.num_actions))
        return jax.vmap(jax.vmap(enc))(obs), act


def make_minigrid(env, **env_params):
    env, env_params = xminigrid.make(env, **env_params)
    match env:
        case TMaze():
            num_actions = 3
            tiles = jnp.array([[Tiles.EMPTY, Colors.EMPTY],
                [Tiles.FLOOR, Colors.GREEN], [Tiles.FLOOR, Colors.BLUE],
                [Tiles.WALL, Colors.GREY], [Tiles.FLOOR, Colors.BLACK]], dtype=jnp.uint8)
        case _:
            num_actions = NUM_ACTIONS
            tiles = TILES_REGISTRY.reshape(-1, 2)
    encoding = MinigridEncoding(num_actions, tiles, env_params.view_size)
    return env, env_params, encoding


class Memory:
    def __init__(self, encoding, size, actions=False):
        self.encoding = encoding
        self.size = size
        self.actions = actions

    def reset(self):
        return (jnp.zeros((self.size, *self.encoding.obs_dims)), jnp.zeros((self.size, *self.encoding.act_dims)))

    def __call__(self, z):
        if self.actions:
            return jnp.concat([z[0].ravel(), z[1].ravel()])
        return z[0].ravel()


class Window(Memory):
    def __init__(self, encoding, m, **kwargs):
        self.m = m
        super().__init__(encoding, m + 1, **kwargs)

    def update(self, z, y, u):
        y, u = self.encoding.encode(y, u)
        return (jnp.roll(z[0], 1, axis=0).at[0].set(y), jnp.roll(z[1], 1, axis=0).at[0].set(u))


class Trace(Memory):
    def __init__(self, encoding, lam=None, lams=None, m=None, episodic=False, **kwargs):
        if lam is not None:
            lams = [lam]
        if m is not None:
            lams = jnp.arange(1/(2*(m + 1)), 1, 1/(m + 1))
        self.lams = jnp.asarray(lams)
        self.lams_y = self.lams[:, None, None, None]
        self.lams_u = self.lams[:, None]
        self.episodic = episodic
        super().__init__(encoding, len(lams), **kwargs)

    def update(self, z, y, u):
        y, u = self.encoding.encode(y, u)
        if self.episodic:
            z, L = z
            return (y + self.lams_y * z[0], u + self.lams_u * z[1]), (1 + self.lams_y * L[0], 1 + self.lams_u * L[1])
        return (1 - self.lams_y) * y + self.lams_y * z[0], (1 - self.lams_u) * u + self.lams_u * z[1]

    def reset(self):
        z = super().reset()
        if self.episodic:
            return z, (jnp.zeros_like(self.lams_y), jnp.zeros_like(self.lams_u))
        return z

    def __call__(self, z):
        if self.episodic:
            z, L = z
            L = eqx.error_if(L, (L[0] == 0).any(), 'Normalization is 0')
            if self.actions:
                return jnp.concat([(z[0] / L[0]).ravel(), (z[1] / L[1]).ravel()])
            return (z[0] / L[0]).ravel()
        return super().__call__(z)


class Transition(PyTreeNode):  # pylint: disable=abstract-method
    features: jax.Array
    action: jax.Array
    reward: jax.Array
    gamma: jax.Array  # 0 if terminated, 1 otherwise (xminigrid)
    done: jax.Array
    value: jax.Array
    log_prob: jax.Array
    advantage: jax.Array
    target: jax.Array


class ActorCritic(nn.Module):
    num_actions: int

    def setup(self):
        self.actor = nn.Sequential([  # pylint: disable=attribute-defined-outside-init
            jnp.ravel,
            nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
            nn.tanh,
            nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
            nn.tanh,
            nn.Dense(self.num_actions, kernel_init=nn.initializers.orthogonal(0.01))
        ])
        self.critic = nn.Sequential([  # pylint: disable=attribute-defined-outside-init
            jnp.ravel,
            nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
            nn.tanh,
            nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
            nn.tanh,
            nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))
        ])

    def __call__(self, z):
        return self.critic(z), distrax.Categorical(self.actor(z))

    def value(self, z):
        return self.critic(z)

    def act(self, z, rng):
        return distrax.Categorical(self.actor(z)).sample(seed=rng)


def ppo(env, env_params, agent, mem, total_steps, n_envs, n_steps, lr, gam, gae_lam, n_epochs,
        n_minibatches, clip_eps, vf_coef, ent_coef, max_grad_norm, rng, wblog=False, wblog_all=False, n_monitor=1000):
    n_updates = total_steps // (n_envs * n_steps)
    i_monitor = max(n_updates // n_monitor, 1)
    rngs = nnx.Rngs(rng)

    # Initialize environment
    timestep = jax.vmap(env.reset, (None, 0))(env_params, jax.random.split(rngs(), n_envs))
    action = -jnp.ones(n_envs, int)

    # Initialize memory
    vmem = jax.vmap(mem)  # Flatten memory features (combine time and tile dimensions)
    memory = jax.vmap(lambda _: mem.reset())(jnp.arange(n_envs))
    memory = jax.vmap(mem.update)(memory, timestep.observation, action)

    # Initialize agent
    params = agent.init(rngs(), vmem(memory)[0])

    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(optax.linear_schedule(lr, 0, n_updates * n_epochs * n_minibatches)))
    opt_state = optimizer.init(params)

    def step(x, _):
        rng, params, timestep, action, memory = x
        rng, *keys = jax.random.split(rng, 3)

        # Sample action, take step
        agent_state = vmem(memory)
        v, pi = jax.vmap(lambda z: agent.apply(params, z))(agent_state)
        action = pi.sample(seed=keys[0])
        log_prob = pi.log_prob(action)
        timestep_ = jax.vmap(env.step, (None, 0, 0))(env_params, timestep, action)

        # Reset environment and memory if done
        timestep = jax.vmap(lambda ts, rng: jax.tree.map(lambda x, y: jax.lax.select(ts.last(), x, y),
            env.reset(env_params, rng), ts))(timestep_, jax.random.split(keys[1], n_envs))
        memory = jax.vmap(lambda r, m: jax.tree.map(
            lambda x, y: jax.lax.select(r, x, y), mem.reset(), m))(timestep_.last(), memory)

        # Update memory with new observation
        memory = jax.vmap(mem.update)(memory, timestep.observation, action)
        return (rng, params, timestep, action, memory), Transition(
            agent_state, action, timestep_.reward, timestep_.discount, timestep_.last(), v.squeeze(), log_prob,
            jnp.full_like(timestep_.reward, jnp.nan), jnp.full_like(timestep_.reward, jnp.nan))

    def gae(trajectory, last_features, params):
        # Compute generalized advantage estimate
        def gae_step(x, transition):
            A, v = x
            delta = transition.reward + gam * transition.gamma * v - transition.value
            A = delta + gam * transition.gamma * gae_lam * A
            return (A, transition.value), A
        v = jax.vmap(lambda z: agent.apply(params, z, method='value'))(last_features).squeeze()
        return jax.vmap(lambda v, traj: jax.lax.scan(gae_step, (0, v), traj, reverse=True)[1])(v, trajectory)

    def loss(params, batch):
        v, pi = jax.vmap(lambda z: agent.apply(params, z))(batch.features)
        value_loss = optax.losses.l2_loss(v.squeeze(), batch.target).mean()  # No value clipping
        logratio = pi.log_prob(batch.action) - batch.log_prob
        ratio = jnp.exp(logratio)
        advantage = (batch.advantage - batch.advantage.mean()) / (batch.advantage.std() + 1e-8)
        policy_loss = -jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantage).mean()
        entropy = pi.entropy().mean()
        return policy_loss + vf_coef * value_loss - ent_coef * entropy, {
            'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': entropy,
            'approx_kl': ((ratio - 1) - logratio).mean(), 'clipfrac': (jnp.abs(ratio - 1) > clip_eps).mean()
        }

    def batch_update(x, batch):
        params, opt_state = x
        (losses, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (losses, metrics)

    def epoch_update(x, _):
        rng, params, opt_state, data = x
        rng, key = jax.random.split(rng)
        perm = jax.random.permutation(key, n_envs * n_steps)
        batch = jax.tree.map(lambda x: rearrange(x[perm], '(m b) ... -> m b ...', m=n_minibatches), data)
        (params, opt_state), (losses, metrics) = jax.lax.scan(batch_update, (params, opt_state), batch)
        return (rng, params, opt_state, data), (losses.mean(), jax.tree.map(jnp.mean, metrics))

    def ret_len(c, x):
        ret, tot_rew, time = c
        reward, done = x
        y = ret + reward * gam**time
        y_ = tot_rew + reward
        l = time + 1
        ret = jax.lax.select(done, jnp.zeros_like(y), y)
        tot_rew = jax.lax.select(done, jnp.zeros_like(y_), y_)
        time = jax.lax.select(done, jnp.zeros_like(l), l)
        return (ret, tot_rew, time), (y, y_, l)

    def monitor(i, rets, tot_rews, lens, dones, metrics):
        pbar.write(f"Steps: {(i + 1) * n_envs * n_steps:{int(np.log10(total_steps)) + 1}d}. "
            # f"mean return: {metrics['mean_return']:8.6f}, "
            f"mean total reward: {metrics['mean_tot_rew']:8.6f}, "
            f"mean episode length: {metrics['mean_ep_len']:8.3f}")
            # f"mean approximate KL divergence: {metrics['approx_kl']:8.6f}")
        sys.stdout.flush()
        if wblog_all:  # only makes sense with n_monitor == n_updates
            idx = jnp.where(dones)
            for t, n in sorted(zip(*idx), key=lambda tn: tn[0]):
                wandb.log({'return': rets[t, n], 'tot_rew': tot_rews[t, n], 'length': lens[t, n]},
                    i * n_envs * n_steps + t)
        if wblog:
            wandb.log({f'losses/{k}': metrics[k] for k in [
                    'total_loss', 'policy_loss', 'value_loss', 'approx_kl', 'entropy', 'clipfrac'
                ]} | {k: metrics[k] for k in ['mean_return', 'mean_tot_rew', 'mean_ep_len']},
                (i + 1) * n_envs * n_steps)

    pbar = JaxTqdm(n_updates, n_monitor)
    @pbar.loop
    def update(i, x):
        rng, params, metrics, opt_state, timestep, action, memory, ret, tot_rew, time = x
        rng, *keys = jax.random.split(rng, 3)

        # Collect trajectories
        (*_, timestep, action, memory), trajectories = jax.lax.scan(
            step, (keys[0], params, timestep, action, memory), length=n_steps)
        (ret, tot_rew, time), (rets, tot_rews, lens) = jax.lax.scan(
            ret_len, (ret, tot_rew, time), (trajectories.reward, trajectories.done))
        trajectories = jax.tree.map(lambda x: rearrange(x, 't n ... -> n t ...'), trajectories)

        # Compute advantages and targets
        trajectories = trajectories.replace(advantage=gae(trajectories, vmem(memory), params))
        trajectories = trajectories.replace(target=trajectories.value + trajectories.advantage)

        # Update parameters
        data = jax.tree.map(lambda x: rearrange(x, 'n t ... -> (n t) ...'), trajectories)
        (_, params, opt_state, _), (losses, metrics_) = jax.lax.scan(
            epoch_update, (keys[1], params, opt_state, data), length=n_epochs)

        # Monitoring
        metrics_ = jax.tree.map(jnp.mean, metrics_) | {'total_loss': losses.mean()} | {
            'mean_return': rets.mean(where=trajectories.done.T),
            'mean_tot_rew': tot_rews.mean(where=trajectories.done.T),
            'mean_ep_len': lens.mean(where=trajectories.done.T)}
        metrics = jax.tree.map(lambda l, m: l.at[i].set(m), metrics, metrics_)
        jax.lax.cond((i + 1) % i_monitor == 0,
            lambda: jax.debug.callback(monitor, i, rets, tot_rews, lens, trajectories.done.T, metrics_), lambda: None)
        return rng, params, metrics, opt_state, timestep, action, memory, ret, tot_rew, time

    ret, tot_rew, time = jnp.zeros(n_envs), jnp.zeros(n_envs), jnp.zeros(n_envs)
    metrics = {k: jnp.zeros(n_updates) for k in ['mean_return', 'mean_tot_rew', 'mean_ep_len',
        'total_loss', 'policy_loss', 'value_loss', 'approx_kl', 'entropy', 'clipfrac']}
    _, params, metrics, *_ = jax.lax.fori_loop(0, n_updates, update,
        (rng, params, metrics, opt_state, timestep, action, memory, ret, tot_rew, time))
    sys.stdout.flush()
    return params, metrics


if __name__ == '__main__':
    config = dict(env='MiniGrid-Empty-16x16', seed=42, memory='trace', env_params=dict(view_size=3),
        trace_params=dict(lam=0.5, episodic=True), window_params=dict(m=3),
        ppo_params=dict(total_steps=1_024_000, n_envs=16, n_steps=128, lr=3e-4, gam=0.99, gae_lam=0.95, n_epochs=2,
        n_minibatches=8, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5))
    rngs = nnx.Rngs(jax.random.key(config['seed']))
    env, env_params, encoding = make_minigrid(config['env'], **config['env_params'])
    agent = ActorCritic(encoding.num_actions)
    if config['memory'] == 'trace':
        mem = Trace(encoding, **config['trace_params'])
    elif config['memory'] == 'window':
        mem = Window(encoding, **config['window_params'])
    else:  # Memoryless
        mem = Window(encoding, 0)
    params, metrics = ppo(env, env_params, agent, mem, **config['ppo_params'], rng=rngs())

    # Show policy
    import matplotlib.pyplot as plt
    plt.ion()
    env_reset = jax.jit(lambda rng: env.reset(env_params, rng))
    env_step = jax.jit(lambda timestep, action: env.step(env_params, timestep, action))
    timestep = env_reset(rngs())
    memory = mem.reset()
    action = -1
    im = plt.imshow(env.render(env_params, timestep))
    while True:
        memory = mem.update(memory, timestep.observation, action)
        action = agent.apply(params, mem(memory), rngs(), method='act')
        timestep = env_step(timestep, action)
        if timestep.last():
            timestep = env_reset(rngs())
            memory = mem.reset()
        im.set_data(env.render(env_params, timestep))
        plt.pause(0.01)
