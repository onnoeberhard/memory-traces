import pickle
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import wandb
from flax import nnx
from traces import ppo

# Configuration options
config = dict(
    memory='trace',  # Type of memory: 'trace', 'window', or None for memoryless
    path=None,       # Path to save the model and metrics, None for no saving
    seed=42,
    env='TMaze64',
    ppo_params=dict(
        total_steps=20_480_000,
        n_envs=16,
        n_steps=128,
        lr=3.e-4,
        gam=0.99,
        gae_lam=0.95,
        n_epochs=2,
        n_minibatches=8,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    ),
    wab=dict(                         # Weights and Biases configuration
        log=False,                    # Enable logging to W&B here
        key='<your_wandb_key_here>',  # Replace with your W&B key
        name='ppo-tmaze',
    ),
    trace_params=dict(
        lams=[0., 0.985]
    ),
    window_params=dict(
        m=64
    )
)
vars().update(config)

# Train PPO agent in T-Maze environment
if wab['log']:
    wandb.login(key=wab['key'])
    wandb.init(name=wab['name'], config=config)
rngs = nnx.Rngs(jax.random.key(seed))
env, env_params, encoding = ppo.make_minigrid(env)
agent = ppo.ActorCritic(encoding.num_actions)
if memory == 'trace':
    mem = ppo.Trace(encoding, **trace_params)
elif memory == 'window':
    mem = ppo.Window(encoding, **window_params)
elif memory is None:  # Memoryless
    mem = ppo.Window(encoding, 0)
model, metrics = ppo.ppo(
    env, env_params, agent, mem, **ppo_params, rng=rngs(), wblog=wab['log'])
if path is not None:
    path = Path(path)
    with open(path / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(path / 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
if wab['log']:
    wandb.finish()

# Evaluate success rate of trained agent
env_reset = jax.jit(lambda rng: env.reset(env_params, rng))
env_step = jax.jit(lambda timestep, action: env.step(env_params, timestep, action))
mem_update = jax.jit(mem.update)
policy = jax.jit(lambda rng, memory: agent.apply(model, mem(memory), rng, method='act'))
timestep = env_reset(rngs())
memory = mem.reset()
action = -1
rewards = 0
episodes = 0
while episodes < 100:
    memory = mem_update(memory, timestep.observation, action)
    action = policy(rngs(), memory)
    timestep = env_step(timestep, action)
    if timestep.last():
        episodes += 1
        rewards += int(timestep.reward)
        timestep = env_reset(rngs())
        memory = mem.reset()
print(f"Success rate: {rewards}/{episodes}")

# Render behavior of trained agent
plt.ion()
timestep = env_reset(rngs())
memory = mem.reset()
action = -1
im = plt.imshow(env.render(env_params, timestep))
while True:
    memory = mem.update(memory, timestep.observation, action)
    action = agent.apply(model, mem(memory), rngs(), method='act')
    timestep = env_step(timestep, action)
    if timestep.last():
        print(f"Episode reward: {int(timestep.reward)}")
        timestep = env_reset(rngs())
        memory = mem.reset()
    im.set_data(env.render(env_params, timestep))
    plt.pause(0.01)
