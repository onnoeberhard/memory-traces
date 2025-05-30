from __future__ import annotations

import jax
import jax.numpy as jnp

from xminigrid.core.actions import take_action
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import EmptyGoal
from xminigrid.core.grid import equal, horizontal_line, room
from xminigrid.core.rules import EmptyRule
from xminigrid.core.observation import transparent_field_of_view
from xminigrid.environment import EnvParams
from xminigrid.types import AgentState, State, StepType, TimeStep
from xminigrid.envs.minigrid.memory import Memory, MemoryEnvCarry, IntOrArray
from xminigrid.registration import _REGISTRY, EnvSpec

_goal_encoding = EmptyGoal().encode()
_rule_encoding = EmptyRule().encode()[None, ...]

_objects = jnp.array(
    (
        (Tiles.FLOOR, Colors.GREEN),
        (Tiles.FLOOR, Colors.BLUE),
    ),
    dtype=jnp.uint8,
)

_wall_tile = TILES_REGISTRY[Tiles.WALL, Colors.GREY]


class TMaze(Memory):
    """Modified from xminigrid.envs.minigrid.memory.Memory"""
    def default_params(self, **kwargs) -> EnvParams:
        params = EnvParams(height=5, width=10, view_size=3)
        params = params.replace(**kwargs)

        if params.max_steps is None:
            # formula directly taken from MiniGrid
            params = params.replace(max_steps=5 * params.width**2)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[MemoryEnvCarry]:
        key, _, _, mem_key, place_key = jax.random.split(key, num=5)

        corridor_length = params.width - 2
        corridor_end = corridor_length

        # setting up the world
        grid = room(params.height, params.width)
        grid = horizontal_line(grid, 1, 1, corridor_length, _wall_tile)
        grid = horizontal_line(grid, 1, 3, corridor_length, _wall_tile)

        # object to memorize
        obj_to_memorize = jax.random.choice(mem_key, _objects, shape=())
        grid = grid.at[2, 1].set(obj_to_memorize)

        # objects to choose
        sides = jax.random.randint(place_key, shape=(), minval=0, maxval=2)
        grid = jax.lax.select(
            sides,
            grid.at[1, corridor_end].set(_objects[0]).at[3, corridor_end].set(_objects[1]),
            grid.at[1, corridor_end].set(_objects[1]).at[3, corridor_end].set(_objects[0]),
        )

        # choosing success and failure positions
        obj_equal_to_upper = equal(obj_to_memorize, grid[1, corridor_end])
        success_pos = jax.lax.select(
            obj_equal_to_upper,
            jnp.array((1, corridor_end)),
            jnp.array((3, corridor_end)),
        )
        failure_pos = jax.lax.select(
            obj_equal_to_upper,
            jnp.array((3, corridor_end)),
            jnp.array((1, corridor_end)),
        )

        # sampling agent position
        agent = AgentState(
            position=jnp.array((2, 1)),
            direction=jnp.asarray(1),
        )
        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=MemoryEnvCarry(success_pos=success_pos, failure_pos=failure_pos),
        )
        return state

    def step(
        self, params: EnvParams, timestep: TimeStep[MemoryEnvCarry], action: IntOrArray
    ) -> TimeStep[MemoryEnvCarry]:
        # disabling pick_up action
        action = jax.lax.select(
            jnp.equal(action, 3),
            jnp.asarray(5),
            action,
        ).astype(jnp.uint8)

        new_grid, new_agent, _ = take_action(timestep.state.grid, timestep.state.agent, action)

        new_state = timestep.state.replace(grid=new_grid, agent=new_agent, step_num=timestep.state.step_num + 1)
        new_observation = transparent_field_of_view(new_state.grid, new_state.agent, params.view_size, params.view_size)

        truncated = new_state.step_num == params.max_steps
        terminated = jnp.logical_or(
            jnp.array_equal(new_agent.position, new_state.carry.success_pos),
            jnp.array_equal(new_agent.position, new_state.carry.failure_pos),
        )
        reward = jax.lax.select(
            jnp.array_equal(new_agent.position, new_state.carry.success_pos),
            1.,
            0.
        )
        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep


def register(id_, entry_point, **kwargs):
    if id_ in _REGISTRY:
        raise ValueError("Environment with such id is already registered. Please choose another one.")
    env_spec = EnvSpec(id=id_, entry_point=entry_point, kwargs=kwargs)
    _REGISTRY[id_] = env_spec

for length in [8, 16, 32, 64, 128, 256, 512, 1024]:
    register(
        id_=f"TMaze{length}",
        entry_point="traces.tmaze:TMaze",
        width=length + 2,
    )
