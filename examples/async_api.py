import abc
import enact
import asyncio
import contextlib
import dataclasses
import numpy as np
from typing import Optional

from async_asteroids import Action, State


class AsyncPolicy(abc.ABC):
  @abc.abstractmethod
  async def compute_actions(self, observations: State) -> Action:
    """Compute actions from observations."""
  
  @abc.abstractmethod
  async def stop(self):
    """Stop the policy."""
  

class Game:
  def __init__(self, policy: AsyncPolicy):
    self.policy = policy
    self.state = State()
    self.state.randomize()
    self.steps = 300

  async def dynamics(self, actions: Action) -> State:
    dt = 0.1
    next_state = State(np.copy(self.state.array))
    # Update velocity by applying acceleration forces.
    thrust = np.array([np.clip(actions.thrust, 0, 1)])
    torque = np.array([np.clip(actions.torque, -1, 1)])
    next_state.velocity += (
        dt * self.state.THRUST_FORCE *
        self.state.forward() * np.expand_dims(thrust, axis=-1))
    next_state.angular_velocity += dt * self.state.TORQUE_FORCE * torque

    # Clamp max speed and apply friction.
    speed = np.clip(
        np.expand_dims(np.linalg.norm(next_state.velocity, axis=-1), -1),
        1e-10, np.infty)

    at_maxed_out_speed = (next_state.velocity / speed) * self.state.MAX_SPEED
    next_state.velocity = np.where(
        speed > self.state.MAX_SPEED, at_maxed_out_speed, next_state.velocity)

    next_state.velocity *= self.state.FRICTION_COEFFICIENT
    next_state.angular_velocity *= self.state.FRICTION_COEFFICIENT

    # Update position by applying velocity.
    next_state.position += dt * next_state.velocity
    next_state.rotation += dt * next_state.angular_velocity

    next_state.has_been_at_goal1 += next_state.at_goal1()
    next_state.has_been_at_goal1 = np.clip(next_state.has_been_at_goal1, 0, 1)

    next_state.has_been_at_goal2 += next_state.at_goal2()
    next_state.has_been_at_goal2 = np.clip(next_state.has_been_at_goal2, 0, 1)

    return next_state
  
  async def update(self):
    actions = await self.policy.compute_actions(self.state)
    self.state = await self.dynamics(actions)
  
  async def game_loop(self):
    for i in range(self.steps):
      await self.update()
    await self.policy.stop()
  
@enact.typed_invokable(Action, enact.NPArray)
class Step(enact.Invokable):
  def call(self, action: Action) -> enact.NPArray:
    api = PolicyAPI.current()
    return enact.NPArray(api.state.position)


@enact.contexts.register
@enact.typed_invokable(enact.NoneResource, enact.NoneResource)
@dataclasses.dataclass
class PolicyAPI(enact.AsyncInvokable, enact.contexts.Context):
  """Runs a policy and exposes an API to the game."""
  policy: enact.AsyncInvokable
  record_step: Step = dataclasses.field(default_factory=Step)

  def __post_init__(self):
    enact.contexts.Context.__init__(self)
    self.started = asyncio.Event()
    self.has_observation = asyncio.Event()
    self.has_action = asyncio.Event()
    self.is_done = asyncio.Event()
    self.game: Optional[Game] = None
    self.action: Optional[Action] = None

  @contextlib.contextmanager
  def connect_to(self, game: Game):
    self.game = game
    with self:
      try:
        yield
      finally:
        self.game = None

  @property
  def state(self) -> State:
    return self.game.state

  async def stop(self):
    """Stop the policy."""
    self.is_done.set()
  
  async def compute_actions(self, observations: State) -> Action:
    await asyncio.wait_for(self.started.wait(), timeout=1.0)
    self.has_observation.set()
    await self.has_action.wait()
    self.has_action.clear()
    assert self.action is not None
    return self.action

  async def init(self):
    self.started.set()
    await asyncio.wait_for(self.has_observation.wait(), timeout=1.0)

  async def step(self, action: Action):
    self.record_step(action)
    self.action = action
    self.has_action.set()
    await self.has_observation.wait()
    self.has_observation.clear()
    
  async def call(self):
    await self.init()
    policy_task = asyncio.create_task(self.policy())
    done, _ = await asyncio.wait(
      [policy_task, self.is_done.wait()],
      return_when=asyncio.FIRST_COMPLETED)
    if policy_task in done and not len(done) == 2:
      raise ValueError('Policy completed before game.')
    policy_task.cancel()
    try:
      await policy_task
    except asyncio.CancelledError:
      pass


@enact.typed_invokable(enact.NoneResource, PolicyAPI)
class CoroutinePolicy(enact.AsyncInvokable):
  def api(self) -> PolicyAPI:
    return PolicyAPI.current()
