import dataclasses
import enact
import numpy as np

from async_api import CoroutinePolicy
from async_asteroids import Action

@enact.register
@dataclasses.dataclass
class AimAt(CoroutinePolicy):

  target: list

  def get_torque(self) -> float:
    """Produce a torque action to navigate the agent toward the target."""
    api = self.api()
    diff_vector = np.array(self.target)[0] - api.state.position[0]
    goal_direction = np.arctan2(diff_vector[1], diff_vector[0])
    delta_angle = np.arctan2(np.sin(goal_direction - api.state.rotation[0]),
                              np.cos(goal_direction - api.state.rotation[0]))
    normalized_delta_angle = delta_angle / np.pi
    return normalized_delta_angle

  def get_thrust(self) -> float:
    """Produce a thrust action to navigate the agent toward the target."""
    return 1.

  async def step(self, action: Action):
    """Step the API with the provided action."""
    api = self.api()
    await api.step(action)

  async def call(self):
    """Call the coroutine."""
    while True:
      torque = self.get_torque()
      thrust = self.get_thrust()

      action = Action(torque, thrust)
      await self.step(action)


@enact.typed_invokable(enact.NoneResource, enact.NoneResource)
class ChangeAim(CoroutinePolicy):

  async def call(self):
    print('Calling ChangeAim.')
    counter = 0
    while True:
      print(f'Counter: {counter}')
      counter += 1
      target = np.random.uniform(0, 25, ((1, 2)))
      aim_at = AimAt(target.tolist())
      await aim_at()
