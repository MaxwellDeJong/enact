import traceback

import io
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import llm_task
import enact


def get_policy_visualizer(Action, create_trajectory, plot_trajectory, control_fn='control'):

  @enact.typed_invokable(enact.Str, enact.Image)
  class PolicyVisualizer(enact.Invokable):
    """Visualizes a policy provided (as a string)."""

    def call(self, code: enact.Str) -> enact.Image:
      """Plots policy trajectories."""
      def_dict = {}
      exec(code, def_dict)
      control = def_dict[control_fn]
      def policy(state):
        if control_fn == 'control_two_goal':
          return Action(np.array(list(control(
            state.position[0],
            state.goal1_position[0],
            state.goal2_position[0],
            state.rotation[0])))[np.newaxis])
        if control_fn == 'control_one_goal':
          return Action(np.array(list(control(
            state.position[0],
            state.goal_position[0],
            state.rotation[0])))[np.newaxis])
        return Action(np.array(list(control(
          state.position[0],
          state.goal1_position[0],
          state.has_been_at_goal1[0],
          state.goal2_position[0],
          state.has_been_at_goal2[0],
          state.rotation[0])))[np.newaxis])



      _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

      for ax in [ax1, ax2, ax3, ax4]:
        t = create_trajectory(policy, (1,), steps=300)
        plot_trajectory(t, axis=ax)
      b = io.BytesIO()
      plt.savefig(b, format='png')
      return enact.Image(PIL.Image.open(b))

  return PolicyVisualizer

#def get_policy_checker(policy_visualizer, sample_input, func_name='control'):
def get_policy_checker(Action, create_trajectory, plot_trajectory, sample_input, func_name='control', code_reference=''):

  @enact.typed_invokable(enact.Str, llm_task.ProcessedOutput)
  class PolicyChecker(enact.Invokable):

    def call(self, input: enact.Str) -> llm_task.ProcessedOutput:
      if not input.startswith('```python') and not input.startswith('```'):
        return llm_task.ProcessedOutput(
          output=None, correction='Input must start with "```".')
      if not input.endswith('```'):
        return llm_task.ProcessedOutput(
          output=None, correction='Input must end with "```".')
      if '```' in input[3:-3]:
        return llm_task.ProcessedOutput(
          output=None, correction='Input must be a single code block```".')
      if input.startswith('```python'):
        code = input[len('```python`'):-len('```')]
      else:
        code = input[len('```'):-len('```')]
      def_dict = {}
      code = code_reference + code
      try:
        exec(code, def_dict)
      except Exception as e:
        return llm_task.ProcessedOutput(
          output=None,
          correction=f'Your code raised an exception while parsing: {e}\n{traceback.format_exc()}')
      control = def_dict.get(func_name)
      if not control:
        return llm_task.ProcessedOutput(
          output=None,
          correction=f'Your code did not define a `{func_name}` function.')
      try:
        result = control(*sample_input)
      except Exception as e:
        print(traceback.format_exc())
        return llm_task.ProcessedOutput(
          output=None,
          correction=f'Your code raised an exception while running: {e}\n{traceback.format_exc()}')
      try:
        thrust, torque = result
        thrust = float(thrust)
        torque = float(torque)
      except TypeError:
        return llm_task.ProcessedOutput(
          output=None,
          correction='Your code could not be unpacked into two float values.')
      critique = enact.RequestInput(enact.Str, 'Please critique the policy or leave empty if ok.')(
        get_policy_visualizer(Action, create_trajectory, plot_trajectory, func_name)()(enact.Str(code)))

      if critique != '':
        return llm_task.ProcessedOutput(
          output=None, correction=f'User critique: {critique}')

      return llm_task.ProcessedOutput(
        output=code, correction=None)

  return PolicyChecker
