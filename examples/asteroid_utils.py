import dataclasses
import io
import traceback

import re
import asteroids

import enact
import llm_task
import matplotlib.pyplot as plt

import PIL.Image

from typing import List, Optional

_BATCH_SHAPE = (1,)


@enact.typed_invokable(enact.Str, enact.Image)
@dataclasses.dataclass
class PolicyVisualizer(enact.Invokable):
  """Visualizes a policy provided (as a string)."""
  control_fn: str
  n_rows: int = 2
  n_cols: int = 2
  policy_steps: int = 300

  def call(self, code: enact.Str) -> enact.Image:
    """Plots policy trajectories."""
    def_dict = {}
    exec(code, def_dict)  # pylint: ignore
    control = def_dict[self.control_fn]

    _, axs = plt.subplots(self.n_rows, self.n_cols)

    for ax in axs.flatten():
      trajectory = asteroids.create_trajectory(
        control, _BATCH_SHAPE, steps=self.policy_steps)
      asteroids.plot_trajectory(trajectory, axis=ax)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    return enact.Image(PIL.Image.open(buffer))


@enact.typed_invokable(enact.Str, enact.Str)
@dataclasses.dataclass
class CreatePolicy(enact.Invokable):
  """Create a policy function from a task description."""
  instructions: str
  task_prompt: str
  func_name: str
  auxiliary_code: str = ''
  examples: List[llm_task.TaskExample] = dataclasses.field(
    default_factory=lambda: [
      llm_task.TaskExample(
        '''```def add(x: int, y: int):\n  <INSERT IMPLEMENTATION HERE>```''',
        '''```python\ndef add(x: int, y: int):\n  return x + y\n```''')])

  def call(self, task_description: enact.Str):
    policy_checker = PolicyChecker(self.func_name, self.auxiliary_code)
    full_prompt = self.instructions + '\n' + self.task_prompt
    code_gen = llm_task.Task(
      full_prompt, self.examples, post_processor=policy_checker)
    return code_gen(task_description)


@enact.typed_invokable(enact.Str, llm_task.ProcessedOutput)
@dataclasses.dataclass
class PolicyChecker(enact.Invokable):
  func_name: str
  auxiliary_code: str = ''

  def call(self, input: enact.Str) -> llm_task.ProcessedOutput:
    print(input)
    if not input.startswith('```python') and not input.startswith('```'):
      return llm_task.ProcessedOutput(
        output=None, correction=enact.Str('Input must start with "```".'))
    if not input.endswith('```'):
      return llm_task.ProcessedOutput(
        output=None, correction=enact.Str('Input must end with "```".'))
    if '```' in input[3:-3]:
      return llm_task.ProcessedOutput(
        output=None, correction=enact.Str('Input must be a single code block```".'))
    if input.startswith('```python'):
      code = input[len('```python`'):-len('```')]
    else:
      code = input[len('```'):-len('```')]
    def_dict = {}
    code = self.auxiliary_code + '\n' + code
    try:
      exec(code, def_dict)
    except Exception as e:
      print(traceback.format_exc())
      return llm_task.ProcessedOutput(
        output=None,
        correction=enact.Str(
          f'Your code raised an exception while parsing: {e}\n{traceback.format_exc()}'))
    control = def_dict.get(self.func_name)
    if not control:
      return llm_task.ProcessedOutput(
        output=None,
        correction=enact.Str(f'Your code did not define a `{self.func_name}` function.'))
    try:
      _ = control(asteroids.State(batch_shape=_BATCH_SHAPE))
    except Exception as e:
      print(traceback.format_exc())
      return llm_task.ProcessedOutput(
        output=None,
        correction=enact.Str(f'Your code raised an exception while running: {e}\n'
                             f'{traceback.format_exc()}'))
    critique = enact.RequestInput(
      enact.Str,
      'Please critique the policy or leave empty if ok.')(
        PolicyVisualizer(self.func_name)(enact.Str(code)))

    if critique != '':
      return llm_task.ProcessedOutput(
        output=None, correction=enact.Str(f'User critique: {critique}'))

    return llm_task.ProcessedOutput(
      output=code, correction=None)


@enact.register
@dataclasses.dataclass
class CodeGenTask(enact.Resource):
  func_name: str
  task_description: str

  def noop(self) -> bool:
    return not self.func_name and not self.task_description


@enact.typed_invokable(enact.NoneResource, enact.Str)
@dataclasses.dataclass
class RecursivePolicyGen(enact.Invokable):

  general_prompt: enact.Str
  source_file: Optional[enact.Str] = None

  # We need a way of discriminating between corrections and new tasks.
  def call(self) -> enact.Str:
    code: str = read_source_file(self.source_file) if self.source_file else ''
    while True:
      task_description: enact.Str = enact.request_input(
        enact.Str,
        context='Please specify the task to complete.')
      func_name = extract_func_name(task_description)
      #if func_name is None:
      code_gen = CodeGenTask(func_name, task_description)
      if code_gen.noop():
        break
      create_policy = CreatePolicy(
        self.general_prompt, code_gen.task_description, code_gen.func_name, code)
      code += '\n' + create_policy(code_gen.task_description) 
    return enact.Str(code)

def extract_func_name(task_str: enact.Str) -> Optional[enact.Str]:
  matches = re.findall('(def)\s(\w+)\([a-zA-Z0-9_:\[\]=, ]*\)', task_str)
  if matches:
    return enact.Str(matches[0][-1])
  return None

def read_source_file(filename: enact.Str) -> str:
  with open(filename, 'r') as f:
    source_str = f.read()
  return source_str