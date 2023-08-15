import dataclasses
import io
import traceback

import re
import asteroids

import enact
import llm_task
import matplotlib.pyplot as plt  # type: ignore

import PIL.Image

from typing import Any, Dict, List, Optional

_BATCH_SHAPE = (1,)
_MAX_RETRIES = 5
_BACKTICKS_MD = '```'
_PROGRAM_START_MD = (_BACKTICKS_MD + 'python', _BACKTICKS_MD)


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
    def_dict: Dict[str, Any] = {}
    exec(code, def_dict)  # pylint: ignore
    policy = def_dict[self.control_fn]

    _, axs = plt.subplots(self.n_rows, self.n_cols)

    for ax in axs.flatten():
      trajectory = asteroids.create_trajectory(
        policy, _BATCH_SHAPE, steps=self.policy_steps)
      asteroids.plot_trajectory(trajectory, axis=ax)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    return enact.Image(PIL.Image.open(buffer))


@enact.typed_invokable(enact.Str, enact.Str)
@dataclasses.dataclass
class CreatePolicy(enact.Invokable):
  """Create a policy function from a task description."""
  general_prompt: str
  func_name: str
  auxiliary_code: str = ''
  examples: List[llm_task.TaskExample] = dataclasses.field(
    default_factory=lambda: [
      llm_task.TaskExample(
        '''```def add(x: int, y: int):\n  <INSERT IMPLEMENTATION HERE>```''',
        '''```python\ndef add(x: int, y: int):\n  return x + y\n```''')])

  def call(self, task_description: enact.Str):
    policy_checker = PolicyChecker(self.func_name, self.auxiliary_code)
    full_prompt = self.general_prompt + '\n' + task_description
    code_gen = llm_task.Task(
      full_prompt, self.examples, post_processor=policy_checker, max_retries=_MAX_RETRIES)
    return code_gen(task_description)


@enact.typed_invokable(enact.Str, llm_task.ProcessedOutput)
@dataclasses.dataclass
class PolicyChecker(enact.Invokable):
  func_name: str
  auxiliary_code: str = ''

  def _get_formatting_errors_correction(self, input: enact.Str) -> Optional[enact.Str]:
    """Return correction string if the input is improperly formatted."""
    if not all(map(input.startswith, _PROGRAM_START_MD)):
      return enact.Str(f'Input must start with "{_BACKTICKS_MD}".')
    if not input.endswith(_BACKTICKS_MD):
      return enact.Str(f'Input must end with "{_BACKTICKS_MD}".')
    if _BACKTICKS_MD in input[3:-3]:
      return enact.Str(f'Input must be a single code block using "{_BACKTICKS_MD}".')
    return None

  def _sanitize_code_str(self, input: enact.Str) -> str:
    """Convert the input to an executable Python string."""
    for program_start in _PROGRAM_START_MD:
      if input.startswith(program_start):
        code = input[len(program_start):-len(_BACKTICKS_MD)]
        break
    return self.auxiliary_code + '\n' + code

  def _get_execution_errors_correction(self, code: str) -> Optional[enact.Str]:
    """Return correction string if code execution produces errors."""
    def_dict: Dict[str, Any] = {}
    try:
      exec(code, def_dict)
    except Exception as e:
      print(traceback.format_exc())
      return enact.Str(
        f'Your code raised an exception while parsing: {e}\n{traceback.format_exc()}')
    func = def_dict.get(self.func_name)
    if not func:
      return enact.Str(f'Your code did not define a `{self.func_name}` function.')
    try:
      func(asteroids.State(batch_shape=_BATCH_SHAPE))
    except Exception as e:
      print(traceback.format_exc())
      return enact.Str(f'Your code raised an exception while running: {e}\n'
                       f'{traceback.format_exc()}')
    return None

  def _get_execution_critique_correction(self, code: str) -> Optional[enact.Str]:
    """Return correction string if the user provides a policy critique."""
    critique = enact.RequestInput(
      enact.Str,
      'Please critique the policy or leave empty if ok.')(
        PolicyVisualizer(self.func_name)(enact.Str(code)))
    if critique == '':
      return None
    return enact.Str(f'User critique: {critique}')

  def call(self, input: enact.Str, prompt_for_user_critique: bool = False) -> (
      llm_task.ProcessedOutput):
    """Process the provided input."""
    output = None
    correction = self._get_formatting_errors_correction(input)

    # Try to execute code if no formatting corrections.
    if correction is None:
      code = self._sanitize_code_str(input)
      correction = self._get_execution_errors_correction(code)

    # Check for user corrections.
    if prompt_for_user_critique:
      if correction is None:
        correction = self._get_execution_critique_correction(code)

    # Set the output if there are no corrections.
    if correction is None:
      output = enact.Str(code)

    return llm_task.ProcessedOutput(output=output, correction=correction)


@enact.typed_invokable(enact.NoneResource, enact.Str)
@dataclasses.dataclass
class RecursivePolicyGen(enact.Invokable):

  general_prompt: enact.Str
  top_level_policy_fn: enact.Str
  source_file: Optional[enact.Str] = None
  code: str = ''
  code_stubs: Dict[enact.Str, str] = dataclasses.field(default_factory=dict)
  code_book: Dict[enact.Str, str] = dataclasses.field(default_factory=dict)

  def _get_context(self) -> enact.Str:
    if not self.code_stubs:
      return enact.Str('')
    prompt_prefix = (
      '\nTo generate this function, you can call the following functions if needed: \n')
    stubs = '\n'.join(self.code_stubs.values())
    prompt_suffix = (
      'Assume all of these functions are already defined and directly callable.')
    return enact.Str(prompt_prefix + stubs + prompt_suffix)

  def _update_code_stubs(self, func_name: enact.Str, code_str: str):
    # We assume the code definition starts with '```', so we remove these
    # characters.
    code_str = code_str[len(_BACKTICKS_MD):]
    code_stub = code_str.split('<INSERT IMPLEMENTATION HERE>')[0]
    self.code_stubs[func_name] = code_stub

  def _update_code_book(self, func_name: enact.Str, code_str: str):
    self.code_stubs[func_name] = code_str

  def call(self, unused_resource: enact.NoneResource) -> enact.Str:
    if self.code == '' and self.source_file:
      self.code = read_source_file(self.source_file)
    while True:
      task_description: enact.Str = enact.request_input(
        enact.Str,
        context='Please specify the task to complete.')
      if not task_description:
        break
      func_name = extract_func_name(task_description)
      if not func_name:
        break
      code_context = self._get_context()
      create_policy = CreatePolicy(
        self.general_prompt, func_name, self.code)
      generated_code = create_policy(task_description + code_context)
      self._update_code_stubs(func_name, task_description)
      self._update_code_book(func_name, generated_code)
      self.code += '\n' + generated_code
    return enact.Str(self.code)

def extract_func_name(code_str: enact.Str) -> Optional[enact.Str]:
  """Extract the function name from a string."""
  matches = re.findall('(def)\s(\w+)\([a-zA-Z0-9_:\[\]=, ]*\)', code_str)
  if matches:
    return enact.Str(matches[0][-1])
  return None

def read_source_file(filename: enact.Str) -> str:
  """Produce a string from a provided file."""
  with open(filename, 'r') as f:
    source_str = f.read()
  return source_str