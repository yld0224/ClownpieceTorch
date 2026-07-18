# Core Module System

from typing import Dict, Iterable, Tuple, Union, Optional
from clownpiece import Tensor, zeros_like
from clownpiece.tensor import empty_like


class Parameter(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=True)
    

class Buffer(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=False)


class Module(object):
  def __init__(self):
    super().__init__()
    object.__setattr__(self, "training", True)
    object.__setattr__(self, "_parameters", {})
    object.__setattr__(self, "_buffers", {})
    object.__setattr__(self, "_modules", {})

  def train(self, flag: bool = True):
    self.training = flag
    for module in self._modules.values():
      if module is not None:
        module.train(flag)
    return self

  def eval(self):
    return self.train(False)

  def __setattr__(self, name, value):
    if isinstance(value, Parameter):
      self._parameters[name] = value
    elif isinstance(value, Buffer):
      self._buffers[name] = value
    elif isinstance(value, Module):
      self._modules[name] = value
    else:
      object.__setattr__(self, name, value)

  def __getattr__(self, name):
    if name in self._parameters:
      return self._parameters[name]
    elif name in self._buffers:
      return self._buffers[name]
    elif name in self._modules:
      return self._modules[name]
    else:
      raise AttributeError("Attribute does not exist")
    
  """
    Forward
  """
    
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  """
    Parameter
  """
  def register_parameter(self, name: str, param: Optional[Parameter]):
    self._parameters[name] = param

  def parameters(self, recursive: bool=True) -> Iterable[Parameter]:
    for param in self._parameters.values():
      if param is not None:
        yield param
    if recursive:
      for module in self._modules.values():
          if module is not None:
            yield from module.parameters(True)

  def named_parameters(self, recursive: bool=True) -> Iterable[Tuple[str, Parameter]]:
    def traverse(module: Module, prefix: str):
      for name, param in module._parameters.items():
        if param is not None:
          full_name = f"{prefix}.{name}" if prefix else name
          yield (full_name, param)
      if recursive:
        for name, child in module._modules.items():
          if child is not None:
            pre = f"{prefix}.{name}" if prefix else name
            yield from traverse(child, pre)
    yield from traverse(self, "")

  """
    Buffer
  """

  def register_buffer(self, name: str, buffer: Optional[Buffer]):
    self._buffers[name] = buffer

  def buffers(self, recursive: bool=True) -> Iterable[Buffer]:
    for buffer in self._buffers.values():
      if buffer is not None:
        yield buffer
    if recursive:
      for module in self._modules.values():
        if module is not None:
          yield from module.buffers(True)

  def named_buffers(self, recursive: bool=True) -> Iterable[Tuple[str, Buffer]]:
    def traverse(module: Module, prefix: str):
      for name, buff in module._buffers.items():
        if buff is not None:
          full_name = f"{prefix}.{name}" if prefix else name
          yield (full_name, buff)
      if recursive:
        for name, child in module._modules.items():
          if child is not None:
            pre = f"{prefix}.{name}" if prefix else name
            yield from traverse(child, pre)
    yield from traverse(self, "")

  """
    Modules
  """

  def register_module(self, name: str, module: Optional["Module"]):
    self._modules[name] = module

  def modules(self, recursive: bool=True) -> Iterable["Module"]:
    yield self
    if recursive:
      for module in self._modules.values():
        if module is not None:
          yield from module.modules(True)

  def named_modules(self, recursive: bool=True) -> Iterable[Tuple[str, "Module"]]:
    def traverse(module, prefix):
      yield (prefix, module)
      if recursive:
        for name, child in module._modules.items():
          if child is not None:
            full_name = f"{prefix}.{name}" if prefix else name
            yield from traverse(child, full_name)
    yield from traverse(self, "")
    
  """
    State Dict
  """

  def state_dict(self) -> Dict[str, Optional[Tensor]]:
    state = {}
    for module_name, module in self.named_modules():
      for name, param in module._parameters.items():
        full_name = f"{module_name}.{name}" if module_name else name
        state[full_name] = param
      for name, buffer in module._buffers.items():
        full_name = f'{module_name}.{name}' if module_name else name
        state[full_name] = buffer
    return state

  def load_state_dict(self, state: Dict[str, Tensor], strict: bool = True):
    current_state = self.state_dict()
    expected_keys = set(current_state.keys())
    provided_keys = set(state.keys())
    if strict:
      missing_keys = expected_keys - provided_keys
      unexpected_keys = provided_keys - expected_keys
      if missing_keys: raise RuntimeError(f"Missing keys: {sorted(missing_keys)}")
      if unexpected_keys: raise RuntimeError(f"Unexpected keys: {sorted(unexpected_keys)}")
    common_keys = expected_keys & provided_keys
    for name in common_keys:
      target = current_state[name]
      source = state[name]
      if target is None or source is None:
        if target is not None or source is not None:
          raise RuntimeError(f"State mismatch for {name}")
        continue
      if target.shape != source.shape:
        raise RuntimeError(f"Shape mismatch for {name}: expected {target.shape}, got {source.shape}")
    for name in common_keys:
      target = current_state[name]
      source = state[name]
      if target is not None:
        target.copy_(source)
    
  """
    Printing
  """
  def __repr__(self) -> str:
    class_name = self.__class__.__name__
    extra = self.extra_repr()
    children = [(name, module) for name, module in self._modules.items() if module is not None]
    if not children:
      return f"{class_name}({extra})" if extra else f"{class_name}()"
    lines = []
    if extra:
      lines.extend(f"  {line}" for line in extra.splitlines())
    for name, module in children:
      child_repr = repr(module).replace("\n", "\n  ")
      lines.append(f"  ({name}): {child_repr}")
    return f"{class_name}(\n" + "\n".join(lines) + "\n)"

  def extra_repr(self) -> str:
    return ""