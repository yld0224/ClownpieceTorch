# Sequential, ModuleList, ModuleDict

from typing import Iterable, Dict, Tuple
from clownpiece.nn.module import Module
class Sequential(Module):
  
  def __init__(self, *modules: Module):
    super().__init__()
    for i in range(len(modules)):
      self.__setattr__(str(i), modules[i])

  def forward(self, input):
    for module in self._modules.values():
      input = module(input)
    return input


class ModuleList(Module):
  
  def __init__(self, modules: Iterable[Module] = None):
    super().__init__()
    if modules is not None:
      self.extend(modules)


  def __add__(self, other: Iterable[Module]):
    return ModuleList(list(self) + list(other))

  def __setitem__(self, index: int, value: Module):
    if index < 0: index += len(self)
    if index < 0 or index > len(self): raise IndexError("ModuleList idx out of bound")
    self.__setattr__(str(index), value)

  def __getitem__(self, index: int) -> Module:
    if index < 0: index += len(self)
    if index < 0 or index > len(self): raise IndexError("ModuleList idx out of bound")
    return self._modules[str(index)]

  def __delitem__(self, index: int):
    modules = list(self._modules.values())
    del modules[index]
    self._modules.clear()
    self.extend(modules)

  def __len__(self):
    return len(self._modules)

  def __iter__(self) -> Iterable[Module]:
    return iter(self._modules.values())

  def append(self, module: Module):
    self.__setattr__(str(len(self)), module)
    return self

  def extend(self, other: Iterable[Module]):
    modules = list(other)
    for module in modules:
      self.append(module)
    return self

class ModuleDict(Module):
  
  def __init__(self, dict_: Dict[str, Module]):
    super().__init__()
    self.update(dict_)

  def __setitem__(self, name: str, value: Module):
    self.register_module(name, value)

  def __getitem__(self, name: str) -> Module:
    return self._modules[name]

  def __delitem__(self, name: str):
    del self._modules[name]

  def __len__(self):
    return len(self._modules)

  def __iter__(self) -> Iterable[str]:
    return iter(self._modules)
  
  def keys(self) -> Iterable[str]:
    return self._modules.keys()

  def values(self) -> Iterable[Module]:
    return self._modules.values()

  def items(self) -> Iterable[Tuple[str, Module]]:
    return self._modules.items()

  def update(self, dict_: Dict[str, Module]):
    for key, value in dict_.items():
      self.__setattr__(key, value)