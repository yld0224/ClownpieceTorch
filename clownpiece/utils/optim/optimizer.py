from clownpiece.autograd import no_grad
from clownpiece.nn import Parameter
from clownpiece.nn.init import zeros_
from clownpiece.tensor import Tensor

from typing import List, Iterable, Dict, Any, Union

class Optimizer:
    param_groups: List[Dict[str, Any]]       # list of parameter groups
    state: Dict[Parameter, Dict[str, Any]]   # mapping param_id -> optimizer state
    defaults: Dict[str, Any]                 # default hyperparams for each group

    def __init__(self, parameters: Union[Iterable[Parameter], Iterable[Dict]], defaults: Dict[str, Any]):
        """
        - `parameters`: an iterable of `Parameter` objects or a list of dicts defining parameter groups.
            - if iterable of `Parameter`, add it as the first param_group.
        - `defaults`: a dict of default hyperparameters (e.g., learning rate).
        """
        self.param_groups = []
        self.state = {}
        self.defaults = defaults
        self._next_param_id = 0
        parameters = list(parameters)
        if isinstance(parameters[0], Parameter):
            self.add_param_group({"params": parameters})
        else:
            for param_group in parameters:
                self.add_param_group(param_group)

    def add_param_group(self, param_group: Dict[str, Any]):
        """Merge defaults into `param_group` and add to `param_groups`."""
        params = list(param_group["params"])
        param_group["params"] = params
        for parameter in params:
            parameter.param_id = self._next_param_id
            self._next_param_id += 1
        for key, value in self.defaults.items():
            param_group.setdefault(key, value)
        self.param_groups.append(param_group)

    def step(self):
        """Perform a single optimization step (update all parameters).
        Must be implemented by subclasses."""
        raise NotImplementedError

    def zero_grad(self, set_to_None: bool = True):
        """Reset gradients for all parameters."""
        for param_group in self.param_groups:
            parameters = param_group["params"]
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                if set_to_None:
                    parameter.grad = None
                else:
                    zeros_(parameter.grad)
  
class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.0, damping: float = 0.0, weight_decay: float = 0.0):
        defaults = {"lr": lr, "momentum": momentum, "damping": damping, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        with no_grad():
            for param_group in self.param_groups:
                lr = param_group["lr"]
                momentum = param_group["momentum"]
                damping = param_group["damping"]
                weight_decay = param_group["weight_decay"]
                for parameter in param_group["params"]:
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad
                    if weight_decay != 0:
                        grad = grad + weight_decay * parameter
                    if momentum != 0:
                        state = self.state.setdefault(parameter.param_id, {})
                        if "momentum_buffer" not in state:
                            velocity = (1 - damping) * grad
                        else:
                            previous_velocity = state["momentum_buffer"]
                            velocity = (momentum * previous_velocity + (1 - damping) * grad)
                        state["momentum_buffer"] = velocity
                        update = velocity
                    else:
                        update = grad
                    parameter.copy_(parameter - lr * update)

class Adam(Optimizer):
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
    super().__init__(params, defaults)

  def step(self):
    with no_grad():
        for param_group in self.param_groups:
            lr = param_group["lr"]
            beta1, beta2 = param_group["betas"]
            eps = param_group["eps"]
            weight_decay = param_group["weight_decay"]
            for parameter in param_group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if weight_decay != 0:
                    grad = grad + weight_decay * parameter
                state = self.state.setdefault(parameter.param_id, {})
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = Tensor.zeros_like(grad)
                    state["exp_avg_sq"] = Tensor.zeros_like(grad)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg = (beta1 * exp_avg + (1 - beta1) * grad)
                exp_avg_sq = (beta2 * exp_avg_sq + (1 - beta2) * (grad * grad))
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                update = (exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps))
                parameter.copy_(parameter - lr * update)