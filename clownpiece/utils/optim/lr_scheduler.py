from .optimizer import Optimizer
from typing import List, Callable

class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.last_epoch = 0 if last_epoch == -1 else last_epoch
        self.optimizer = optimizer
        self.base_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        self.last_lr = self.base_lrs.copy()

    def get_lr(self) -> List[float]:        
        raise NotImplementedError

    def step(self, epoch: int = None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        new_lr = self.get_lr()
        i = 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr[i]
            i += 1
        self.last_lr = list(new_lr)
        
    def get_last_lr(self) -> List[float]:
       return self.last_lr
      
    
class LambdaLR(LRScheduler):
    """
    Lambda learning rate scheduler.
    Applies a user-defined function to the learning rate.
    """
    def __init__(self, optimizer, lr_lambda: Callable[[int], float], last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.lr_lambda = lr_lambda

    def get_lr(self) -> List[float]:
        c = self.lr_lambda(self.last_epoch)
        return [lr * c for lr in self.base_lrs]
    
class ExponentialLR(LRScheduler):
    """
    Exponential learning rate scheduler.
    Multiplies the learning rate by a factor every epoch.
    """
    
    def __init__(self, optimizer, gamma: float = 0.1, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        c = self.gamma ** self.last_epoch
        return [lr * c for lr in self.base_lrs]
        
class StepLR(LRScheduler):
    """
    Step learning rate scheduler.
    Decreases the learning rate by a factor every `step_size` epochs.
    """
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        c = self.gamma ** (self.last_epoch // self.step_size)
        return [lr * c for lr in self.base_lrs]