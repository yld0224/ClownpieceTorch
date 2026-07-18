# MSE, CrossEntropy

from clownpiece.nn.module import Module
from clownpiece import Tensor
from clownpiece.autograd import no_grad


# loss.py
class MSELoss(Module):
  def __init__(self, reduction: str = 'mean'):
    super().__init__()
    self.reduction = reduction

  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    loss = (input - target) * (input - target)
    if self.reduction == "none":
      return loss
    result = loss
    for _ in range(len(loss.shape)):
      result = result.sum(0)
    if self.reduction == "sum":
      return result
    return result / len(loss)

class CrossEntropyLoss(Module):
  def __init__(self, reduction: str = 'mean'):
    super().__init__()
    self.reduction = reduction

  def forward(self, logits: Tensor, target: Tensor) -> Tensor:
    with no_grad():
      max_logits, _ = logits.max(-1, True)
    shifted = logits - max_logits
    log_softmax = shifted - shifted.exp().sum(-1, True).log()
    indices = target.unsqueeze(-1)
    one_hot = Tensor.zeros(logits.shape, requires_grad = False)
    one_hot.scatter_(-1, indices, Tensor.ones(indices.shape, requires_grad = False))
    loss = -(log_softmax * one_hot).sum(-1)
    if self.reduction == "none":
      return loss
    result = loss
    for _ in range(len(loss.shape)):
      result = result.sum(0)
    if self.reduction == "sum":
      return result
    return result / len(loss)