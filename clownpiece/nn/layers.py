# Linear, Embedding, LayerNorm, BatchNorm, MultiheadAttention

from typing import Optional
from clownpiece.tensor import Tensor
from clownpiece.nn.module import Module, Parameter, Buffer
from clownpiece.autograd import no_grad
from . import init
import math



class Linear(Module):

  def __init__(self, in_features: int, out_features: int, bias: bool=True):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(Tensor.zeros([out_features, in_features]))
    bound = 1 / math.sqrt(self.in_features)
    init.uniform_(self.weight, -bound, bound)
    if bias:
      self.bias = Parameter(Tensor.zeros([out_features]))
      init.uniform_(self.bias, -bound, bound)
    else:
      self.register_parameter("bias", None)
    

  def forward(self, x: Tensor) -> Tensor:
    out =  x @ self.weight.transpose(-1, -2)
    if self.bias is not None:
      out = out + self.bias
    return out

  def extra_repr(self):
    return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}")


class Embedding(Module):
    def __init__(self, num_embd: int, embd_dim: int):
      super().__init__()
      self.num_embd = num_embd
      self.embd_dim = embd_dim
      self.weight = Parameter(Tensor.zeros([num_embd, embd_dim]))
      init.normal_(self.weight, 0, 1.0)

    def forward(self, x: Tensor) -> Tensor:
      one_hot_shape = list(x.shape) + [self.num_embd]
      one_hot = Tensor.zeros(one_hot_shape)
      one_hot.scatter_(-1, x, Tensor.ones(x.shape))
      return one_hot @ self.weight
    
class LayerNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
      super().__init__()
      self.num_features = num_features
      self.eps = eps
      self.affine = affine
      if affine:
        self.weight = Parameter(Tensor.ones([num_features]))
        self.bias = Parameter(Tensor.zeros([num_features]))
      else:
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
      ori_shape = x.shape
      reshaped = x.reshape([-1, self.num_features])
      mean = reshaped.mean(-1, True)
      var = reshaped.var(-1, True, False)
      output = (reshaped - mean) / (var + self.eps).sqrt()
      if self.affine:
        output = output * self.weight + self.bias
      return output.reshape(ori_shape)

class BatchNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
      super().__init__()
      self.num_features = num_features
      self.eps = eps
      self.momentum = momentum
      self.affine = affine
      if affine:
        self.weight = Parameter(Tensor.ones([num_features]))
        self.bias = Parameter(Tensor.zeros([num_features]))
      else:
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
      self.running_mean = Buffer(Tensor.zeros([num_features]))
      self.running_var = Buffer(Tensor.ones([num_features]))

    def forward(self, x: Tensor) -> Tensor:
      ori_shape = x.shape
      x = x.reshape([-1, self.num_features])
      if self.training:
        mean = x.mean(0, True)
        var = x.var(0, True, False)
        batch_mean = mean.reshape([self.num_features])
        batch_var = var.reshape([self.num_features])
        with no_grad():
          new_running_mean = ((1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean)
          new_running_var = ((1.0 - self.momentum) * self.running_var + self.momentum * batch_var)
          self.running_mean.copy_(new_running_mean)
          self.running_var.copy_(new_running_var)
      else:
        mean = self.running_mean
        var = self.running_var
      output = (x - mean) / (var + self.eps).sqrt()
      if self.affine:
        output = self.weight * output + self.bias
      return output.reshape(ori_shape)
      
    
class MultiheadAttention(Module):
    def __init__(self, hidden_dim: int, num_heads: int, bias: bool = True):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.num_heads = num_heads
      self.head_dim = hidden_dim // num_heads
      self.q_proj = Linear(hidden_dim, hidden_dim, bias)
      self.k_proj = Linear(hidden_dim, hidden_dim, bias)
      self.v_proj = Linear(hidden_dim, hidden_dim, bias)
      self.out_proj = Linear(hidden_dim, hidden_dim, bias)

    def forward(self, hidden_states: Tensor, attn_mask: Optional[Tensor] = None):
      batch_size = hidden_states.shape[0]
      seq_len = hidden_states.shape[1]
      q = self.q_proj(hidden_states)
      k = self.k_proj(hidden_states)
      v = self.v_proj(hidden_states)
      q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
      k = k.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
      v = v.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
      q = q.transpose(-3, -2)
      k = k.transpose(-3, -2)
      v = v.transpose(-3, -2)
      scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
      if attn_mask is not None:
        scores = scores + attn_mask
      with no_grad():
        max_scores, _ = scores.max(-1, True)
      attention = (scores - max_scores).softmax(-1)
      output = attention @ v
      output = output.transpose(-3, -2)
      output = output.reshape([batch_size, seq_len, self.hidden_dim])
      return self.out_proj(output)