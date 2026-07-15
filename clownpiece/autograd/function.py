"""
    Implement Various Functions
"""

from typing import List, Union
import copy

from clownpiece.tensor import Tensor, zeros, zeros_like
from clownpiece.autograd.autograd import Node, Edge
from clownpiece.autograd.no_grad import no_grad
from clownpiece.utils_ import wrap_tuple


class Context():
    def __init__(self):
        self.saved_tensors = []
        
    def save_for_backward(self, *args) -> None:
        self.saved_tensors.extend(
            [self.repack_tensor(tensor) for tensor in args if isinstance(tensor, Tensor)]
        )
        
    def get_saved_tensors(self) -> List[Tensor]:
        return self.saved_tensors
    
    @staticmethod
    def repack_tensor(tensor: Tensor):
        # avoid cyclic reference
        if isinstance(tensor, Tensor):
            return copy.copy(tensor) # shallow copy
        else:
            return tensor
    

class Function(Node):
    """
    Base class for all functions.
    """
    ctx: Context
    
    def __init__(self):
        super().__init__()
        self.ctx = None
        
    @staticmethod
    def forward(ctx: Context, *args):
        raise NotImplementedError("Forward method not implemented")

    @staticmethod
    def backward(ctx: Context, *args):
        raise NotImplementedError("Backward method not implemented")    
    
    # run forward pass
    def apply(self, *args, **kwargs):
        self.ctx = Context()
        self.next_edges = [Edge.gradient_edge(arg) for arg in args if isinstance(arg, Tensor)]
        with no_grad():
            outputs = self.forward(self.ctx, *args, **kwargs)
        for output_nr, output in enumerate(wrap_tuple(outputs)):
            if isinstance(output, Tensor):
                output.grad_fn = self
                output.output_nr = output_nr
                output.requires_grad_(True)
        return outputs

    
    # run backward pass
    def run(self, *args):
        with no_grad():
            grad_inputs = self.backward(self.ctx, *args)
        return grad_inputs

class AccumulateGrad(Function):
    """
    Accumulate gradient to .grad field
    
    grad_fn for leaf tensors
    """
    def __init__(self, input: Tensor):
        super().__init__() 
        self.tensor = input
    
    @staticmethod
    def forward(ctx: Context):
        return None
    
    def backward(self, ctx: Context, output_grad: Tensor):
        if self.tensor.grad is None:
            self.tensor.grad = output_grad.clone()
        else:
            self.tensor.grad += output_grad
        return

      

"""
    Clone Contiguous
"""

class Clone(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.clone()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output

class Contiguous(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.contiguous()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output
    
"""
    Subscriptor
"""

class Subscriptor(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, index_or_slice: Union[int, slice, List[int], List[slice]]):
        ctx.input_shape = input.shape
        ctx.index_or_slice = index_or_slice
        return input[index_or_slice]
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        grad_input = zeros(ctx.input_shape)
        grad_input[ctx.index_or_slice].copy_(grad_output)
        return grad_input
    
"""
    Element-wise Binary and Unary Operators
"""

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.__neg__()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.__neg__()

# backward method for broadcast
def reduce_broadcast(grad_output: Tensor, input_shape: List[int], output_shape: List[int], end_dim: int = 0) -> Tensor:
    offset = len(output_shape) - len(input_shape)
    result = grad_output
    for i in range(offset):
        result = result.sum(0, False)
    for i in range(len(input_shape) - end_dim):
        if input_shape[i] == 1 and output_shape[i + offset] != 1:
            result = result.sum(i, True)
    return result

# binary op forward decorator
def binary_op_forward_wrapper(forward_impl):
    def wrapped_forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        ctx.input_shapes = (list(input1.shape), list(input2.shape))
        output = forward_impl(ctx, input1, input2)
        ctx.output_shape = list(output.shape)
        return output
    return wrapped_forward


# binary op backward decorator
def binary_op_backward_wrapper(backward_impl):
    def wrapped_backward(ctx: Context, grad_output: Tensor):
        grad_inputs = wrap_tuple(backward_impl(ctx, grad_output))
        return (reduce_broadcast(grad_inputs[0], ctx.input_shapes[0], ctx.output_shape), 
                reduce_broadcast(grad_inputs[1], ctx.input_shapes[1], ctx.output_shape))
    return wrapped_backward


class Add(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1.__add__(input2)
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        return (grad_output, grad_output)
    
class Sub(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1.__sub__(input2)
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        return (grad_output, grad_output.__neg__())
    
class Mul(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1.__mul__(input2)
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        input1, input2 = ctx.get_saved_tensors()
        return (grad_output.__mul__(input2), grad_output.__mul__(input1))
    
class Div(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1.__truediv__(input2)
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        input1, input2 = ctx.get_saved_tensors()
        grad_input1 = grad_output / input2
        grad_input2 = -grad_output * input1 / (input2 * input2)
        return (grad_input1, grad_input2)
    
class Sign(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.sign()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return zeros_like(grad_output)
    
class Abs(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.abs()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * input.sign()
    
class Sin(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.sin()
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * input.cos()

class Cos(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.cos()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * (-input.sin())

class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        out = input.tanh()
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        out, = ctx.get_saved_tensors()
        return grad_output * (1 - out * out)

class Clamp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, min_val: float, max_val: float):
        ctx.min_val = min_val
        ctx.max_val = max_val
        out = input.clamp(min_val, max_val)
        ctx.save_for_backward(input)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        mask = (input >= ctx.min_val) * (input <= ctx.max_val)
        return grad_output * mask

class Log(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.log()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * (1 / input)

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        out = input.exp()
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * input

class Pow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, exponent: float): 
        ctx.save_for_backward(input)
        ctx.exp = exponent
        return input.pow(exponent)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * ctx.exp * (input.pow(ctx.exp - 1))
    
class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        out = input.sqrt()
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * (1 / (2 * input))
    
"""
    Matrix Multiplication
"""

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        return input1.matmul(input2)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        A, B = ctx.get_saved_tensors()
        A1 = A.unsqueeze(0) if len(A.shape) == 1 else A
        B1 = B.unsqueeze(1) if len(B.shape) == 1 else B
        C1 = grad_output
        if len(A.shape) == 1 and len(B.shape) == 1:
            C1 = grad_output.reshape([1, 1])
        elif len(A.shape) == 1:
            C1 = grad_output.unsqueeze(-2)
        elif len(B.shape) == 1:
            C1 = grad_output.unsqueeze(-1)
        grad_A1 = C1.matmul(B1.transpose(-1, -2))
        grad_B1 = A1.transpose(-1, -2).matmul(C1)
        grad_A1 = reduce_broadcast(grad_A1, A1.shape, grad_A1.shape, end_dim = 2)
        grad_B1 = reduce_broadcast(grad_B1, B1.shape, grad_B1.shape, end_dim = 2)
        grad_A = grad_A1.squeeze(-2) if len(A.shape) == 1 else grad_A1
        grad_B = grad_B1.squeeze(-1) if len(B.shape) == 1 else grad_B1
        return (grad_A, grad_B)

"""
    Reduction and Normalization Operations
"""

def reduce_forward_wrapper(forward_impl):
    pass

class Sum(Function):
    @staticmethod
    @reduce_forward_wrapper
    def forward(ctx: Context, input: Tensor, dim: Union[int, List[int], None], keepdims: bool = False):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Max(Function):
    @staticmethod
    @reduce_forward_wrapper
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor, grad_indices: Tensor = None):
        pass
    
class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
"""
    Shape Manipulation
"""

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, perm: List[int]):
        pass

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim0: int, dim1: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class View(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Narrow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, start: int, length: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Chunk(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, chunks: int, dim: int = 0):
        pass
        
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass
    
class Split(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, split: Union[int, List[int]], dim: int = 0):
        pass

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass
    
class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        pass
        
class Squeeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):    
        pass
    
class Unsqueeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class BroadcastTo(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Broadcast(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass