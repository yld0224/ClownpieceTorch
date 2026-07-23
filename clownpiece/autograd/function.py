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

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Union[int, List[int], None], keepdims: bool = False):
        input_shape = list(input.shape)
        ndim = len(input_shape)
        if dim is None: dims = list(range(ndim))
        elif isinstance(dim, int): dims = [dim]
        else: dims = list(dim)
        normalized_dims = []
        for current_dim in dims:
            if current_dim < 0:
                current_dim += ndim
            if current_dim < 0 or current_dim >= ndim:
                raise IndexError("dimension out of range")
            normalized_dims.append(current_dim)
        ctx.input_shape = input_shape
        ctx.dims = normalized_dims
        return input.sum(normalized_dims, keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        reduced_shape = [1 if dim in ctx.dims else size for dim, size in enumerate(ctx.input_shape)]
        return grad_output.reshape(reduced_shape).broadcast_to(ctx.input_shape)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        input_shape = list(input.shape)
        ndim = len(input_shape)
        normalized_dim = dim + ndim if dim < 0 else dim
        if normalized_dim < 0 or normalized_dim >= ndim:
            raise IndexError("dimension out of range")
        values, indices = input.max(normalized_dim, keepdims)
        ctx.input_shape = input_shape
        ctx.dim = normalized_dim
        ctx.keepdims = keepdims
        ctx.save_for_backward(indices)
        return values, indices

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor, grad_indices: Tensor = None):
        indices, = ctx.get_saved_tensors()
        if not ctx.keepdims:
            grad_output = grad_output.unsqueeze(ctx.dim)
            indices = indices.unsqueeze(ctx.dim)
        grad_input = zeros(ctx.input_shape)
        grad_input.scatter_(ctx.dim, indices, grad_output)
        return grad_input

class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int):
        ndim = len(input.shape)
        normalized_dim = dim + ndim if dim < 0 else dim
        if normalized_dim < 0 or normalized_dim >= ndim:
            raise IndexError("dimension out of range")
        out = input.softmax(dim)
        ctx.dim = normalized_dim
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        s, = ctx.get_saved_tensors()
        dot = (grad_output * s).sum(ctx.dim, keepdims = True)
        return s * (grad_output - dot)
    
"""
    Shape Manipulation
"""

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, perm: List[int]):
        inv_perm = []
        for _ in range(len(perm)):
            inv_perm.append(None)
        for i in range(len(perm)):
            inv_perm[perm[i]] = i
        ctx.perm = inv_perm
        return input.permute(perm)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.permute(ctx.perm)
    
class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim0: int, dim1: int):
        ctx.dim0 = dim1
        ctx.dim1 = dim0
        return input.transpose(dim0, dim1)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.transpose(ctx.dim0, ctx.dim1)

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.ori_shape = list(input.shape)
        return input.reshape(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.reshape(ctx.ori_shape)
    
class View(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.ori_shape = list(input.shape)
        return input.view(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.reshape(ctx.ori_shape)
    
class Narrow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, start: int, length: int):
        ctx.input_shape = list(input.shape)
        ctx.dim = dim
        ctx.start = start
        ctx.length = length
        return input.narrow(dim, start, length)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        grad_input = zeros(ctx.input_shape)
        target = grad_input.narrow(ctx.dim, ctx.start, ctx.length)
        target.copy_(grad_output)
        return grad_input
    
class Chunk(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, chunks: int, dim: int = 0):
        ctx.dim = dim
        return input.chunk(chunks, dim)

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        return Tensor.cat(list(grad_outputs), ctx.dim)
    
class Split(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, split: Union[int, List[int]], dim: int = 0):
        ctx.dim = dim
        return input.split(split, dim)

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        return Tensor.cat(list(grad_outputs), ctx.dim)
    
class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        ctx.dim = dim
        return Tensor.stack(list(inputs), dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        tensors =  grad_output.split(1, ctx.dim)
        return tuple(tensor.squeeze(ctx.dim) for tensor in tensors)
    
class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        ctx.dim = dim
        ctx.split = [tensor.shape[dim] for tensor in inputs]
        return Tensor.cat(list(inputs), dim)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return tuple(grad_output.split(ctx.split, ctx.dim))
        
class Squeeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        ctx.dim = dim
        return input.squeeze(dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):    
        return grad_output.unsqueeze(ctx.dim)
    
class Unsqueeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        ctx.dim = dim
        return input.unsqueeze(dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.squeeze(ctx.dim)
    
class BroadcastTo(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.ori_shape = list(input.shape)
        return input.broadcast_to(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return reduce_broadcast(grad_output, ctx.ori_shape, grad_output.shape)
    
class Broadcast(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor):
        ctx.ori_shapes = [list(tensor.shape) for tensor in inputs]
        return Tensor.broadcast(*inputs)
        
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        output = list()
        for i in range(len(ctx.ori_shapes)):
            output.append(reduce_broadcast(grad_outputs[i], ctx.ori_shapes[i], grad_outputs[i].shape))
        return tuple(output)

class Mean(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        input_shape = list(input.shape)
        normalized_dim = dim + len(input_shape) if dim < 0 else dim
        ctx.input_shape = input_shape
        ctx.reduced_shape = list(input_shape)
        ctx.reduced_shape[normalized_dim] = 1
        ctx.N = input_shape[normalized_dim]
        return input.mean(normalized_dim, keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        grad_output = grad_output.reshape(ctx.reduced_shape)
        return (grad_output / ctx.N).broadcast_to(ctx.input_shape)

class Var(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False, unbiased: bool = True):
        input_shape = list(input.shape)
        normalized_dim = dim + len(input_shape) if dim < 0 else dim
        N = input_shape[normalized_dim]
        mean = input.mean(normalized_dim, True)
        denominator = N - 1 if unbiased else N
        ctx.input_shape = input_shape
        ctx.reduced_shape = list(input_shape)
        ctx.reduced_shape[normalized_dim] = 1
        ctx.denominator = denominator
        ctx.save_for_backward(input, mean)
        return input.var(normalized_dim, keepdims, unbiased)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, mean = ctx.get_saved_tensors()
        grad_output = grad_output.reshape(ctx.reduced_shape)
        grad_output = grad_output.broadcast_to(ctx.input_shape)
        return grad_output * (input - mean) * (2.0 / ctx.denominator)


class Unfold(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel_height: int, kernel_width: int):
        ctx.shape = input.shape
        ctx.kernel_height = kernel_height
        ctx.kernel_width = kernel_width
        return input.unfold(kernel_height, kernel_width)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.fold(ctx.shape, ctx.kernel_height, ctx.kernel_width)

class Fold(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, output_shape, kernel_height: int, kernel_width: int):
        ctx.kernel_height = kernel_height
        ctx.kernel_width = kernel_width
        return input.fold(output_shape, kernel_height, kernel_width)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.unfold(ctx.kernel_height, ctx.kernel_width)