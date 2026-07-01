# Clownpiece-Torch 第二周

本周我们将专注于构建一个**自动微分引擎（Autograd Engine）**。

"Autograd" 是 automatic differentiation 的缩写。这个强大的组件是现代深度学习框架的核心，它自动化了梯度计算的复杂过程，使用户无需手动推导和实现反向传播。

PyTorch 中的一个简单示例：
```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

z = x * y + y**2  # z = 2*3 + 3^2 = 6 + 9 = 15

z.backward()

print(f"x.grad = {x.grad}")  # ∂z/∂x = y = 3.0
print(f"y.grad = {y.grad}")  # ∂z/∂y = x + 2*y = 2 + 6 = 8.0
```
输出：
```python
x.grad = tensor([3.])
y.grad = tensor([8.])
```

如果你觉得这像魔法一样难以理解，不用担心。我们将深入探究 autograd 引擎是如何实现的。

## 计算图

autograd 引擎的核心是**计算图（Computation Graph）**。这是一个有向无环图（DAG），表示所执行的操作序列。

* **节点（Nodes）**：在此图中，节点表示张量或操作。
    - 入边为零的节点是**输入张量（Input Tensors）**
    - 其余节点表示操作（抽象为 **Function**）及其结果（称为**中间张量，Intermediate Tensors**）
        - 中间张量的 `requires_grad=True`
        - 一些没有出边的中间张量即为**输出张量（Output Tensors）**
* **边（Edges）**：边捕获操作和张量之间的依赖关系，展示数据如何从输入张量、通过各种操作，最终产生输出张量。

例如，如果你计算 $c = a + b$，图中会有输入张量 $a$ 和 $b$ 的节点、加法 ($+$) 操作节点，以及结果张量 $c$ 的节点。边连接 $a$ 和 $b$ 到加法操作，以及加法操作到 $c$。

计算中使用的参数也被视为输入。例如，当你执行线性变换 $y = W@x + \text{bias}$ 时，$W$、$x$ 和 $\text{bias}$ 都被视为输入（$@$ 表示矩阵乘法）。

> 注意：计算图是动态构建和维护的，因此你可以在前向传播中使用条件分支。

## Function

Function 是可微分操作的抽象。它由两部分组成：

- $\text{forward}(\text{inputs}) \to \text{output}$：前向计算输出的操作（即 $f(x)$）。可能有多个输入和输出。
- $\text{backward}(\text{output\_grads}) \to \text{inputs\_grad}$：反向计算梯度的函数。它应用链式法则从输出梯度推导出输入梯度。

计算输入梯度所需的数据可保存在前向传播中，并在反向传播中取回。这些数据由每次函数调用关联的**上下文对象（Context）** 管理（同一函数的不同调用创建不同的上下文）。

例如，$x \mapsto x^2$ 的 Function：
- $\text{forward}(x) \to x^2$，并保存 $x$ 以备反向使用。
- $\text{backward}(grad) \to grad \cdot (2x)$，应用链式法则 $[f(x^2)]' = f'(x^2) \cdot (2x)$。

有些激活函数（如 $\text{ReLU}$）在某些点不可微（例如 $x = 0$），但用户仍可自定义反向函数——例如在不可微点分配梯度为 $0$。

类似地，某些变换如 $\text{Dropout}$ 涉及非确定性行为（例如随机将输出置为零），但在可微分框架中仍可处理：只要我们不要求对底层采样分布求梯度，就可以将随机性视为在前向传播中固定的（例如保存哪些元素被清零），仅对确定性路径计算梯度。

## 叶张量（Leaf Tensors）

**叶张量**是（反向）计算图中的特殊节点，反向传播在此停止。有两类：

1. 所有**不需要梯度**的张量。
2. 所有**需要梯度的输入张量**，因为输入不由任何操作创建。

再次以线性变换 $y = W@x + \text{bias}$ 为例：
- 在 $y$ 上调用 backward，$y$ 不是叶张量。
- $x$ 是不需要梯度的输入，属于第一类叶张量。
- $W$ 和 $\text{bias}$ 是需要梯度的参数，属于第二类叶张量。

> 注意：在 PyTorch 中还有一个与梯度累积相关的属性 `retains_grad`。我们将避免这种复杂性，仅使用 `requires_grad`。

## Autograd 引擎工作原理

### 前向传播

在前向传播期间，数据流经神经网络（或任何计算模型），操作被应用到输入张量以产生输出张量。每当一个操作被执行时，autograd 引擎会仔细记录这些操作的细节以构建计算图。

1. **操作记录**：当一个数学操作（如加法、乘法、卷积、激活函数）在一个或多个张量上执行，且至少有一个输入张量需要梯度（如模型参数）时，autograd 引擎介入——不仅计算结果，还将操作本身注册到计算图中。

2. **计算图创建**：对每个注册的操作，在计算图中创建一个"节点"。该节点封装了足够的信息以便在反向传播时计算梯度。

到前向传播结束时，一个完整的计算图已被构建，表示了将输入数据转换为最终输出的整个操作序列（以 DAG 形式）。

### 反向传播

反向传播是计算输出张量相对于计算图中所有需要梯度的张量的梯度的过程。

形式化地说，记输出张量为 $O_1, \ldots, O_n$，它们的梯度为 $O_1', \ldots, O_n'$，则对某个张量 $w$ 的梯度为：

$$
   w'= \frac{\partial O_1}{\partial w} \cdot O_1' + \frac{\partial O_2}{\partial w} \cdot O_2' + \cdots + \frac{\partial O_n}{\partial w} \cdot O_n'
$$

通常，输出张量是单个标量损失值，记作 $L$，其梯度为 $1$。则张量 $w$ 的梯度恰好是 $\frac{\partial L}{\partial w}$，可用于梯度下降。

1. **启动**：反向传播从所有输出张量开始，每个张量带有一个用户指定的梯度 $O_i'$。这些输出张量是（反向）计算图的根。

2. **图遍历（逆拓扑序）**：引擎按逆序遍历计算图，向叶张量方向移动。这种逆序遍历遵循拓扑顺序：对给定节点，仅当其所有输出的梯度都已计算并累积到缓冲区后，它才能被执行以获取其输入的梯度。

概念上，你可以认为 autograd 引擎从前向计算图存储的信息中构建一个表示反向传播的**反向计算图**，并执行该图。

---

考虑一个示例：
```python
A: Tensor 2x2
B: Tensor 2x2

C = A @ B
D = sin(B)
E = log(D)
F = C + D + E
```

对应的反向计算图中各 Function 的 backward 如下：
- $\text{sum}: (x, y, z) \mapsto x+y+z$ 的 backward 为 $s' \mapsto (s', s', s')$（无广播时）
- $\log: x \mapsto \ln x$ 的 backward 为 $y' \mapsto \dfrac{y'}{x}$
- $\sin: x \mapsto \sin x$ 的 backward 为 $y' \mapsto \cos(x) \cdot y'$
- $\text{matmul}: (A, B) \mapsto A@B$ 的 backward 为 $C' \mapsto (C'@B^T, A^T@C')$（可通过将矩阵乘法展开为加法和乘法来验证）

## 📘 补充教程

- [**PyTorch Autograd 机制（官方文档）**](https://pytorch.org/docs/stable/notes/autograd.html)
- [**Autograd 自动微分（PyTorch 官方教程）**](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [**理解 PyTorch Autograd（Medium）**](https://medium.com/geekculture/understanding-pytorchs-autograd-a-complete-guide-240a07d4a4c6)
- [**torchviz（可视化工具）**](https://github.com/szagoruyko/pytorchviz)

---

# 代码指南

## 代码结构概览

```bash
clownpiece
|--tensor
|--autograd
| |- autograd.py    # 核心 autograd 引擎
| |- __init__.py
| |- function.py    # 实现各种 Function
| |- no_grad.py     # 管理 no_grad 上下文
|- __init__.py
|- tensor.py        # TensorBase（无梯度追踪）和 Tensor（有梯度追踪）
|...
```

* **`tensor.py`**：包含你在第一周实现的 `TensorBase` 类。`Tensor` 类是 `TensorBase` 的子类。目前 `Tensor` 在功能上与 `TensorBase` 相同。本周你将扩展 `Tensor` 类以支持梯度追踪和自动微分。

* **`autograd.py`**：包含 autograd 引擎的关键组件，用于构建和执行计算图。

* **`no_grad.py`**：管理禁用梯度追踪的 `no_grad` 上下文。

* **`function.py`**：包含 `Function` 基类和各种具体 Function 的实现。

## 如何测试

类似第一周，进入 `tests/week2/`，运行 `grade_part{i}.py` 或 `grade_all.py`。调试模式：`DEBUG=1 python grade_part{i}.py`。

### 重要提示：实现方式并不唯一

实现 autograd 引擎有多种设计选择。例如关于反向执行，有三条路径：
- 在前向传播时直接构建反向计算图。
- 构建完整的前向计算图，然后在前向计算图上执行反向。
- 构建完整的前向计算图，再构建反向计算图，最后执行。

关于计算图的表示：
- 仅 Function 是节点，张量（函数的输入/输出）附加到 Function 上；边连接 Function，附带 `input_nr` 字段记录对应于哪个输出张量。
- Function 和张量都是节点；边连接 Function 和张量。

**在以下代码指南中，我们将遵循与 PyTorch 相同的设计**：
- 在前向传播时直接构建反向计算图。
- 仅 Function 是节点（这与我们在概念计算图中介绍的有所不同，但更容易实现）。

以下讨论中，所有"计算图"均指反向图，"输入"指 output_grads，"输出"指 input_grads（除非另有说明）。

---

## 第〇部分：Autograd 核心

> 预计代码量：~300 行

首先查看 `autograd/no_grad.py`，我们已为你编写了 `no_grad` 上下文管理器和 `is_grad_enabled` 方法。

请将 `no_grad` 和 `is_grad_enabled` 添加到 `autograd/__init__.py` 中以便从 autograd 模块导入。后续教程中我们不会显式指导你修改 `__init__.py`，请根据需要自行添加要导出的类/方法。

现在开始在 `Tensor` 类中添加新字段以支持梯度追踪：

```python
class Tensor(TensorBase):
  requires_grad: bool       # 是否需要梯度追踪
  grad: Optional["Tensor"]  # 该张量的梯度，可能为 None
  grad_fn: Optional["Function"]  # 产生该张量的 Function 实例
  output_nr: int            # 该张量是 grad_fn 前向输出的第几个结果（在反向图中成为边的 input_nr）

  def __init__(self, data: TensorBase, requires_grad: bool = None):
    super().__init__(data)
    self.grad_fn = None
    self.grad = None
    self.output_nr = 0
    self.requires_grad_(requires_grad)

  def requires_grad_(self, requires_grad: bool = None):
    if requires_grad is None:
      requires_grad = is_grad_enabled()
    self.requires_grad = requires_grad
```

> 提示：类型注解中使用 `"Tensor"` 而非 `Tensor`，因为类在自身内部定义时尚未完成。Python 支持带引号的类型注解。此技巧也可用于避免循环导入同时保留类型检查。

### Node 和 Edge

```python
class Node():
    node_id: int
    next_edges: List["Edge"]

    def __init__(self):
        self.node_id = None
        self.next_edges = []

    def run(self, *args, **kargs):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id

class Edge():
    input_nr: int               # 指向目标 Node 的第几个输入
    node: Optional[Node]        # 边指向的目标 Node

    def __init__(self, input_nr: int, node: Optional[Node]):
        self.input_nr = input_nr
        self.node = node

    @staticmethod
    def gradient_edge(tensor: Tensor) -> "Edge":
      # TODO: 你需要实现
      # 情况1：非叶张量 → 使用其 grad_fn 和 output_nr
      # 情况2：叶张量且需要梯度 → AccumulateGrad
      # 情况3：叶张量且不需要梯度 → node = None
      pass
```

### Function 类

```python
class Function(Node):
    ctx: Context

    def __init__(self):
        super().__init__()
        self.ctx = None

    @staticmethod
    def forward(ctx: Context, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, *args):
        raise NotImplementedError

    def apply(self, *args, **kwargs):
      # TODO: 你需要实现
      # step 1. 初始化 self.ctx 并填充 self.next_edges
      # step 2. 在 no_grad 下执行 self.forward(...)
      # step 3. 设置输出张量的 grad_fn 为 self，并将 requires_grad 设为 True
      # step 4. 返回输出
      pass

    def run(self, *args):
      # TODO: 你需要实现
      # step 1. 在 no_grad 下执行 self.backward(...)
      # step 2. 返回梯度
      pass
```

所有自定义 Function 都应继承 `Function` 并实现各自的 `forward` 和 `backward` 静态方法。

### 特殊节点

```python
class AccumulateGrad(Function):
    """叶张量的 grad_fn，将梯度累积到 .grad 字段"""
    def __init__(self, input: Tensor):
      # TODO: 存储输入张量引用
      pass

    @staticmethod
    def forward(ctx: Context):
        return None  # 永不调用

    def backward(self, ctx: Context, output_grad: Tensor):
      # TODO: 将 output_grad 累积到 input_tensor.grad
      pass

class GraphRoot(Node):
    """计算图的根节点，存储用户指定的初始梯度"""
    def __init__(self, tensor: Tensor, grad: Tensor):
      # TODO: 存储 grad，创建指向 tensor.grad_fn 的边
      pass

    def run(self, *args, **kargs):
      # TODO: 返回存储的梯度
      pass
```

> 注意：`AccumulateGrad.backward` 不是 `@staticmethod`，而是实例方法，通过 `self.input_tensor` 访问需要累积梯度的叶张量。`

### 图的执行

```python
class NodeTask():
    """包装一个 Node 及其所有输入梯度，是 GraphTask 中可执行的单元"""
    base: "GraphTask"
    node: Node
    inputs: List[Tensor]

    def run(self):
        # step1. 用输入运行节点
        # step2. 将输出梯度填充到 GraphTask 的输入缓冲区
        pass

class GraphTask():
    roots: List[Node]
    nodes: List[Node]
    dependencies: Dict[Node, int]       # 入度计数（拓扑排序用）
    inputs_buffer: Dict[Node, List[Tensor]]  # 累积中间梯度结果

    def __init__(self, roots):
        # 过滤掉 None 的根，若为空则抛出 "roots is empty"
        self._construct_graph()

    def _construct_graph(self):
        # BFS 分配 node_id（必须在加入 visited 之前，因为 hash 依赖 node_id）
        # 构建 dependencies 计数和 inputs_buffer
        pass

    def _run_single_thread(self):
        # 拓扑排序：队列初始为根节点，节点完成后递减下游依赖计数
        # 依赖计数为 0 时从 inputs_buffer 读取梯度并入队
        pass

    def _run_multi_thread(self):
        # 使用 queue.Queue 作为线程安全队列
        # ⚠️ 不能直接用 "while queue not empty" 作为退出条件
        # （因为临时空队列不等于所有工作完成）
        # 应使用完成计数 + 活跃计数来判断
        pass

    def fill_input(self, node, input_grad, input_nr):
        # 将 input_grad 累积到 inputs_buffer[node][input_nr]
        pass
```

> **为什么多线程时不能用 `while queue is not empty` 作为退出条件？**请在报告中解释。

> **为什么即使算子层面有多线程，反向执行的多线程仍然重要？**请在报告中解释。

### backward 函数

```python
def backward(tensors, grads=None):
    if grads is None:
        grads = [ones_like(tensor) for tensor in tensors]
    graph_roots = [
        GraphRoot(tensor, grad)
        for tensor, grad in zip(tensors, grads)
        if tensor.requires_grad
    ]
    gt = GraphTask(graph_roots)
    gt.run()
```

> 注意：我们默认提供 `ones_like` 作为梯度，而在 PyTorch 中仅对标量输出允许不传梯度参数。此行为与 PyTorch 不同。`

---

## 第一部分：Clone / Contiguous / Subscriptor

以 `Clone` 为例介绍如何将 `Function` 绑定到 `Tensor` 方法上。

当 `is_grad_enabled_with_params` 返回 True 时走 autograd 路径（调用 `Function.apply`），否则走基础路径（直接调用 `TensorBase` 方法）。

```python
def is_grad_enabled_with_params(*args):
    flatten_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flatten_args.extend(arg)
        else:
            flatten_args.append(arg)
    return is_grad_enabled() and any(
        tensor.requires_grad for tensor in flatten_args
        if isinstance(tensor, Tensor)
    )
```

为避免在每个方法中重复编写两个分支，我们提供了 `tensor_op` **装饰器**：
```python
@staticmethod
def tensor_op(op_name, Function_name):
    def decorator(function):
        def wrapped_function(*args, **kwargs):
            if not is_grad_enabled_with_params(*args):
                # 非梯度路径：直接调用 TensorBase 方法
                op = getattr(TensorBase, op_name)
                raw_results = op(*args, **kwargs)
                # 将结果包装为 Tensor（requires_grad=False）
                ...
            # 梯度路径：动态导入 Function 类并调用
            module = importlib.import_module("clownpiece.autograd.function")
            FunctionClass = getattr(module, Function_name)
            return function(*args, **kwargs, FunctionClass=FunctionClass)
        return wrapped_function
    return decorator
```

使用示例——`Tensor.clone()` 简化为：
```python
@tensor_op('clone', 'Clone')
def clone(self, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self)
```

> 注意：不要在 clone 或 contiguous 中复制 `grad_fn` 字段，这会破坏计算图的完整性。

请完成：

- `Tensor.clone(self)` & `class Clone(Function)` —— backward 直接返回 `grad_output`
- `Tensor.contiguous(self)` & `class Contiguous(Function)` —— 同上
- `Tensor.__getitem__(self, index_or_slice)` & `class Subscriptor(Function)` —— backward 创建 `zeros(input_shape)`，用 `copy_` 将 `grad_output` 复制到对应位置

---

## 第二部分：一元运算

逐元素一元运算的 backward 即为计算其导数。给定 $y = f(x)$ 和 $\frac{\partial L}{\partial y}$，求 $\frac{\partial L}{\partial x} = f'(x) \cdot \frac{\partial L}{\partial y}$。

为提升效率，常用保存的前向结果替代重新计算。例如 $\tanh$：$y = \tanh(x)$，导数为 $\frac{dy}{dx} = 1 - \tanh^2(x) = 1 - y^2$，使用前向输出 $y$ 而非重新计算 $\tanh(x)$。

请完成：

| 操作 | 导数 |
|------|------|
| `__neg__` | $-1$ |
| `sign` | $0$（处处为零） |
| `abs` | $\text{sign}(x)$ |
| `sin` | $\cos(x)$ |
| `cos` | $-\sin(x)$ |
| `tanh` | $1 - y^2$（利用前向输出 y） |
| `clamp` | $1$（在区间内），$0$（区间外） |
| `log` | $1/x$ |
| `exp` | $e^x = y$（利用前向输出 y） |
| `pow` | $\text{exponent} \cdot x^{\text{exponent}-1}$ |
| `sqrt` | $1/(2\sqrt{x}) = 1/(2y)$ |

---

## 第三部分：加减乘除

逐元素四则运算的 backward 很简单。关键挑战在于处理**广播**。

广播的 backward 是什么？
- 广播是沿某些维度复制数据。
- 复制的 backward 是将梯度**求和**。
- 所以广播的 backward 是沿被广播的维度求和（如有填充维度则挤压）。

你可以在 `function.py` 中实现两个装饰器来统一处理：
- `binary_op_forward_wrapper`：保存输入形状到 ctx
- `binary_op_backward_wrapper`：调用 backward 后再调用 `reduce_broadcast` 归约梯度

还需要支持将标量自动提升为单元素张量（可通过装饰器 `scalar_args_to_tensor` 实现）。

注意：除以零会产生 NaN，但不应抛出异常。任何涉及 NaN 的操作都不可微，通常 NaN 参与的任何操作都会产生另一个 NaN，无需特殊处理。

---

## 第四部分：矩阵乘法

对于常规 2D × 2D 矩阵 $C = A@B$：
$$
  \frac{\partial L}{\partial A} = C' @ B^T,\quad \frac{\partial L}{\partial B} = A^T @ C'
$$

另一个挑战是正确处理各种输入形状情况。回顾 matmul 的形状规则：
- 两个 1 维 → 点积标量
- 左 1 维、右 ≥ 2 维 → 左添加维度 1，返回时移除
- 右 1 维、左 ≥ 2 维 → 右添加维度 1 并转置，返回时移除
- 均 ≥ 2 维 → 最后两维为矩阵维，前面维度广播

你的 backward 实现必须正确处理 grad_output 的填充和跨广播维度的归约，以匹配原始操作数的形状。所有情况都会被测试。

---

## 第五部分：Sum / Max / Softmax

- **Sum**：backward 沿归约维度扩展（广播）梯度。
- **Max**：backward 创建 zeros(input_shape)，用 `scatter_` 沿归约维度按 argmax 索引将 grad_output 分散到 grad_input 中。
- **Softmax**：记 $\mathbf{s} = \text{softmax}(\mathbf{x})$，$\mathbf{g} = \frac{\partial L}{\partial \mathbf{s}}$，则：
$$
  \frac{\partial L}{\partial \mathbf{x}} = \mathbf{s} \cdot (\mathbf{g} - \langle\mathbf{g}, \mathbf{s}\rangle)
$$

---

## 第六部分：形状操作

理解形状操作导数的黄金法则：
> 前向是按某种映射移动数据，反向则是按**逆映射**移动梯度。

例如：
- `permute()` 通过重排轴移动数据，backward 将轴重排回来。
- `squeeze()` 从 $d$ 维索引空间映射到 $(d-1)$ 维，因为挤压的维度大小必为 1，所以映射是双射的、可微的。
- `narrow()` 将数据映射到特定轴和索引范围的子空间，范围内的数据双射映射到输出，范围外的数据接收零梯度。

> 小心处理列表格式的输入或输出！你可能需要使用 `*args` 和 `return *tensors` 以避免破坏 autograd 引擎的 `input_nr` 逻辑。

请完成：`Permute`、`Transpose`、`Reshape`、`View`、`Narrow`、`Chunk`、`Split`、`Stack`、`Cat`、`Squeeze`、`Unsqueeze`、`BroadcastTo`、`Broadcast`。

---

## 附加：Mean 和 Var

- **Mean**：backward 将 grad_output 除以 $N$ 后广播回原形状。
- **Var**：请仔细思考方差函数的导数。使用公式 $\text{Var}(x) = \frac{1}{n}\sum(x_i - \mu)^2$（有偏）或 $\frac{1}{n-1}\sum(x_i - \mu)^2$（无偏）进行求导。可借助前向保存的 $\mu$ 和 $x$ 来简化 backward 的计算。

---

## 提交作业

首先确保通过 `grade_all.py`。然后在 `docs/week2` 下撰写详细报告，描述你遇到的挑战、解决方案和收获。最后将整个项目文件夹打包为 `lab-week2.zip` 提交到 Canvas。
