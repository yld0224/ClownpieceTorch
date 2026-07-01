# Clownpiece-Torch 第三周

第二周我们构建了强大的 autograd 引擎，能够追踪计算并自动计算梯度。虽然这是现代深度学习框架的核心，但仅用原始张量操作来编写复杂模型会显得繁琐且杂乱。本周我们将构建一个受 PyTorch `torch.nn.Module` 启发的**模块系统（Module System）**，为模型构建过程带来结构性、可复用性和便利性。

模块系统提供了一种将神经网络的各个部分封装为可复用组件的方式。它处理可学习参数、子模块和状态缓冲区的管理，使你能够以干净、面向对象的方式定义复杂架构。

PyTorch 示例：
```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
print(model)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

## 统一计算与状态

第一个设计理念是**紧密耦合部分的统一管理**。神经网络层不仅仅是一个函数，它是一个有状态的计算——具有确定的变换（计算）和跨调用持久化的内部变量（状态）。模块通过将**计算**和**状态**组织在一起来提供帮助。

模块将这三者优雅地组织为：
- **前向传播（Forward）**：定义模块对输入施加的变换，结合用户输入和模块内部状态产生输出。
- **参数（Parameters）**：可学习的状态，通常称为**权重**。训练模型就是优化这些参数以实现某个目标。
- **缓冲区（Buffers）**：不可学习的状态。例如 BatchNorm 中的运行均值和方差。它们随参数保存，但不被优化器更新。

> "不可学习"不等于"不可变"。这更多是一个模型结构性概念：是否能被优化，或仅用于临时存储。

### 示例：Linear 模块

```python
class Linear(Module):
    W: Tensor  # shape [out_features, in_features]
    b: Tensor  # shape [out_features]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W.transpose(-1, -2) + self.b
```

这里 $y = x @ W^T + b$。为什么存储转置后的 $W$？因为当作为 matmul 的右操作数时，$W^T$ 会被转置回 $W$，此时它是连续的！

## 模块化与层次化

### 模块化 → 简洁、复用、灵活

将复杂神经网络分解为更小的可管理模块，设计过程就简单得多。模块高度可复用——主流框架为 Linear、Conv2d、BatchNorm 等常见模块提供了高度优化、经过充分测试的实现；即使是领域特定模块（如 FlashAttention、RoPE）也可以在多种 Transformer 模型中被广泛采用。模块化还带来极大的灵活性——你可以轻松置换或引入新组件，无需重建整个网络。

### 层次化 → 设计和管理便利

模块系统天然是**层次化**的。高层模块由更小、更基本的模块组成，但反过来不行。从功能角度看，基础层和复杂块之间没有明显区别——它们在模块抽象下是统一的。

借助层次结构，我们可以将模块的组成概念化为树形结构，父模块可以管理所有子模块的状态。这种集中式状态管理对保存、更新或恢复整个模块状态非常有益。

### 示例：展开类 GPT 模型

```
GPTModel
├── Embedding
├── Positional Encoding
└── Transformer Blocks
    ├── Transformer Block 1
    │   ├── Multi-Head Attention
    │   │   ├── Linear (Q, K, V)
    │   │   └── Linear (output)
    │   ├── Layer Normalization
    │   └── Feed-Forward Network
    │       ├── Linear
    │       └── Activation
    └── Transformer Block 2
        └── ...
```

注意，树形层次结构并不意味着子模块按顺序递归执行。确切的计算逻辑由用户在 forward 函数中定义，可能形成复杂的 DAG。

## 分层系统设计

本周做代码时你可能会发现：autograd 引擎和张量库隐藏了底层计算和反向追踪的大部分复杂性，模块系统感觉像是 autograd 系统上的简单包装器。

这正是我们如此设计的原因：将系统功能分离到不同的层中，高层仅依赖低层（通常只依赖相邻层）。这为设计和实现都带来极大的简洁性。

模块系统完全不知道 autograd 引擎或张量库到底如何工作——它只假设它们会按约定完成各自的工作。反之亦然。

> 不过从设计者角度看，设计良好的接口让高层能高效便捷地利用低层是很重要的。这需要系统的全局视角。

---

## 补充教程

- [**`torch.nn.Module` 官方文档**](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [**使用 PyTorch 构建模型**](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
- [**保存和加载 PyTorch 模型**](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

**强烈建议在编写代码之前熟悉 PyTorch 的模块系统！**

---

# 代码指南

## 代码结构

```bash
clownpiece
|-nn
| |- activations.py   # 激活函数
| |- containers.py    # Sequential, ModuleList, ModuleDict
| |- init.py          # 参数初始化
| |- layers.py        # Linear, Embedding, LayerNorm, BatchNorm, MultiheadAttention
| |- loss.py          # MSELoss, CrossEntropyLoss
| |- module.py        # Module 核心抽象类
|-...
```

## 第一部分：核心模块系统

### 参数/缓冲区/子模块管理

首先定义状态存储：`Parameter` 和 `Buffer`，它们都是 Tensor 的平凡子类，带有首选的 `requires_grad` 值。

```python
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)

class Buffer(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=False)
```

模块的成员变量：
```python
class Module(object):
    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Buffer]]
    _modules: Dict[str, "Module"]
```

通过重写 `__setattr__(self, name, value)` 方法实现自动注册：当 `self.name = value` 发生时，检测 `value` 是否为 `Parameter`、`Buffer` 或 `Module` 实例，是则注册到对应字典中。

> 良好实践：添加机制确保 Module 的子类必须在 `__init__` 中调用 `super().__init__()`（用户经常忘记！）。例如添加 `_init_called` 布尔变量。

请完成 `parameters()`、`named_parameters()`、`buffers()`、`named_buffers()`、`modules()`、`named_modules()` 等遍历方法。使用 `yield` 和 `yield from` 语义。

### 状态字典（State Dict）

模块系统管理状态，因此必须提供保存和恢复状态的机制。

- **`state_dict()`**：返回扁平字典 `{name: Tensor}`（包含参数和缓冲区）。键名与 `named_parameters()` 和 `named_buffers()` 返回的名称一致（用 `.` 连接属性名）。状态字典中的张量是浅引用，无物理复制。
- **`load_state_dict(state_dict, strict=True)`**：从状态字典加载。`strict=True` 时检查：(1) 所有键都被使用，(2) 张量形状匹配，(3) 所有期望的键都存在。任一约束违反则抛出 `RuntimeError`。禁用可选参数的值为 `None` 应包含在状态字典中。

### 模块字符串表示：`__repr__` 和 `extra_repr`

格式要求：
- 包含类名
- 如有 `extra_repr`，在类名后显示：`ClassName(extra_repr)`
- 如有子模块，递归缩进列出：每个子模块为 `(name): ChildRepr`，每层缩进两空格

子类应重写 `extra_repr()` 而非 `__repr__()`。例如 Linear：`extra_repr` 返回 `"in_features=784, out_features=128, bias=True"`。

---

## 第二部分：最简单的具体模块

### Linear

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # 使用 Parameter 包装 W 和 b
        # 使用 init.py 中的方法初始化权重和偏置
        # 典型初始化：uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        pass

    def forward(self, x: Tensor) -> Tensor:
        # y = x @ W.T + b
        pass
```

### Tanh

Tanh 是将 `Tanh` Function 包装为模块的简单示例。

完成这些后运行 `grade_part1.py` 测试核心模块系统，`grade_part2.py` 测试 Linear 和 Tanh。

---

## 第三部分：初始化

完成 `init.py` 中的初始化函数（均需 `no_grad`，以 `_` 后缀表示原地修改）：

**简单初始化**：`constants_`、`zeros_`、`ones_`、`normal_`、`uniform_`

**高级初始化**：

- **Xavier（Glorot）**：适用于 sigmoid/tanh 等饱和激活函数。
  - uniform：$W \sim \mathcal{U}(-a, a)$，其中 $a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$
  - normal：$W \sim \mathcal{N}(0, \sigma^2)$，其中 $\sigma = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$

- **Kaiming（He）**：专为 ReLU 族设计，补偿 ReLU 将一半输出置零导致的方差减半。
  - uniform：$b = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}$
  - normal：$\sigma = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}$

其中 `fan_in = shape[-1]`，`fan_out = shape[-2]`。gain 查表获取（`tanh` 为 5/3，`relu` 为 $\sqrt{2}$ 等）。

测试对 200,000 个数据点进行均值（绝对偏差 < 0.1）和方差（相对误差 < 30-40%）的统计验证。

---

## 第四部分：具体模块

### 激活函数

- **Sigmoid**：$\sigma(x) = \frac{1}{1 + e^{-x}}$，输出范围 (0, 1)，常用于二分类输出层。
- **ReLU**：$f(x) = \max(0, x)$，最广泛使用的激活函数，计算高效，但存在"神经元死亡"问题。
- **Leaky ReLU**：$f(x) = \begin{cases} x & x > 0 \\ \alpha x & x \le 0 \end{cases}$，解决 Dying ReLU 问题。

> 这些操作应基于已有基础操作组合实现（如 `x * (x >= 0)` 实现 ReLU）。你也可以在 `function.py` 中添加对应的 Function 以便 autograd 追踪。

### 容器

- **Sequential**：按传入顺序依次执行子模块，`forward` 隐式定义为链式传递。
- **ModuleList / ModuleDict**：以列表/字典方式组织子模块，**没有默认 forward 方法**。解决了"将模块放入普通 list/dict 不会被追踪为子模块"的问题。

### 层

- **Embedding**：$\text{out}[i] = \text{weight}[i]$，权重形状 `(num_embd, embd_dim)`，可看作以 one-hot 向量为输入的线性层。
- **LayerNorm**：$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$，沿最后若干维归一化。
- **BatchNorm**：沿批维度归一化，训练时维护运行均值和方差的指数移动平均，评估时使用运行统计量。
- **MultiheadAttention**：分头注意力，$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$。

### 损失函数

- **MSELoss**：回归任务标准损失，$\mathcal{L} = \frac{1}{N}\sum(y_{\text{true}} - y_{\text{pred}})^2$。
- **CrossEntropyLoss**：分类任务标准损失，$\mathcal{L} = -\log(p_k)$（$p_k$ 为正确类别的预测概率）。数值稳定性提示：用 `logits - logits.exp().sum(dim=-1, keepdims=True).log()` 替代 `logits.softmax().log()`。

## 综合任务

完成 `tests/week3/estate_value_predict/main.ipynb`（房价预测回归）和 `tests/week3/mnist/main.ipynb`（手写数字分类）。

## 可选挑战：添加 Conv2D 支持

由于张量库和 autograd 引擎功能尚不完整，卷积模块暂未支持。你需要：
1. 在 C++ 后端添加 `unfold`（im2col）和 `fold`（col2im）基础操作
2. 在 autograd 中添加对应的 Function
3. 实现 `Conv2D` 模块（简化为 `stride=1, padding='same'`）

即使无法给出可工作的实现，清晰阐述原理也可获得部分分数。

## 提交作业

通过 `grade_all.py` 后，在 `docs/week3` 下撰写报告，打包为 `lab-week3.zip` 提交。
