# Week 2 自动微分实验报告

## 实验内容

本实验在 Week 1 Tensor 计算库的基础上实现一个反向模式自动微分引擎。主要工作包括：

- 使用 `Node`、`Edge`、`Function` 和 `Context` 表示计算图及其反向计算规则；
- 为张量操作实现forward和backward；
- 使用多线程工作队列执行反向图中的就绪节点；
- 将计算结果和梯度与 PyTorch 对照验证。


## 遇到的挑战与解决方案

### 1. 理解叶子梯度累加节点

最初把 `AccumulateGrad` 当作普通 Function，认为它还需要根据 forward 和 context 计算输入梯度。实际上它是计算图末端的虚拟节点：传给它的 `output_grad` 已经是对应叶子的梯度贡献，它只需要把该贡献写入或累加到 `.grad`。

解决方法是明确区分“根据链式法则计算局部梯度”的普通 Function 和“汇总多条反向路径”的叶子累加节点，并让 `Edge.gradient_edge()` 根据 Tensor 是否具有 `grad_fn` 选择正确节点。

### 2. 多输出 Function 的梯度编号

`max`、`chunk`、`split` 和 `broadcast` 会产生多个输出。如果所有输出只共享 `grad_fn` 而没有编号，反向时无法判断梯度属于哪个输出。

解决方法是在 `Function.apply()` 中按输出顺序设置 `output_nr`，并让 `Edge.input_nr` 和节点输入缓冲区使用该编号。多输出 Function 的 backward 使用 `*grad_outputs` 接收各输出梯度；多输入 Function 则返回与原输入顺序一致的 tuple。

### 3. 广播梯度不能直接返回

二元操作前向可能把 `[2, 1]` 和 `[3]` 广播为 `[2, 3]`。此时局部梯度虽然数值公式正确，但形状仍是输出形状，不能直接累加到原输入。

解决方法是比较输入和输出 shape：先消除输出新增的前导维度，再沿输入大小为 1 且发生扩展的维度求和并保留该维，最终恢复原输入形状。矩阵乘法复用了该逻辑，但排除了最后两个矩阵维度。

### 4. `keepdims` 与维度恢复

归约操作在 `keepdims=False` 时会删除维度。若直接把 `grad_output` 广播到输入形状，右对齐规则可能把梯度放到错误维度，例如对 `[2, 3]` 沿 `dim=1` 求均值后得到 `[2]`，它不能直接广播回 `[2, 3]`。

解决方法是保存输入 shape 和归约维度，backward 先将梯度 reshape 为 `[2, 1]`，再广播为 `[2, 3]`。同一方法用于 `sum`、`mean`、`var` 和 `max` 的梯度恢复。

### 5. Softmax 内积与矩阵乘法的区别

Softmax 公式中的 $\langle g,s\rangle$ 表示沿指定维度对 `g * s` 求和，并不是 `g.matmul(s)`。把它实现为矩阵乘法时，两个 `[2, 3]` Tensor 会直接发生 shape 不匹配。

解决方法是计算 `(grad_output * s).sum(dim, keepdims=True)`，保留归约维以便结果广播回 softmax 输出形状。

### 6. 视图操作和原地写入

`view.backward` 收到的梯度不一定连续，因此不能继续使用要求连续内存的 `view`。`narrow.backward` 也不能把较小的输出梯度直接复制到完整输入形状的 Tensor。

解决方法是让 `view.backward` 使用允许必要复制的 `reshape`；对 `narrow`，先取得全零输入梯度的对应窄视图，再通过 `copy_` 写入该区域。由于窄视图共享 Storage，写入会反映到完整梯度 Tensor。

### 7. Python 参数展开与返回值语义

实现多输入和多输出操作时，需要区分列表、tuple 和参数展开。`*inputs` 在函数定义中收集参数，在调用处则展开参数。曾经将 `Tensor.cat(*inputs)` 写成把每个 Tensor 直接作为独立参数，但该接口要求第一个参数是 Tensor 列表；另一个问题是调用 `tensor.squeeze()` 后没有接收返回值，导致 singleton 维度没有真正删除。

解决方法是根据接口明确使用 `Tensor.cat(list(inputs), dim)`，并使用 tuple 推导式收集 `squeeze` 的返回结果。反向返回多个输入梯度时统一保持与输入顺序一致。

### 8. 多线程反向执行与梯度模式隔离

反向图中只有依赖归零的节点才能执行，同时多个前驱可能并发写入同一节点的梯度缓冲区。若梯度模式使用普通全局变量，一个 worker 进入 `no_grad` 还可能错误影响其他线程。

解决方法是用锁保护依赖计数和输入梯度缓冲区，用线程安全队列调度就绪任务，并将梯度模式存储在线程局部变量中。主线程等待队列全部完成后再回收 worker，同时传播反向任务中的异常。

## 正确性验证

在仓库根目录执行：

```bash
conda run -n clowntorch python tests/week2/grade_all.py
```

测试结果为：

```text
score: 510/510
# passed: 51
# failed: 0
all tests passed!
```



## 总结与收获

Week 2 的主要收获是理解了反向模式自动微分并不是对前向表达式进行符号求导，而是在前向过程中记录由 Tensor、Function 和 Edge 组成的计算图，再从输出梯度出发按链式法则反向传播局部梯度。`AccumulateGrad`、`output_nr` 和输入梯度缓冲区分别解决了叶子累加、多输出编号和多路径汇合问题。

实现具体 backward 时，除了导数公式，还必须同时考虑 shape、广播、视图和多输入输出顺序。数值公式正确并不代表实现正确：广播梯度需要归约，归约梯度需要恢复维度，形状操作需要应用逆映射，矩阵乘法还要区分向量、矩阵和批量维度。

在工程方面，我进一步熟悉了 Python 的装饰器、context manager、`*args`、tuple 解包、动态属性和线程同步方式，也认识到 Python 自动微分层与 C++ Tensor 后端之间的接口约定必须一致。通过与 PyTorch 的输出和梯度逐项对照，可以把计算图问题、数学公式问题和底层 Tensor 问题分开定位。
