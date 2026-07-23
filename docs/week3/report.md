# Week 3 模块系统与 Conv2D 实验报告


## 实验范围

本周在已有 Tensor 和 autograd 之上构建模块系统，使参数、缓冲区和子模块能够被统一注册、递归遍历、保存与恢复，并使用这些抽象组合常见神经网络层。实现过程中没有把重点放在重复编写 forward 公式，而是重点处理模块状态管理、广播与 shape、数值稳定性以及不同软件层之间的接口约定。除此之外，我完成了可选挑战：从 C++ Tensor 后端开始补充 `unfold` 和 `fold`，接入 autograd，最终实现支持非方形奇数 kernel 的 `Conv2D`。

## 主要挑战与解决方案

### 1. 自动注册与普通 Python 属性必须区分

`Module.__setattr__` 会根据赋值对象的类型，将 `Parameter`、`Buffer` 和子 `Module` 分别写入 `_parameters`、`_buffers` 和 `_modules`。这些对象并不一定作为普通实例属性保存，而是由 `__getattr__` 在对应字典中查找。因此，同名普通属性可能遮蔽已注册状态。

实现 Conv2D 时曾先执行 `self.bias = bias` 保存布尔开关，之后再将 Parameter 赋给 `self.bias`。由于普通布尔属性仍然存在，读取 `self.bias` 得到的会是布尔值，而不是 `_parameters["bias"]` 中的 Parameter。解决方法是避免创建同名普通属性：有偏置时直接注册 Parameter，无偏置时使用 `register_parameter("bias", None)`。如果需要单独保存开关，应使用 `use_bias` 等不同名称。

### 2. 递归状态遍历需要同时保留层次和稳定名称

模块可能形成多层树结构，单纯遍历当前 `_parameters` 会漏掉子模块中的权重；将所有参数直接放入一个全局列表又会失去名称和层次。最终使用生成器递归遍历子模块，并用点号连接路径，例如 `block.attention.q_proj.weight`。`state_dict()` 使用相同规则展开 Parameter 和 Buffer，使保存、加载和 `named_parameters()` 具有一致的键。

可选参数为 `None` 时仍保留键，可以让 `load_state_dict(strict=True)` 区分“模块明确禁用了该状态”和“状态字典缺少该键”。加载前先检查 missing key、unexpected key 和 shape，再执行原地 `copy_`，避免验证到一半时已经修改部分参数。

### 3. 高层公式正确不代表广播和 shape 一定正确

Linear、归一化、注意力和损失函数的数学表达式都比较直接，但实际实现还必须匹配 Tensor 库的 shape 语义。例如 Linear 的权重保存为 `[out_features, in_features]`，forward 时转置后才能作为矩阵乘法右操作数；LayerNorm 和 BatchNorm 需要先 reshape 到统一二维布局，归一化后再恢复原 shape；bias 的一维 shape 则依靠右对齐广播扩展到 batch 和空间维。

梯度也需要遵守同样的 shape 约定。前向广播的参数在 backward 中必须沿扩展维求和，归约操作则需要先补回 singleton 维再广播。解决问题时不能只检查 forward 数值，还要同时检查输入梯度和参数梯度的 shape。

### 4. 参数初始化必须是原地且不进入计算图

初始化函数带 `_` 后缀，语义是修改传入 Parameter，而不是返回一个替代对象。如果初始化过程被 autograd 记录，会给参数附加无意义的 `grad_fn`。因此所有初始化函数统一在 `no_grad` 下生成临时 Tensor，再用 `copy_` 写入已有参数。Xavier 和 Kaiming 初始化则先根据权重布局计算 fan-in/fan-out，再选择对应分布范围。

### 5. 数值稳定性和梯度链路需要一起验证

CrossEntropy 若直接计算 `softmax().log()`，较大的 logits 容易先在 `exp` 中溢出。实现中使用 log-sum-exp 形式：

\[
\log p = x-\log\sum_i e^{x_i}
\]

并在指数运算前减去最大值。验证时不仅比较 loss，还执行 backward，确认 logits 能收到有限梯度。类似地，注意力中的 softmax 也先减去行最大值，避免只在小输入上看似正常。

## 可选挑战：Conv2D

### 1. 实现范围

本实现支持：

- 输入布局 `NCHW`；
- `stride=1`；
- `padding="same"`，越界位置补零；
- `dilation=1`；
- `groups=1`；
- 高和宽可以不同，但均限制为正奇数；
- 可选 bias。

奇数限制使上下、左右 padding 可以分别写成 `kernel_height // 2` 和 `kernel_width // 2`，从而保持输出空间尺寸为 `H×W`。

### 2. Unfold 的布局和索引

输入 shape 为：

\[
X\in\mathbb{R}^{N\times C\times H\times W}
\]

`unfold(k_h,k_w)` 将每个输出位置对应的局部窗口展开为一行，输出：

\[
X_{\text{col}}\in
\mathbb{R}^{N\times(HW)\times(Ck_hk_w)}
\]

空间位置和窗口元素的线性编号分别为：

\[
l=yW+x
\]

\[
q=c(k_hk_w)+k_yk_w+k_x
\]

对应输入坐标：

\[
y_{\text{in}}=y+k_y-\left\lfloor\frac{k_h}{2}\right\rfloor,\qquad
x_{\text{in}}=x+k_x-\left\lfloor\frac{k_w}{2}\right\rfloor
\]

坐标越界时写入 0。C++ 实现直接根据 Tensor 的 stride 和 offset 读取 Storage，因此也能正确处理非连续输入视图。虽然 im2col 会产生额外内存，但它把卷积的核心计算转换成批量矩阵乘法，避免在 Python 层逐 kernel 循环。

### 3. Fold 不是普通意义上的逆操作

`fold` 接收 `[N, H*W, C*k_h*k_w]`，按照与 unfold 相反的索引映射写回 `[N,C,H,W]`。一个输入像素可能出现在多个重叠窗口中，因此 fold 必须对这些贡献求和。

所以通常有：

\[
\operatorname{fold}(\operatorname{unfold}(X))
=D\odot X
\]

其中 \(D\) 是每个像素被窗口覆盖的次数，而不是全 1 Tensor。这个行为不是错误，因为 autograd 需要的是 unfold 的伴随算子，而不是数值逆函数。

把 unfold 写成线性算子 \(U\)：

\[
Y=UX
\]

则其反向传播为：

\[
\frac{\partial L}{\partial X}
=U^T\frac{\partial L}{\partial Y}
=\operatorname{fold}\left(\frac{\partial L}{\partial Y}\right)
\]

反过来，fold 的 backward 是 unfold：

\[
\frac{\partial L}{\partial P}
=U\frac{\partial L}{\partial Z}
\]

实现 `Unfold.forward` 时必须保存原始输入 shape，因为 backward 收到的梯度 shape 已经是展开后的三维 shape，不能用它作为 fold 的输出 shape。

### 4. 跨层绑定与 autograd 接入

这项挑战需要保持四层接口一致：

1. `tensor.h/tensor.cc` 定义并实现 C++ `Tensor::unfold` 和 `Tensor::fold`；
2. `tensor_pybind.cc` 将 `output_shape`、kernel height 和 kernel width 绑定到 Python；
3. `TensorBase` 直接调用 `_impl`，`Tensor` 通过 `tensor_op` 选择是否创建 autograd Function；
4. `Unfold` 和 `Fold` Function 实现互为伴随的 backward。

其中容易出现的问题包括 pybind 绑定到错误成员函数、Tensor 方法遗漏装饰器注入的 `FunctionClass` 参数，以及 backward 使用展开后 shape。解决方法是分别测试无梯度的底层调用和开启梯度后的高层调用，再检查每一层输入输出 shape，而不是只测试最终 Conv2D。

### 5. Conv2D 的矩阵化实现

卷积权重为：

\[
W\in\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}\times k_h\times k_w}
\]

将其 reshape 为：

\[
W_{\text{flat}}\in
\mathbb{R}^{C_{\text{out}}\times(C_{\text{in}}k_hk_w)}
\]

卷积可写为：

\[
Y_{\text{col}}
=X_{\text{col}}W_{\text{flat}}^T
\]

其中：

\[
Y_{\text{col}}\in
\mathbb{R}^{N\times(HW)\times C_{\text{out}}}
\]

加入一维 bias 后，将结果转置并 reshape 为 `[N,C_out,H,W]`。由于 unfold、reshape、transpose、matmul 和 broadcast add 都已接入 autograd，输入、权重和 bias 的梯度可以沿同一计算图自动传播，不需要单独编写 Conv2D backward。

## 正确性验证

### 课程评分脚本

在仓库根目录执行：

```bash
conda run -n clowntorch python tests/week3/grade_all.py
```

结果为：

```text
Part 1: 80/80
Part 2: 90/90
Part 3: 80/80
Part 4: 160/160
Total: 410/410
4/4 parts passed
```

仓库提供的 Week 3 grader 不包含 Conv2D bonus，因此另外增加了独立测试。

### Conv2D bonus 自测

测试文件为 [`tests/week3/test_conv2d_bonus.py`](../../tests/week3/test_conv2d_bonus.py)，运行：

```bash
conda run -n clowntorch python tests/week3/test_conv2d_bonus.py
```

结果为：

```text
Conv2D bonus tests passed: 4/4
```

测试覆盖：

1. `unfold` 与 `fold` 的伴随恒等式
   \(\langle Ux,p\rangle=\langle x,U^Tp\rangle\)；
2. `3×5` 非方形 kernel、2 个输入通道、3 个输出通道的精确前向结果；
3. input、weight 和 bias 的反向梯度数值及 shape；
4. bias 参数注册与无 bias 状态；
5. 偶数 kernel 被明确拒绝。

在全 1 输入和全 1 权重下，`3×5` kernel 的一个输出通道为：

```text
12 16 20 20 16 12
18 24 30 30 24 18
18 24 30 30 24 18
12 16 20 20 16 12
```

内部位置完整覆盖 \(2\times3\times5=30\) 个输入元素，边缘数值随有效覆盖面积下降，符合 same zero-padding 的预期。将输出梯度设为全 1 后，input、weight、bias 均得到正确梯度，其中每个 bias 梯度为 \(4\times6=24\)。

## 局限与改进方向

当前 Conv2D 只覆盖课程要求的简化范围，不支持 stride、dilation、groups 和可配置 padding。偶数 kernel 的 same padding 需要上下或左右不对称填充，也尚未实现。im2col 会显著增加临时内存，较大的输入可以考虑分块展开，或在 C++ 后端实现更直接的卷积 kernel。当前 unfold/fold 使用串行嵌套循环，后续还可以利用 Week 1 的 `parallel_for` 按 batch 和输出空间位置并行。

## 总结与收获

Week 3 最重要的收获不是记住各层的 forward 公式，而是理解模块系统如何在不参与具体计算的情况下管理状态和层次结构。Parameter 注册、递归遍历、state dict 和训练模式构成了高层模型组织能力；真正的数值计算和梯度仍由 Tensor 与 autograd 完成。

Conv2D bonus 进一步体现了分层接口一致性的重要性。一个算子只有同时具备底层数值语义、Python 绑定、autograd 伴随操作和 Module 级 shape 组织，才算真正可用。通过伴随恒等式和精确梯度测试，可以把 C++ 索引问题、绑定问题、计算图问题和高层矩阵布局问题分别定位，而不是只根据最终输出 shape 判断实现是否正确。
