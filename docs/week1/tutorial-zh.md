# Clownpiece-Torch 第一周

欢迎来到 Clownpiece-Torch 的第一周！本周的目标是构建一个基础的 C++ 张量（Tensor）库。这个库将作为我们类 PyTorch 机器学习框架的基石。到本周结束时，你将对张量在内存中的表示和操作方式有扎实的理解，这对实现高效的深度学习计算至关重要。

## 什么是张量（Tensor）？

从本质上讲，**张量**是将标量、向量和矩阵推广到更高维度的数学对象。

* 一个**标量**（例如 `5.0` 这样的单个数值）可以看作 0 维张量。
* 一个**向量**（例如 `[1.0, 2.0, 3.0]`）是 1 维张量。
* 一个**矩阵**（例如 `[[1, 2], [3, 4]]`）是 2 维张量。

张量可以拥有任意数量的维度（也称为**轴**）。例如，一个 3 维张量可以表示一幅彩色图像（高 × 宽 × 通道），而一个 4 维张量可以表示一批彩色图像（批大小 × 高 × 宽 × 通道）。

在深度学习中，张量是用于存储和转换数据的主要数据结构。它们可以表示：

* **输入数据**：图像、文本序列、音频信号等。
* **模型参数**：神经网络层的权重和偏置。
* **梯度**：反向传播过程中计算出的、用于更新模型参数的值。
* **中间激活值**：神经网络中各层的输出。

高效处理这些多维数组，是任何深度学习框架性能的关键所在。在高级加速器（GPU、NPU 等）中，张量上的计算在硬件层面被高度优化以最大化并行度。

## 存储与元数据的解耦

现代张量库的一个关键设计原则是将**存储**与**元数据**分离。

* **存储**：指张量数值数据实际存放的内存块。通常是一个连续的一维数组（就像你用 `new float[n]` 得到的那样）。无论张量的"形状"如何，所有元素都存放在这个扁平的内存块中。存储上附加了一个**数据类型**信息。

* **元数据**：定义了如何将存储中的数据解释为多维数组。它包括：
    * **形状（Shape）**：一个整数列表，定义张量在每个维度上的大小。张量的总元素数量（**numel**）是其各维形状的乘积。
        - 例如，形状 $(2, 3, 4)$ 描述了一个 3 维张量，有 $2$ 个"平面"，$3$ 个"行"，$4$ 个"列"；numel 为 $2 \times 3 \times 4 = 24$ 个元素。
    * **步长（Strides）**：一个整数列表，表示在一维存储中，沿张量各维度移动一个单位所需的步数。
        - 对于形状为 $[D_0, D_1, \ldots, D_{n-1}]$ 的张量，步长 $[S_0, S_1, \ldots, S_{n-1}]$ 告诉我们：要在第 $k$ 维上从索引 $i$ 移动到 $i+1$，需要在底层一维存储中前进 $S_k$ 个元素。
        - 如果张量在所有维度上都没有转置，则有 $S_k = \prod_{i=k+1}^{n-1} D_i$。
    * **偏移量（Offset）**：一个整数，表示张量第一个元素在存储中的偏移。当当前张量是从一个更大的张量切片而来时，偏移量可能不为零。
        - 要访问下标为 $(i_0, \ldots, i_n)$ 的元素，其在存储中的索引（一维存储数组中的位置）计算如下：
      $$
        \text{element index} = \text{offset} + \sum_{j=0}^{n-1} i_j \cdot S_j
      $$

元数据与存储信息的结合即为张量的本质——"存储 + 元数据"完备地刻画了一个张量。

这种存储与元数据的解耦之所以强大，是因为它使诸如转置、重塑（在兼容时）或对张量进行切片等操作只需更改形状、步长和偏移量这些元数据即可完成，而无需在内存中实际移动或复制数据。这非常高效。

**示例：**

考虑一个形状为 $(2, 3)$ 的二维张量（矩阵），数据类型为 float32，按行优先（C 风格数组）存储。

$$
A =
\begin{pmatrix}
e_0 & e_1 & e_2 \\
e_3 & e_4 & e_5
\end{pmatrix}
$$

其存储为：
$$
\begin{cases}
  \text{data} = [e_0, e_1, e_2, e_3, e_4, e_5], \\
  \text{dtype} = \text{float32}
\end{cases}
$$

其元数据为：
$$
\begin{cases}
\text{shape} = (2, 3) \\
\text{strides} = (3, 1) \\
\text{offset} = 0
\end{cases}
$$

此处 $\text{strides[0] = 3}$，因为每行有 3 个元素。

---

> 下面我们介绍一些张量操作，以及它们如何体现存储与元数据解耦的意义。

## 转置 / 置换

**置换（Permutation）** 是比简单矩阵转置更通用的操作。如果一个张量 $A$ 有 $n$ 个维度，如 $(d_0, d_1, \ldots, d_{n-1})$，则由 $(p_0, p_1, \ldots, p_{n-1})$ 定义的置换张量为：

$$
  A'[(i_0, \ldots, i_{n-1})] = A[i_{p_0}, \ldots, i_{p_{n-1}}]
$$

即，它将 $A'$ 的第 $k$ 维映射到 $A$ 的第 $p_k$ 维。

在实现中，置换操作根据 $(p_0, p_1, \ldots, p_{n-1})$ 重新排列张量的 `shape` 和 `strides` 元数据。底层数据存储与原始张量共享，保持不变。

数学上可以证明，在重新排序步长且存储不变的情况下，计算出的存储偏移量正确地反映了置换操作。这种技巧巧妙地避免了数据复制和新内存分配的开销。

**示例：转置张量 A**

为将 $A$ 转置为 $A^T$，我们需要交换维度 0 和维度 1。这对应于置换 $(p_0, p_1) = (1, 0)$。

转置后的张量 $A^T$ 为：
$$
  A^T=\begin{pmatrix}
  e_0 & e_3\\
  e_1 & e_4\\
  e_2 & e_5\\
  \end{pmatrix}
$$

$A^T$ 的表示为：
$$
\begin{cases}
\text{shape} = (3, 2) \\
\text{strides} = (1, 3) \\
\text{offset} = 0
\end{cases}
$$

形状维度的交换按置换 $(1,0)$ 进行，这点很直观。关键的洞见在于：对步长应用同样的置换，即可重新解释存储中已有的数据来匹配新的维度顺序。

考虑访问 $A^T_{2, 0}$（即元素 $e_2$）：
$$
\begin{aligned}
\text{element index} &= \text{offset} + 2 \times \text{strides}[0] + 0 \times \text{strides}[1] \\
&= 0 + 2 \times 1 + 0 \times 3 \\
&= 2
\end{aligned}
$$
这正确地定位到了 $[e_0, e_1, e_2, e_3, e_4, e_5]$ 中的元素 $e_2$。

## 重塑 / 视图

重塑张量意味着改变其形状，同时保持总元素数量和元素顺序不变。重塑过程中可以改变维度的数量。

当我们谈论重塑的"元素顺序"时，是根据张量当前形状和步长下的逻辑"展平"版本来考虑的。关键是，这种逻辑展平顺序可能与底层存储中数据的简单线性顺序不对应——如果张量经过置换则更是如此。

**视图（View）与重塑（Reshape）**

* **严格视图（如 `view()`）**：此操作仅在重塑**无需任何数据复制**时成功。如果因为新的逻辑元素顺序无法通过新的步长映射到现有存储而需要复制，此操作将抛出错误。

* **灵活重塑（如 `reshape()`）**：此操作尝试在不复制的情况下返回视图。但当不可行时，它会生成一个反映重塑结果的新连续张量。

**何时需要复制？**

一般来说，当无法在不重新排列内存元素的情况下完成重塑时，就需要复制。然而，精确判断"重塑是否兼容"的逻辑难以表述，相关文档也较少。为简单起见，我们仅在张量是**连续的**（具体而言，行优先连续或 C 连续）时允许视图操作。

> 注意：实际上存在一些不连续但视图仍然可行的情况，如果你感兴趣，请参考 [PyTorch ATen 源码](https://github.com/pytorch/pytorch/blob/2225231a144f182e2e1bd26a619cc1bafad49e6d/aten/src/ATen/TensorUtils.cpp#L330)。

**连续性定义**

* 一个张量是连续的，当且仅当其元素存储在连续内存中，且元素的逻辑顺序与物理顺序一致。
* 等价地，一个张量是连续的，当且仅当：
$$
    \forall i \in [d], \text{stride}[i] = \prod_{k=i+1}^{d} \text{shape}[k]
$$
其中 $d$ 是张量的维数，空集的 $\prod$ 按约定值为 $1$。

显然，对连续张量进行视图操作总是可行的。要在连续张量上执行视图操作，只需按 $\text{stride}[i] = \prod_{k=i+1}^{d} \text{shape}[k]$ 重新计算步长即可，其中 shape 是目标新形状。

**示例：重塑张量 A**

回顾：
$$
A =
\begin{pmatrix}
e_0 & e_1 & e_2 \\
e_3 & e_4 & e_5
\end{pmatrix}
$$

此时 $A$ 是连续的。我们可以将 $A$ 视图为 $\text{shape}=(6)$：
$$
A_1 = (e_0, e_1, e_2, e_3, e_4, e_5)
$$
其中 $\text{stride}=(1)$。

也可以将 $A$ 视图为 $\text{shape}=(3, 2)$：
$$
A_2 = \begin{pmatrix}
e_0 & e_1 \\
e_2 & e_3 \\
e_4 & e_5
\end{pmatrix}
$$
其中 $\text{stride}=(2, 1)$。

然而，将 $A^T$ 视图为 $\text{shape}=(2, 3)$ 是不允许的，因为它不连续。

---

## 广播（Broadcast）

广播是一种有用的变换，允许张量操作中的操作数在某些维度上被扩展以匹配其形状。

例如，你可以将一批矩阵输入（即第一维为批维度的 3 维张量）与一个权重矩阵（即 2 维张量）进行矩阵乘法，而无需显式地沿批维度复制权重矩阵。标量 + 张量也是广播的一个应用：将标量广播到与张量匹配的大小，然后执行逐元素加法。

**正式定义**，两个张量的广播形状由以下规则确定：

1. 通过在维度较少的张量形状前面添加 $1$ 来对齐维度数。
2. 对每个维度（从最后一个开始）：
   - 如果两个大小相等，则保留该大小。
   - 如果其中一个大小为 $1$，则将其广播到与另一个大小匹配。
   - 如果大小不同且都不为 $1$，则两个张量不可广播，应抛出错误。

确定广播形状后，两个张量沿扩展的维度分别被复制以匹配形状。广播不涉及物理副本，而是巧妙地将被广播维度的步长设为 $0$。因此，广播可能导致不连续的张量。

**示例：**

* 设 $\text{shape}_A = (3, 2, 1)$，$\text{shape}_B = (2, 1)$，则 $A, B$ 可广播，结果形状为 $(3, 2, 1)$。
* 设 $\text{shape}_A = (1, 2)$，$\text{shape}_B = (3, 2)$，则 $A, B$ 可广播，结果形状为 $(3, 2)$。
* 设 $\text{shape}_A = (2, 2)$，$\text{shape}_B = (3, 2)$，则 $A, B$ 不可广播。

* 设 $A=(1, 2)$，$B=\begin{pmatrix}3 & 4 \\ 5 & 6\end{pmatrix}$。对于 $A+B$，我们先将 $A$ 广播为 $\begin{pmatrix}1 & 2 \\ 1 & 2\end{pmatrix}$，然后 $A+B=\begin{pmatrix}4 & 6 \\ 6 & 8\end{pmatrix}$。

---

## 📘 补充教程

* [**PyTorch 张量入门**](https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)：一个初学指南，深入讲解 `torch.Tensor` 类的使用。如果你不熟悉 PyTorch 中的张量，可以看一看。

* [**PyTorch 内部机制（ezyang 博客）**](https://blog.ezyang.com/2019/05/pytorch-internals)：一篇深入剖析 PyTorch 核心张量抽象的详细文章——涵盖元数据（layout、device、dtype、stride）以及这些概念如何支撑视图创建、切片和更广泛的 C++ 核函数与 autograd 层。

---

# 代码指南

---

## 环境准备

在开始之前，请确保你的环境满足以下要求：

### 1. Linux 环境

本项目需要 Linux 环境。

* **WSL（Windows Subsystem for Linux）** 可以接受。
* **重要提示**：不要将项目目录放在 Windows 文件系统中（如 `/mnt/c/...` 下）。WSL 中跨 Windows-Linux 边界的文件访问会经过虚拟化的网络层，速度较慢。应将项目放在 WSL 的原生 Linux 文件系统中（如 `~/` 下的某个位置）。

### 2. C++ 工具链：`g++`（支持 C++17）

确保安装了以下工具：

* 支持 **C++17** 的 `g++`

对于 Debian/Ubuntu：
```bash
sudo apt update
sudo apt install build-essential
```

### 3. Conda（Python 环境管理）

如果尚未安装，请[安装 Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)。然后创建并激活项目虚拟环境：

```bash
conda create -n ct python=3.10
conda activate ct
```

### 4. Python 依赖

在 `ct` 环境激活后，安装所需的 Python 包：

```bash
pip install pybind11
```

为避免 C++ 头文件找不到的问题（如 `#include <pybind11/pybind11.h> not found`），请将 conda 环境下的 pybind11 头文件路径添加到 VSCode C++ 扩展的配置中。

### 5. 安装 PyTorch 作为参考

PyTorch 本身不是本项目的构建或运行依赖，但评分器库（graderlib）使用了 PyTorch。**你必须安装 PyTorch 才能运行评分器。** 此外，你可以参考 PyTorch 的行为来更好地理解你需要实现什么。

---

## 代码结构概览

```bash
clownpiece
|--autograd     # 第二周内容，现在不需要
|--tensor       # 你要实现的 C++ 张量库
| |- meta.h
| |- meta.cc
| |- tensor.h
| |- tensor.cc
| |- tensor.py      # ⚠️ 实际位于 clownpiece/ 下
| |- tensor_pybind.cc
|- __init__.py
|- setup.py
|- tensor.py         # 第二周的 autograd 绑定
|...               # clownpiece 的其余部分
tests
|--week1           # 第一周的测试用例
| |- grade_part{1-8}.py
| |- grade_all.py
| |- grade_comprehensive.ipynb
| |- graderlib.py
```

* **`meta.h` / `meta.cc`**：处理基本定义和工具函数。
    * `meta.h` 定义了张量库中使用的通用类型，如 `dtype`（数据类型，例如 `float`）、`shape_t`（张量维度）、`stride_t`（张量步长）。它还声明了随机数生成函数。
    * `meta.cc` 提供了 `meta.h` 中声明的随机数生成函数的实现。
    必要时你可以修改这些文件，例如添加新的类型别名或工具函数。

* **`tensor.h`**：`Tensor` 类本身的头文件。它包含 `Tensor` 类的声明，包括成员变量（如存储指针、形状、步长、偏移量）以及转置、重塑、算术运算等操作的方法签名。你可以修改此文件，例如添加新方法或友元函数声明。

* **`tensor.cc`**：核心实现文件，你将在其中编写 `tensor.h` 中声明的 `Tensor` 类方法的逻辑。

* **`tensor.py`**：定义了一个 Python `Tensor` 类。这个 Python 类将作为你构建的 C++ `Tensor` 对象的包装器。一般来说你不需要大幅修改此文件，因为它主要将调用委托给 C++ 后端。

* **`tensor_pybind.cc`**：使用 `pybind11` 库将你的 C++ `Tensor` 类及其方法暴露给 Python。它创建了必要的绑定，使得 `tensor.py` 包装器能够调用你的 C++ 代码。你一般不需要修改此文件。

* **`setup.py`**：用于将你的 C++ 源文件（`tensor.cc`、`meta.cc`、`tensor_pybind.cc`）编译成共享库（Python 扩展模块，如 Linux 上的 `.so` 文件）。这允许 Python 导入和使用你的 C++ 张量库。

---

## 如何编译和测试你的代码？

运行以下命令重新编译你的代码：
```bash
cd clownpiece && pip install -e .
```

然后，进入目录 `tests/week1` 运行测试脚本，或直接运行：
```bash
python tests/week1/grade_part{i}.py
```

每个部分对应一个测试脚本 `grade_part{i}.py`。你也可以运行 `grade_all.py` 一次性测试所有部分。

还有一个综合测试脚本：`grade_comprehensive.ipynb`。你必须同时通过 `grade_all.py` 和 `grade_comprehensive.ipynb`。

要以调试模式查看完整回溯，使用 `DEBUG=1 python grade_part{i}.py`。

**重要提示：你必须首先实现 `data_at` 方法（按逻辑顺序获取第 i 个元素）才能通过任何测试，因为评分器库依赖此函数。**

> 提示：下面给出的编码顺序是建议，而非硬性要求。你可能会发现先跳到后面的部分实现工具函数对完成早期部分有帮助。请通读 `tensor.h` 以掌握我们张量库的概貌。

---

接下来，让我们进入编码部分。

我们的接口设计兼容 PyTorch（除非另有说明）。如果你对某个函数的行为不确定，请参考 PyTorch 完善的文档。

## 第一部分：构造函数、赋值、析构函数和 item()

```cpp
  // 零维空张量
  Tensor();

  // 包含一个标量数据的零维张量
  Tensor(dtype value);

  // 指定形状的张量，数据未初始化
  explicit Tensor(const shape_t& shape);

  // 指定形状的张量，数据初始化为给定值
  explicit Tensor(const shape_t& shape, dtype value);

  // 指定形状的张量，数据由生成器函数初始化
  explicit Tensor(const shape_t& shape, std::function<dtype()> generator);

  // 指定形状的张量，使用向量作为底层数据
  explicit Tensor(const shape_t& shape, const vec<dtype>& data);

  // 由元数据 + 存储构造的张量（视图构造器）
  explicit Tensor(const shape_t& shape, const stride_t& stride, int offset, Storage storage);

  // 浅拷贝张量
  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);

  // 仅对单元素张量有效
  Tensor& operator=(dtype value);

  // 析构函数
  ~Tensor();

  // 转换为 dtype 值（仅对单元素张量有效）
  dtype item() const;
```

这里的所有函数都非常直接，不需要进一步解释。注意构造函数需要正确初始化 `shape_`、`stride_`、`offset_` 和 `storage_` 字段。

## 第二部分：工具函数、clone、contiguous、copy_、scatter_

```cpp
  int numel() const;
  int dim() const;
  veci size() const;
  int size(int dim) const;
  bool is_contiguous() const;

  Tensor clone() const;
  Tensor contiguous() const;
  Tensor copy_(const Tensor& other) const;
  Tensor scatter_(int dim, const Tensor& index, const Tensor& src) const;
```

---

> **int numel() const**：返回张量中的元素总数。注意：必须处理维度为零（标量或完全空张量）的情况。

> **int dim() const**：返回张量的维度数。

> **veci size() const** / **int size(int dim) const**：如果不提供 `dim`，返回整个形状向量。否则返回指定维度的大小。`dim` 遵循 Python 风格的索引，可为负数。如果越界则抛出 `std::runtime_error`。

> **bool is_contiguous() const**：返回张量是否在内存中连续。提示：在构造或赋值时预计算 `is_contiguous_` 和 `numel_` 以减少运行时开销。

> **Tensor clone() const**：返回一个新的连续张量，底层数据已复制。在较新版本的 PyTorch 中 clone 不一定返回连续张量，但在我们的实现中，请确保返回连续张量。

> **Tensor contiguous() const**：返回包含与自身张量相同数据的连续张量。如果自身张量已经连续，则返回自身视图；否则返回克隆的张量。

> **Tensor copy_(const Tensor& other) const**：将 other 的数据复制到当前张量的底层存储中，并返回自身。提示：编写 `offset_at(int n)` 和 `data_at(int n)` 等工具函数会很有用，它们计算按扁平逻辑顺序第 i 个元素在存储中的偏移量。

> **Tensor scatter_(int dim, const Tensor& index, const Tensor& src) const**：沿给定维度 `dim` 按 `index` 分散 `src` 张量。例如：
```python
result[index[i][j][k]][j][k] = src[i][j][k]  # 若 dim == 0
result[i][index[i][j][k]][k] = src[i][j][k]  # 若 dim == 1
result[i][j][index[i][j][k]] = src[i][j][k]  # 若 dim == 2
```
`index` 和 `src` 必须与自身张量去掉 `dim` 后的形状匹配。

## 第三部分：下标操作符

```cpp
  Tensor operator[](const vec<slice_t>& slices) const;
  Tensor operator[](slice_t slice) const;
  Tensor operator[](const veci& indices) const;
  Tensor operator[](int index) const;
```

---

> **Tensor operator[](const vec<slice_t>& slices) const** / **Tensor operator[](slice_t slice) const**：返回自身张量的子张量视图，以给定的切片为参数。切片遵循 Python 风格，可包含负索引。
- 如果切片为空（如 begin ≥ end），这些维度应设为空切片（例如 `[0:0]`），但不要抛出异常。
- 如果切片越界，应将切片裁剪到范围内，而不是抛出异常。
- 无需支持 `step` 特性（Python 切片中的第三个参数），可假设 step=1。
- 如果切片数量少于张量维数，则从最前面的维度开始切片。切片后维度**不应**被挤压。

示例：
```python
>> A = Tensor([[0, 1],
               [2, 3]])
>> A[0:1]
Tensor([[0, 1]])
>> A[0:-1]
Tensor([[0, 1]])
>> A[0:-100]   # 越界，裁剪后为空
Tensor([[]])
```

> **Tensor operator[](const veci& indices) const** / **Tensor operator[](int index) const**：使用特定整数索引访问元素或子张量。索引遵循 Python 风格，可包含负索引。如果越界则抛出 `std::runtime_error`。如果索引数量少于张量维数，则从最前面的维度开始索引。索引后维度应被挤压。

示例：
```python
>> A = Tensor([[0, 1],
               [2, 3]])
>> A[1]
Tensor([2, 3])   # 不是 Tensor([[2, 3]])
```

> 注意：在 Python 绑定中，我们仅使用单个 `int index` 或 `slice_t slice` 版本。Python 的实际索引可以是混合了整数和切片的列表。在 C++ 中处理这些类型差异很繁琐，所以我们通过 `tensor_pybind.cc` 将复杂的 Python 索引统一转换为 C++ 层可处理的形式。

## 第四部分：逐元素一元和二元运算符

```cpp
  // 一元取反
  Tensor operator-() const;

  // 逐元素加法、减法、乘法、除法
  friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator/(const Tensor& lhs, const Tensor& rhs);

  // 比较
  friend Tensor operator==(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator!=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator<(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator<=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator>=(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator>(const Tensor& lhs, const Tensor& rhs);

  // 其他一元数学运算
  Tensor sign() const;
  Tensor abs() const;
  Tensor sin() const;
  Tensor cos() const;
  Tensor tanh() const;
  Tensor clamp(dtype min, dtype max) const;
  Tensor log() const;
  Tensor exp() const;
  Tensor pow(dtype exponent) const;
  Tensor sqrt() const;
```

> 一元数学运算也有对应的非成员函数版本。

---

> **逐元素运算要点**：
1. 返回的张量与自身张量形状相同，创建一个新的存储。
2. **广播**：二元操作支持广播。如果 `lhs` 和 `rhs` 形状不同但兼容，在操作前先广播到共同形状。如果形状不可广播则抛出 `std::runtime_error`。
3. 标量操作数（如 `tensor + 5`）通过构造函数隐式转换为零维张量，然后参与广播。
4. 比较操作符：如果比较成立返回 1，否则返回 0。

> **各一元运算定义**：
- $\operatorname{sign}(x) = \begin{cases} 1 & x>0 \\ 0 & x=0 \\ -1 & x < 0 \end{cases}$
- $\operatorname{abs}(x) = |x|$
- $\operatorname{clamp}(x, \text{lower}, \text{upper}) = \min\{\text{upper}, \max\{\text{lower}, x\}\}$
- $\exp(x) = e^x$
- $\operatorname{pow}(x, \text{exponent}) = x^{\text{exponent}}$

> 提示：实现通用的 `apply_unary_op` 和 `apply_binary_op` 函数以统一接口、减少编码工作量。

## 第五部分：矩阵乘法

```cpp
  friend Tensor matmul(const Tensor& lhs, const Tensor& rhs);
  friend Tensor operator^(const Tensor& lhs, const Tensor& rhs); // 等价于 matmul
```

矩阵乘法的规则稍复杂，请参考 [PyTorch 文档](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul)：

- 如果两个操作数都是 1 维，返回点积标量（0 维张量）。
- 如果左操作数是 1 维、右操作数至少 2 维，则在左操作数前添加一个 1 维度使其变为 2 维。添加的维度在返回时移除。
- 如果右操作数是 1 维、左操作数至少 2 维，则在右操作数后添加一个 1 维度使其变为 2 维，然后对右操作数进行转置。添加的维度在返回时移除。
- 如果两个操作数都至少 2 维，则最后两维视为矩阵维度，前面的维度进行广播。
  - 例如，$L=(i\times 1 \times n \times m)$，$R = (j \times m \times l)$，则返回结果为 $(i \times j \times n \times l)$。

> 提示：将中间张量视图化为 $(\{-1, N\})$ 的形状通常能使逻辑更清晰。同时，使用转置将中间轴重排到最后一维也便于利用更快的连续内存访问。

## 第六部分：归约和归一化操作

```cpp
  Tensor sum(int dim, bool keepdims=false) const;
  std::pair<Tensor, Tensor> max(int dim, bool keepdims=false) const; // 返回 (values, indices)
  Tensor softmax(int dim) const;
```

---

> **Tensor sum(int dim, bool keepdims=false) const**：沿维度 `dim` 求和。
- `dim`：要归约的维度，支持负索引。
- `keepdims`：若为 `true`，输出张量与自身张量维度相同，`dim` 维的大小为 1。若为 `false`（默认），`dim` 维被挤压。

> **std::pair<Tensor, Tensor> max(int dim, bool keepdims=false) const**：返回一对张量：沿维度 `dim` 的最大值及这些最大值的索引。索引主要用于 autograd 目的。

> **Tensor softmax(int dim) const**：沿维度 `dim` 返回 softmax 结果。
$$
  \operatorname{softmax}(A) = \dfrac{A.\text{exp}()}{A.\text{exp}().\text{sum}(\text{dim}, \text{keepdims}=true)}
$$

## 第七部分：形状操作

```cpp
  Tensor permute(veci p) const;
  Tensor transpose(int dim1, int dim2) const;
  Tensor reshape(const shape_t& purposed_shape, bool copy=false) const;
  Tensor view(const shape_t& purposed_shape) const;
  Tensor narrow(int dim, int start, int length, bool copy=false) const;
  vec<Tensor> chunk(int chunks, int dim) const;
  vec<Tensor> split(int dim, int split_size) const;
  vec<Tensor> split(int dim, veci split_sections) const;
  static Tensor stack(const vec<Tensor>& inputs, int dim=0);
  static Tensor cat(const vec<Tensor>& inputs, int dim=0);
  Tensor squeeze(int dim) const;
  Tensor unsqueeze(int dim) const;
  Tensor broadcast_to(const shape_t& shape) const;
  static std::pair<Tensor, Tensor> broadcast(const Tensor& lhs, const Tensor& rhs);
  static vec<Tensor> broadcast(const vec<Tensor>& tensors);
```

---

> **Tensor permute(veci p) const**：返回原始张量按 `p` 重排维度的视图。`p` 可能只包含部分维度，此时其余维度保持原顺序不变。支持负索引。

> **Tensor transpose(int dim1, int dim2) const**：返回 `dim1` 和 `dim2` 两维交换后的视图。两维可相等。

> **Tensor reshape(const shape_t& purposed_shape, bool copy=false) const**：
- 形状推断：`purposed_shape` 中最多支持一个 `-1`，将从 numel 和其他指定维度推导。
- 复制逻辑：如果视图可行且 `copy=false`，返回视图；否则返回新的连续张量。

> **Tensor view(const shape_t& purposed_shape) const**：仅在张量连续时返回视图，否则抛出 `std::runtime_error`。

> **Tensor narrow(int dim, int start, int length, bool copy=false) const**：返回沿 `dim` 维在 `[start:start+length]` 区间的切片。`copy=false` 则返回视图，否则返回连续副本。

> **vec<Tensor> chunk(int chunks, int dim) const**：尝试将张量沿 `dim` 维分割为 `chunks` 个块，每个块是输入张量的视图。如果沿 `dim` 维的大小不可被整除，除最后一块外所有块大小相同。

> **vec<Tensor> split(int dim, int split_size) const** / **vec<Tensor> split(int dim, veci split_sections) const**：按大小或指定分段分割张量。

> **static Tensor stack(const vec<Tensor>& inputs, int dim=0)**：沿新维度（插入在 `dim` 之前）堆叠输入。

> **static Tensor cat(const vec<Tensor>& inputs, int dim=0)**：沿已有维度 `dim` 拼接输入。

> **Tensor squeeze(int dim) const**：移除大小为 1 的维度 `dim`。如果 `dim` 处大小不为 1，则抛出错误。返回共享同一存储的视图。

> **Tensor unsqueeze(int dim) const**：在 `dim` 之前插入大小为 1 的新维度。`dim` 范围从 `0` 到 `dim()`（含）。返回共享同一存储的视图。

> **Tensor broadcast_to(const shape_t& shape) const**：将张量广播到目标形状。形状必须可广播。

> **broadcast 静态方法**：将一对或一组张量广播到共同形状。广播机制详见前文——被广播的维度步长设为 `0` 以避免物理复制。

## 第八部分：辅助构造函数

```cpp
Tensor to_singleton_tensor(dtype value, int dim);
Tensor ones(const shape_t& shape);
Tensor ones_like(const Tensor& ref);
Tensor zeros(const shape_t& shape);
Tensor zeros_like(const Tensor& ref);
Tensor randn(const shape_t& shape);
Tensor randn_like(const Tensor& ref);
Tensor empty(const shape_t& shape);
Tensor empty_like(const Tensor& ref);
Tensor arange(dtype start, dtype end, dtype step);
Tensor range(dtype start, dtype end, dtype step);
Tensor linspace(dtype start, dtype end, int num_steps);
```

这些函数都有对应的成员函数版本（如 `ones_like()`、`randn_like()` 等）和非成员函数版本。

- **to_singleton_tensor(dtype value, int dim)**：创建形状为 `[1]*dim`、包含单个 `value` 的张量。
- **ones / zeros / randn / empty**：创建指定形状的填充张量或未初始化张量。`randn` 从标准正态分布 $\mathcal{N}(0,1)$ 中采样。
- **arange(start, end, step)**：返回 1 维张量，大小为 $\lfloor (\text{end} - \text{start}) / \text{step} \rfloor$，起始值 `start`（含）、终止值 `end`（不含）。
- **range(start, end, step)**：类似 `arange` 但两端都包含，大小为 $\lfloor(\text{end} - \text{start})/\text{step}\rfloor + 1$。
- **linspace(start, end, num_steps)**：在 `start` 和 `end`（含）之间线性等分 `num_steps` 个点。

## 附加部分（第三周需要）：mean 和 var

```cpp
  Tensor mean(int dim, bool keepdims=false) const;
  Tensor var(int dim, bool keepdims=false, bool unbiased=true) const;
```

- **mean**：沿给定维度计算平均值。
- **var**：沿给定维度计算方差。`unbiased=True` 时除以 $n-1$，否则除以 $n$。测试包含在第六部分（初始为注释状态）。

---

## 可选挑战：并行化张量计算

### 为什么需要并行化？

现代 CPU 和 GPU 配备了多个核心，允许计算并发执行。然而，Python 中张量操作的简单实现通常是顺序执行的，使大多数计算资源未被充分利用。通过并行化操作，我们可以减少运行时间、提高可扩展性——尤其是对于矩阵乘法和大规模张量变换等计算密集型任务。

本挑战探讨了不同的并行化策略来加速张量操作。

### 并行化策略

鼓励你实现并评估**三种不同方法**来并行化张量操作：

1. **每个操作一个线程（Thread-per-Operation）**：这种方法实现简单，但可能因线程创建开销过大（特别是对小规模任务）导致性能不佳。

2. **隐式并行（Implicit Parallelism）**：使用 [OpenMP](https://www.openmp.org/) 或其他隐式并行库自动并行化张量数据上的循环。OpenMP 透明地处理线程创建、任务划分和同步。

3. **线程池（Thread Pool）**：创建固定大小的线程池并按块分发任务。这减少了线程创建开销，可以在多个操作中复用线程。

### 评估任务

对每种方法在以下两个任务上、不同输入规模下进行性能测试和报告：
1. 逐元素操作（如 $y = x * 2 + 3$）
2. 矩阵乘法（$C = A @ B$，其中 $A \in \mathbb{R}^{N \times M}, B \in \mathbb{R}^{M \times K}$）

### 报告要求
- 清楚描述每种方法的实现和优化方式、测试设置（CPU 规格、数据集大小、时间测量方法、线程数量等）。
- 提供基准测试结果，画出三种方法和两种任务下的运行时间 vs 输入规模的图表，与串行基线实现对比。
- 分析哪种方法总体最优及原因、并行化开销在何时候超过收益、对哪些操作并行化最有利。

> 注意：请使用编译时标志控制使用哪种方法，提交时请确保默认编译行为为串行版本，以便助教无需 OpenMP 或自定义线程池即可运行你的程序。

---

## 提交作业

首先，确保通过了 `grade_all.py` 和 `grade_comprehensive.ipynb`。

然后，在 `docs/week1` 下撰写详细**报告**，描述你遇到的挑战、解决方案和收获（请附上 `grade_all.py` 输出的评分摘要）。报告占总分的一部分。

最后将整个项目文件夹打包为 `lab-week1.zip`，提交到 Canvas。请确保助教可以运行 `grade_all.py` 和 `grade_comprehensive.ipynb`，并从你的提交中找到报告。
