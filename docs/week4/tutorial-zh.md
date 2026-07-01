# Clownpiece-Torch 第四周：收官之作

欢迎来到 Clownpiece-Torch 系列的最后一周！本周我们将完成深度学习库的周边工具：

- **数据加载工具**：`Dataset`、`CSVDataset`、`ImageDataset`、`Dataloader` 以及图像变换
- **优化算法**：`Optimizer`、`SGD`、`Adam`
- **学习率调度器**：`LRScheduler`、`LambdaLR`、`ExponentialLR`、`StepLR`

按照传统，最后一周的工作量约为前几周的一半。

---

# 第一部分：数据集与数据加载器

## 1.1 Dataset 抽象

`Dataset` 基类位于 `clownpiece/utils/data/dataset.py`，提供两个关键接口：

- `dataset[index]`：按索引获取数据项。索引可以是整数（map-style）或其他形式，返回值可以是张量或元组。
- `len(dataset)`：获取数据项总数。

```python
class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
```

> 还有另一种较少见的数据集类型叫"可迭代数据集（iterable-dataset）"，用于随机访问缓慢甚至不可行的场景（如从远程服务器流式接收数据）。

## 1.2 CSVDataset

从 CSV 文件逐行读取，对每行应用可选的 transform：

```python
class CSVDataset(Dataset):
    def __init__(self, file_path: str, transform: Callable = None):
        # 加载 CSV，应用 transform
        pass
```

示例：
```python
dataset = CSVDataset('data/values.csv',
                     transform=lambda row: Tensor([float(x) for x in row]))
```

## 1.3 ImageDataset

从按子目录组织的图像文件夹中加载数据，每个子目录对应一个类别标签：

```python
class ImageDataset(Dataset):
    def __init__(self, file_path: str, transform: Callable = None):
        # 1. 读取子目录作为类别
        # 2. 为每个子目录分配 label_id
        # 3. 读取子目录中的文件
        # 4. 将 PIL Image 转为 np.ndarray
        # 5. 应用 transform
        # 6. 存储 (image, label_id) 对
        pass
```

提供的图像变换函数（返回 `np.ndarray -> np.ndarray` 或 `np.ndarray -> Tensor` 的变换）：
- `sequential_transform(*trans)`：组合多个变换
- `resize_transform(size)`：缩放图像
- `normalize_transform(mean, std)`：归一化
- `to_tensor_transform()`：转换 np.ndarray 为 Tensor

示例：
```python
transform = sequential_transform(
    resize_transform((28, 28)),
    normalize_transform(0.5, 0.2),
    to_tensor_transform()
)
dataset = ImageDataset('data/animal_dataset', transform=transform)
```

## 1.4 Dataloader

`Dataloader` 包装 `Dataset` 以提供批量迭代、可选打乱和样本整理：

```python
class Dataloader:
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 drop_last=False, sampler=None, collate_fn=None):
        pass

    def __iter__(self):
        # yield 一个批次的数据
        pass

    def __len__(self):
        # 返回批次数（而非数据集中的样本数）
        pass
```

**参数说明**：
- `shuffle` 和 `sampler` 互斥
- `drop_last`：如果数据集大小不能被 batch_size 整除，是否丢弃最后不足一批的余数
- `sampler`：提供 `__iter__`（逐个 yield 索引）和 `__len__` 的可迭代对象；为 None 时提供默认采样器（全量或打乱全量）
- `collate_fn`：定义如何将一批样本整理为批次数据。默认为：标量转 Tensor 后在第 0 维 stack，Tensor 直接 stack，元组则对每个元素分别应用上述规则

---

# 第二部分：优化器

所有优化器继承自 `clownpiece/utils/optim/optimizer.py` 中的 `Optimizer` 基类：

```python
class Optimizer:
    param_groups: List[Dict]     # 参数组列表，每组包含 'params' 和超参数
    state: Dict[Parameter, Dict] # 优化器内部状态（如动量项）
    defaults: Dict               # 添加新组时的默认超参数

    def step(self):
        raise NotImplementedError

    def zero_grad(self, set_to_None: bool = True):
        # 重置所有参数的梯度
        pass
```

## 2.2 SGD（随机梯度下降）

### 数学公式

| 变体 | 更新规则 |
|------|----------|
| 普通 SGD | $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ |
| 权重衰减 | $\theta_{t+1} = \theta_t - \eta(\nabla L(\theta_t) + \lambda \theta_t)$ |
| 动量 | $v_t = \mu v_{t-1} + (1-\gamma)\nabla L(\theta_t)$，$\theta_{t+1} = \theta_t - \eta v_t$ |
| 动量 + 权重衰减 | $v_t = \mu v_{t-1} + (1-\gamma)(\nabla L(\theta_t) + \lambda\theta_t)$，$\theta_{t+1} = \theta_t - \eta v_t$ |

> 注：标准 PyTorch SGD 的 dampening 默认为 0（即 $\gamma=0$），公式简化为 $v_t = \mu v_{t-1} + g_t$。

## 2.3 Adam（自适应矩估计）

Adam 结合了动量（一阶矩）和 RMSProp（二阶矩），为每个参数自适应地缩放学习率。

**一阶矩 $m_t$**（梯度的指数衰减平均）：
$$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**二阶矩 $v_t$**（梯度平方的指数衰减平均）：
$$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**偏差校正**：由于 $m_0 = v_0 = 0$，早期步骤会偏向零。校正：
$$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**最终更新**：
$$
  \theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

> 如果有权重衰减，$g_t = \nabla_\theta L(\theta_t) + \lambda \theta_t$。

---

# 第三部分：学习率调度器

调度学习率可以提升训练稳定性和收敛效果。`LRScheduler` 基类位于 `clownpiece/utils/optim/lr_scheduler.py`：

```python
class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        # 存储 base_lrs（初始学习率），用于闭式公式计算
        pass

    def get_lr(self) -> List[float]:
        # 子类重写，基于 last_epoch 提供闭式公式
        raise NotImplementedError

    def step(self, epoch=None):
        # epoch=None 则递增至下一轮
        # 计算新 LR 并赋值给 optimizer.param_groups[i]['lr']
        pass
```

### 具体调度器

- **LambdaLR**：$\text{lr} = \text{base\_lr} \times \text{lr\_lambda}(\text{epoch})$，可自定义任意形状。
- **ExponentialLR**：$\text{lr} = \text{base\_lr} \times \gamma^{\text{epoch}}$，每轮衰减固定比例。
- **StepLR**：类似 ExponentialLR，但每 `step_size` 轮才更新一次：$\text{lr} = \text{base\_lr} \times \gamma^{\lfloor \text{epoch} / \text{step\_size} \rfloor}$。

---

# 评分

首先确保通过 `tests/week4/grade_part{i}.py` 和 `grade_all.py`。

然后完成 `tests/week4/estate_value_predict` 和 `fashion-mnist`，用我们自己的工具重写第三周的训练任务。

---

# 最终报告

除第四周的技术报告外，还需准备一份**展示 PPT** 来总结你在整个项目中的工作和收获。请附上各周评分器的输出摘要。
