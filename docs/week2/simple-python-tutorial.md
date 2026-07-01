# Python 进阶语法简明教程

本文档覆盖 ClownpieceTorch 第二周及之后需要用到的 Python 进阶语法。如果你已熟悉这些内容，可跳过；否则建议在开始编码前通读一遍。

---

## 1. 装饰器（Decorator）

装饰器是"函数的函数"——它接收一个函数，在它外面包裹一些逻辑，然后返回包裹后的新函数。

### 基本形式

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("调用前")
        result = func(*args, **kwargs)
        print("调用后")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("World")
# 输出:
# 调用前
# Hello, World!
# 调用后
```

`@my_decorator` 等价于 `say_hello = my_decorator(say_hello)`。

### 带参数的装饰器

有时装饰器本身需要参数。这时需要再包一层函数：

```python
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet():
    print("Hi!")

greet()  # 输出三次 "Hi!"
```

这里 `@repeat(3)` 等价于 `greet = repeat(3)(greet)`。`repeat(3)` 返回 `decorator`，然后 `decorator(greet)` 返回 `wrapper`。

### 项目中实际使用的带参装饰器

项目中 `tensor_op` 就是一个带参数的装饰器：

```python
def tensor_op(op_name, Function_name):
    def decorator(function):
        def wrapped_function(*args, **kwargs):
            if not is_grad_enabled_with_params(*args):
                op = getattr(TensorBase, op_name)
                raw_results = op(*args, **kwargs)
                # ...包装返回...
            # 动态导入类并调用
            module = importlib.import_module("clownpiece.autograd.function")
            FunctionClass = getattr(module, Function_name)
            return function(*args, **kwargs, FunctionClass=FunctionClass)
        return wrapped_function
    return decorator

# 使用
@tensor_op('__add__', 'Add')
def __add__(self, other, FunctionClass=None):
    return FunctionClass().apply(self, other)
```

---

## 2. `*args` 和 `**kwargs`

### `*args`：接收任意数量的位置参数

```python
def sum_all(*args):
    total = 0
    for x in args:
        total += x
    return total

sum_all(1, 2, 3)       # 6
sum_all(1, 2, 3, 4, 5) # 15
```

`*args` 也可以用于**解包**：

```python
def add(a, b):
    return a + b

nums = [3, 5]
add(*nums)  # 等价于 add(3, 5)，输出 8
```

### `**kwargs`：接收任意数量的关键字参数

```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=20)
# name: Alice
# age: 20
```

同样可以**解包**：

```python
options = {"lr": 0.01, "momentum": 0.9}
optimizer = SGD(params, **options)  # 等价于 SGD(params, lr=0.01, momentum=0.9)
```

### 两者组合：通用的函数包装

```python
def log_call(func):
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__}，参数: {args}, {kwargs}")
        return func(*args, **kwargs)
    return wrapper
```

这是装饰器中最常见的模式。`*args, **kwargs` 将原函数的**所有**参数原封不动地透传。

---

## 3. `zip` 与 `tuple`

### `zip`：并行迭代

```python
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]

for name, score in zip(names, scores):
    print(f"{name}: {score}")
# Alice: 95
# Bob: 87
# Charlie: 92

# 也可以直接构建字典
dict(zip(names, scores))  # {'Alice': 95, 'Bob': 87, 'Charlie': 92}
```

项目中的应用（backward 函数）：
```python
graph_roots = [
    GraphRoot(tensor, grad)
    for tensor, grad in zip(tensors, grads)
    if tensor.requires_grad
]
```

### `tuple`：不可变序列

```python
t = (1, 2, 3)
t[0]       # 1
len(t)     # 3
a, b, c = t  # 解包：a=1, b=2, c=3

# 单元素元组注意逗号
single = (1,)  # 不加逗号会被当作括号表达式

# 空元组
empty = ()
```

`wrap_tuple` 工具函数常用于统一处理"可能是单值也可能是元组"的返回值：

```python
def wrap_tuple(x):
    return (x,) if not isinstance(x, (list, tuple)) else tuple(x)

wrap_tuple(5)        # (5,)
wrap_tuple((1, 2))   # (1, 2)
wrap_tuple([1, 2])   # (1, 2)
```

---

## 4. 生成器（Generator）与 `yield` / `yield from`

### `yield`：惰性生成值

生成器函数不会一次性返回所有值，而是每调用一次 `next()` 产生一个值：

```python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i += 1

g = count_up_to(3)
next(g)  # 1
next(g)  # 2
next(g)  # 3
# next(g) 会抛出 StopIteration

# 常用于 for 循环
for x in count_up_to(5):
    print(x)
```

生成器的优势是**内存效率**：不需要一次性把整个列表放入内存。

### `yield from`：委托给子生成器

```python
def chain_generators():
    yield from [1, 2, 3]
    yield from [4, 5, 6]

list(chain_generators())  # [1, 2, 3, 4, 5, 6]
```

等价于：
```python
def chain_generators():
    for item in [1, 2, 3]:
        yield item
    for item in [4, 5, 6]:
        yield item
```

项目中的应用（`Module.parameters()`）：
```python
def parameters(self, recursive=True):
    # 先 yield 自身的参数
    for param in self._parameters.values():
        if param is not None:
            yield param
    # 再递归 yield 子模块的参数
    if recursive:
        for module in self._modules.values():
            yield from module.parameters(recursive=True)
```

---

## 5. 上下文管理器（Context Manager）与 `with` 语句

### 基本用法

上下文管理器用于管理资源的获取和释放。最常见的是打开文件：

```python
with open("file.txt", "r") as f:
    content = f.read()
# 退出 with 块时自动调用 f.close()，即使发生异常也会执行
```

### 使用 `@contextmanager` 定义

最简单的方式是用 `contextlib.contextmanager` 装饰一个包含 `yield` 的生成器函数：

```python
from contextlib import contextmanager

@contextmanager
def no_grad():
    """临时禁用梯度追踪"""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield  # 在此处执行 with 块内的代码
    finally:
        _grad_enabled = previous  # 无论如何都会恢复

# 使用
with no_grad():
    # 此处的张量操作不会追踪梯度
    result = model(input_data)
# 退出后梯度追踪恢复
```

执行流程：
1. `yield` 之前的代码在进入 `with` 块时执行（保存旧状态，设置新状态）
2. `with` 块内的代码在 `yield` 处执行
3. `yield` 之后（`finally` 中）的代码在退出 `with` 块时执行（恢复旧状态），即使发生异常也会执行

### 类形式的上下文管理器

也可以通过定义 `__enter__` 和 `__exit__` 方法来实现：

```python
class set_grad_enabled:
    def __init__(self, mode):
        self.mode = mode
        self.prev = is_grad_enabled()

    def __enter__(self):
        _get_local().grad_enabled = self.mode  # 设置新状态

    def __exit__(self, exc_type, exc_val, exc_tb):
        _get_local().grad_enabled = self.prev   # 恢复旧状态

# 使用
with set_grad_enabled(False):
    result = model(input_data)
```

> **注意**：`@contextmanager` 装饰器只能用于包含 `yield` 的生成器函数，不能用于类。类形式的上下文管理器直接定义 `__enter__`/`__exit__` 即可，无需装饰器。

---

## 6. 类型注解（Type Hints）与字符串引用

### 基本类型注解

```python
def add(a: int, b: int) -> int:
    return a + b

x: List[int] = [1, 2, 3]
name: Optional[str] = None  # Optional[X] 等价于 Union[X, None]
```

Python **不会**在运行时强制类型注解——它们仅用于文档和静态检查工具（如 Pylance/mypy）。

### 前向引用：用字符串解决"类尚未定义"的问题

当一个类的方法引用该类自身时，类体还在定义中，直接使用类名会报错：

```python
class Node:
    def get_children(self) -> List["Node"]:  # ✅ 用字符串 "Node"
        ...
```

项目中的应用：
```python
class Tensor(TensorBase):
    requires_grad: bool
    grad: Optional["Tensor"]        # Tensor 类尚未完成定义，必须用字符串
    grad_fn: Optional["Function"]   # Function 在另一个模块中定义
    output_nr: int
```

### 处理循环导入

当模块 A 导入模块 B，模块 B 又导入模块 A 时，会形成循环导入。在 ClownpieceTorch 中，`autograd.py` 和 `function.py` 相互引用：

```python
# autograd.py
from clownpiece.tensor import Tensor

def gradient_edge(tensor: Tensor):
    from clownpiece.autograd.function import AccumulateGrad  # 延迟导入
    ...
```

```python
# function.py
from clownpiece.autograd.autograd import Node, Edge

class Function(Node):
    @staticmethod
    def forward(ctx, *args):
        ...
```

通过在**方法内部**进行导入（而非模块顶部），可以打破循环导入。Python 的 import 语句可以在代码的任何位置执行，导入过的模块会被缓存，不会重复执行。

---

## 7. `isinstance` 与 `getattr`

### `isinstance`：类型检查

```python
isinstance(3.14, float)        # True
isinstance([1, 2], (list, tuple))  # True（可以传入元组检查多种类型）
isinstance(Tensor(...), Tensor) # True
```

项目中的应用：
```python
# 判断输出是否为张量，如果是则设置 grad_fn
if isinstance(output, Tensor):
    output.grad_fn = self

# 判断是否为可迭代类型以展平参数
if isinstance(arg, (list, tuple)):
    flatten_args.extend(arg)
```

### `getattr`：动态获取属性

```python
class MyClass:
    def foo(self):
        return "bar"

obj = MyClass()
method = getattr(obj, "foo")  # 等价于 obj.foo
method()  # "bar"

# 带默认值
getattr(obj, "nonexistent", "default")  # "default"
```

项目中的应用（`tensor_op` 装饰器）：
```python
op = getattr(TensorBase, op_name)  # 动态获取 TensorBase 的方法
raw_results = op(*args, **kwargs)  # 调用该方法
```

---

## 8. `importlib.import_module`：动态导入

当模块名在编写代码时不确定（例如作为字符串变量传入），使用 `importlib.import_module`：

```python
import importlib

# 等价于 import clownpiece.autograd.function
module = importlib.import_module("clownpiece.autograd.function")

# 等价于 from clownpiece.autograd.function import Add
Add = getattr(module, "Add")
```

项目中的应用（`tensor_op` 装饰器）：
```python
module = importlib.import_module("clownpiece.autograd.function")
FunctionClass = getattr(module, Function_name)  # 如 "Add" → Add 类
return function(*args, **kwargs, FunctionClass=FunctionClass)
```

---

## 9. `copy.copy` 浅拷贝

```python
import copy

a = [1, 2, [3, 4]]
b = copy.copy(a)
b[0] = 99        # 不影响 a
b[2][0] = 99     # 影响 a！因为 [3,4] 是共享引用

# 对于自定义对象，浅拷贝复制对象本身，但不复制其属性引用的对象
```

项目中的应用（`Context.repack_tensor`）：
```python
@staticmethod
def repack_tensor(tensor: Tensor):
    if isinstance(tensor, Tensor):
        return copy.copy(tensor)  # 浅拷贝以避免循环引用
    return tensor
```

---

## 10. `enumerate`：带索引的迭代

```python
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

# 指定起始索引
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}: {fruit}")
# 1: apple
# 2: banana
# 3: cherry
```

项目中的应用（`Function.apply` 中设置 output_nr）：
```python
outputs = wrap_tuple(outputs)
for i, output in enumerate(outputs):
    if isinstance(output, Tensor):
        output.output_nr = i  # 第 i 个输出
```

---

## 快捷键查表

| 语法 | 用途 | 项目中常见位置 |
|------|------|---------------|
| `@decorator` | 包装函数 | `@tensor_op('clone', 'Clone')` |
| `*args, **kwargs` | 透传参数 | `Function.apply()`, 装饰器内部 |
| `yield` / `yield from` | 惰性生成 | `Module.parameters()` |
| `with ctx:` | 资源管理 | `with no_grad():` |
| `isinstance(x, T)` | 类型判断 | 区分 Tensor 和其他参数 |
| `getattr(obj, name)` | 动态属性 | `tensor_op` 中动态调用 TensorBase 方法 |
| `importlib.import_module` | 动态导入 | `tensor_op` 中动态加载 Function 类 |
| `"ClassName"` 注解 | 前向引用 | `Optional["Tensor"]` |
| `copy.copy(obj)` | 浅拷贝 | `Context.repack_tensor` |
| `enumerate(seq)` | 带索引迭代 | 设置 `output_nr` |
| `zip(a, b)` | 并行迭代 | backward 中构建 GraphRoot |
| `tuple(x)` / `wrap_tuple` | 统一为元组 | 处理可能有多个返回值的 Function |
