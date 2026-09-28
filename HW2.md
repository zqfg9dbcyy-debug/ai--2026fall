# 《人工智能中的编程》第二次作业

## 实现 Tensor 与激活函数

## Part 1：Tensor

请按照课程第二讲 PPT 中的内容，实现用于在 CPU 和 GPU 之间管理数据的 `Tensor` 类。

在设计 `Tensor` 类时，请实现基本的构造函数和析构函数，以及 `.gpu()` 与 `.cpu()` 方法。可以参考如下函数接口：

```cpp
Tensor tensor(shape, device);
Tensor c = tensor.cpu();
Tensor g = tensor.gpu();
```

提示：

1. 请使用 C++ 标准库中提供的容器定义 `Tensor` 的形状。
2. 本次作业中仅要求基本的内存管理功能，PyTorch 风格的 `reshape` / `permute` / slicing / indexing 操作不在要求范围内。但作为整个框架的基石，请大家在设计 `Tensor` 时进一步思考如何为后续操作保留更多的灵活性，推荐参考 [torch.storage](https://docs.pytorch.org/docs/stable/storage.html) 的设计理念。
3. 可以使用智能指针更简洁地进行内存管理，请参考 [cppreference](https://www.cppreference.com/w/cpp/memory.html) 中的对应内容。
4. 不要求实现不同数据精度的 `Tensor`，统一使用 `float` 类型即可。学有余力的同学可以尝试实现对不同数据类型的支持。

---

## Part 2：Activation

在 `Tensor` 类的基础上，请实现 ReLU 和 Sigmoid 激活函数，包括正向计算和反向传播。

#### 2.1 ReLU

ReLU 的正向计算定义为：

$$
\operatorname{ReLU}(x) = \max(x, 0)
$$

其中，*x* 为输入值，*ReLU(x)* 为输出值。由此推导出 ReLU 的反向传播为：

$$
\frac{\partial L}{\partial x} =
\begin{cases}
\dfrac{\partial L}{\partial y}, & x > 0, \\
0, & x \leq 0.
\end{cases}
$$

其中，*L* 是损失函数，*x* 是 ReLU 函数的输入，*y* 是 ReLU 函数的输出。

> 备注：ReLU 在 *x = 0$*处不可微，这里选取与 PyTorch 相同的定义，即在 *x = 0* 处将梯度设为 0，方便大家调试。

#### 2.2 Sigmoid

Sigmoid 是另一种常用的激活函数，其定义如下：

$$
\operatorname{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
$$

其中，*x*为输入值，*σ(x)*为输出值。由此推导出 Sigmoid 的反向传播为：
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} y(1-y)
$$

其中，*L* 是损失函数，*x* 是 Sigmoid 函数的输入，*y* 是 Sigmoid 函数的输出。

激活函数的接口实现为接受 `Tensor` 输入或接受数据指针输入均可。

---

### 评分标准

本次作业满分 10 分。

| 项目 | 分值 |
| --- | ---: |
| 正确实现 `Tensor` 类 | 4 |
| 正确实现 ReLU 函数的正向与反向传播 | 2 |
| 正确实现 Sigmoid 函数的正向与反向传播 | 4 |

---

### 提交要求

请提交代码与一份简要的实验报告，介绍你是如何实现的，以及你是如何确认实现正确性的。

**作业提交截止时间：2026 年 10 月 8 日 23:59:59。**

每迟交一天扣 1 分，扣一半为止。
