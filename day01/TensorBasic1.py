#!/bin/usr/env python3
# -*- coding utf-8 -*-

# 将新版本的特性导入当前版本中,且必须放在第一行
from __future__ import print_function
# 导入torch库
import torch as t
import numpy as np

# 查看版本号
print('torch.__version__=', t.__version__)  # 1.13.1

# Tensor基本介绍
"""
Tensor，又名张量，读者可能对这个名词似曾相识，因它不仅在PyTorch中出现过，它也是Theano、TensorFlow、
Torch和MxNet中重要的数据结构。关于张量的本质不乏深度的剖析，但从工程角度来讲，可简单地认为它就是一个数组，且支持高效的科学计算。
它可以是一个数（标量）、一维数组（向量）、二维数组（矩阵）和更高维的数组（高阶数据）。
Tensor和Numpy的ndarrays类似，但PyTorch的tensor支持GPU加速。
本节将系统讲解tensor的使用，力求面面俱到，但不会涉及每个函数,对于更多函数及其用法，可以去官方文档查阅。
"""
# 1. 基础操作
"""
学习过Numpy的读者会对本节内容感到非常熟悉，因tensor的接口有意设计成与Numpy类似，以方便用户使用。
但不熟悉Numpy也没关系，本节内容并不要求先掌握Numpy。
从接口的角度来讲，对tensor的操作可分为两类：
1. `torch.function`，如`torch.save`等。
2. 另一类是`tensor.function`，如`tensor.view`等。
为方便使用，对tensor的大部分操作同时支持这两类接口，在后续不做具体区分，
如`torch.sum (torch.sum(tensor_a, a))`与`tensor.sum (tensor_a.sum(a))`功能等价。
而从存储的角度来讲，对tensor的操作又可分为两类：
1. 不会修改自身的数据，如 `tensor_a.add(a)`， 加法的结果会返回一个新的tensor。
2. 会修改自身的数据，如 `tensor_a.add_(a)`， 加法的结果仍存储在a中，a被修改了。
函数名以`_`结尾的都是inplace方式, 即会修改调用者自己的数据，在实际应用中需加以区分。
"""

# 创建Tensor
"""
创建Tensor的方法有很多，常用的如下：
函数	                功能
Tensor(*sizes)	                       基础构造函数
tensor(data,)	                       类似np.array的构造函数
ones(*sizes)	                       全1Tensor
zeros(*sizes)	                       全0Tensor
eye(*sizes)	                           对角线为1，其他为0
arange(s,e,step)	                   从s到e，步长为step
linspace(s,e,steps)	                   从s到e，均匀切分成steps份
rand/randn(*sizes)	                   均匀/标准分布
normal(mean,std)/uniform(from,to)	   正态分布/均匀分布
randperm(m)	                           随机排列

这些创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu).

其中使用`Tensor`函数新建tensor是最复杂多变的方式，
它既可以接收一个list，并根据list的数据新建tensor，也能根据指定的形状新建tensor，
还能传入其他的tensor，下面举几个例子。
"""

# 指定tensor的形状
a = t.Tensor(2, 3)
print('tensor_a=', a)
# 可以用list构建
a = t.tensor([[1, 2], [3, 4]], dtype=t.float64)
print('tensor_a=', a)
# 也可以用ndarray构建
b = t.tensor(np.array([[1, 2], [3, 4]]), dtype=t.uint8)
print('a=', b)
# 把tensor转为list
b.tolist()
print('b_tolist=', b.tolist())
# tensor.size()`返回`torch.Size`对象，它是tuple的子类，但其使用方式与tuple略有区别
b_size = b.size()
print('b_size=', b_size)

# b中元素总个数，2*3，等价于b.nelement()
print('a.num()=', b.numel())

# 创建一个和b形状一样的tensor
c = t.Tensor(b_size)
# 创建一个元素为2和3的tensor
d = t.Tensor((2, 3))
print('c=', c)
print('d=', d)

# 除了tensor.size()，还可以利用tensor.shape直接查看tensor的形状，tensor.shape等价于tensor.size()
print('c.shape=', c.shape)

# 其他常用的创建tensor的方法
print('t.ones(2, 3)=', t.ones(2, 3))
print('t.zeros(2, 3)=', t.zeros(2, 3))
print('t.arange(1, 6, 2)=', t.arange(1, 6, 2))
print('t.linspace(1, 10, 3)=', t.linspace(1, 10, 3))
print('指定CPU的t.randn(2, 3)=', t.randn(2, 3, device=t.device('cpu')))
print('t.randperm(5)=', t.randperm(5))  # 长度为5的随机排列
print(' t.eye(2, 3)=', t.eye(2, 3, dtype=t.int))  # 对角线为1, 不要求行列数一致

# torch.tensor使用的方法和参数几乎和np.array完全一致
scalar = t.tensor(3.14159)
print('scalar: %s, shape of sclar: %s' % (scalar, scalar.shape))
vector = t.tensor([1, 2])
print('vector: %s, shape of vector: %s' % (vector, vector.shape))
tensor1 = t.Tensor(1, 2)  # 注意和t.tensor([1, 2])的区别
print('tensor1', tensor1)
print('tensor.shape=', tensor1.shape)

matrix = t.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
print('matrix=', matrix)
print('matrix.shape=', matrix.shape)

tensor2 = t.tensor([[0.11111, 0.222222, 0.3333333]],
                   dtype=t.float64,
                   device=t.device('cpu'))
print('tensor2=', tensor2)

empty_tensor = t.tensor([])
print('empty_tensor.shape=', empty_tensor.shape)

# 默认常用类型
# torch.Tensor默认为torch.float32
print('torch.Tensor默认为:{}'.format(t.Tensor(1).dtype))
# torch.tensor默认为torch.int64
print('torch.tensor默认为:{}'.format(t.tensor(1).dtype))

