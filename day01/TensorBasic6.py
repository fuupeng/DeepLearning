#!/bin/usr/env python3
# -*- coding utf-8 -*-
from __future__ import print_function
import torch as t
import numpy as np


# 9. Tensor 和numpy
"""
Tensor和Numpy数组之间具有很高的相似性，彼此之间的互操作也非常简单高效。
需要注意的是，Numpy和Tensor共享内存。
由于Numpy历史悠久，支持丰富的操作，所以当遇到Tensor不支持的操作时，可先转成Numpy数组，处理后再转回tensor，其转换开销很小。
"""


a = np.ones([2, 3], dtype=np.float32)
print('tensor_a=', a)

# 从numpy转化
b = t.from_numpy(a)
print('a=', b)
# 可以直接将numpy对象传入Tensor
b = t.Tensor(a)
print('a=', b)

# 修改元素
a[0, 1] = 100
print('a=', b)

# tensor_a, a, c三个对象共享内存
c = b.numpy()
print('c=', c)

# 当numpy的数据类型和Tensor的类型不一样的时候，数据会被复制，不会共享内存
a1 = np.ones([2, 3])
# a1 的类型 float64
print('a1 的类型', a1.dtype)
b = t.Tensor(a1)
# a 的类型 torch.float32
print('a 的类型', b.dtype)
# 注意c的类型（dtype=torch.float64）
c = t.from_numpy(a1)
print('c=', c)
a1[0, 1] = 100
#  b与a不共享内存，所以即使a改变了，b也不变
print('a=', b)

#  不论输入的类型是什么，t.tensor都会进行数据拷贝，不会共享内存
b1 = t.tensor(a1)
print('b1=', b1)

# 广播法则
"""
广播法则(broadcast)是科学运算中经常使用的一个技巧，它在快速执行向量化的同时不会占用额外的内存/显存。 Numpy的广播法则定义如下：

1：让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分通过在前面加1补齐；
2：两个数组要么在某一个维度的长度一致，要么其中一个为1，否则不能计算；
3：当输入数组的某个维度的长度为1时，计算时沿此维度复制扩充成一样的形状。

PyTorch当前已经支持了自动广播法则，学会通过以下两个函数的组合手动实现广播法则：

unsqueeze或者view，或者tensor[None],：为数据某一维的形状补1，实现法则1
expand或者expand_as，重复数组，实现法则3；该操作不会复制数组，所以不会占用额外的空间。
注意，repeat实现与expand相类似的功能，但是repeat会把相同数据复制多份，因此会占用额外的空间
"""

d1 = t.ones(3, 2)
print('d1=\n', d1)
d2 = t.ones(2, 3, 1)
print('d2=\n', d2)
print('d1+d2=\n', d1+d2)
# 自动广播法则
"""
第一步：d1是2维,d2是3维，所以先在较小的d1前面补1 ，
              即：a1.unsqueeze(0)，a1的形状变成（1，3，2），a2的形状是（2，3，1）,
第二步:   d1和d2在第一维和第三维形状不一样，其中一个为1 ，
              可以利用广播法则扩展，两个形状都变成了（2，3，2）
"""

# 手动广播法则
# 或者 a.view(1,3,2).expand(2,3,2)+a.expand(2,3,2)
d3 = d1[None].expand(2, 3, 2) + d2.expand(2,3,2)
print('d1+d2=\n', d3)

# expand不会占用额外空间，只会在需要的时候才扩充，可极大节省内存
d4 = d1.unsqueeze(0).expand(10000000000000, 3,2)
print('d4=\n', d4)