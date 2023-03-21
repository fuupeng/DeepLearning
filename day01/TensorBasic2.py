#!/bin/usr/env python3
# -*- coding utf-8 -*-

# 将新版本的特性导入当前版本中,且必须放在第一行
from __future__ import print_function
# 导入torch库
import torch as t
import numpy as np

# 常用Tensor操作
"""
通过`tensor.view`方法可以调整tensor的形状，但必须保证调整前后元素总数一致。
`view`不会修改自身的数据，返回的新tensor与源tensor共享内存，也即更改其中的一个，另外一个也会跟着改变。
在实际应用中可能经常需要添加或减少某一维度，这时候`squeeze`和`unsqueeze`两个函数就派上用场了。
"""
# 1. 调整形状

a = t.arange(0, 6, dtype=t.float)
a_view = a.view(2, 3)
print('a_view=', a_view)

# 当某一维为-1的时候，会自动计算它的大小
b = a.view(-1, 3)
b_size = b.shape
print('b_size=', b_size)

# 增加维度
# 在第1维（下标从0开始）上增加“１”
b_unsqueeze1 = b.unsqueeze(1)
b_unsqueeze_size1 = b_unsqueeze1.shape
print('b_unsqueeze_size1=', b_unsqueeze_size1)
print('b_unsqueeze1=', b_unsqueeze1)

# -2表示倒数第2个维度
b_unsqueeze2 = b.unsqueeze(-2)
b_unsqueeze_size2 = b_unsqueeze2.shape
print('b_unsqueeze_size2=', b_unsqueeze_size2)
print('b_unsqueeze2=', b_unsqueeze2)

c = b.view(1, 1, 1, 2, 3)
# 压缩第0维的“１”
c_unsqueeze1 = c.squeeze(0)
print('c_squeeze_size1=', c_unsqueeze1.shape)
print('c_unsqueeze1=', c_unsqueeze1)

# 降低维度
# 压缩第0维的“１”
d = c.view(1, 1, 1, 2, 3)
d_squeeze1 = d.squeeze(0)
print('d_squeeze_size1=', d_squeeze1.shape)
print('d_squeeze1=', d_squeeze1)

# 把所有维度元素个数为“1”的压缩
d_squeeze2 = d.squeeze()
print('d_squeeze_size2=', d_squeeze2.shape)
print('d_squeeze2=', d_squeeze2)

# 修改a[1]
a[1] = 100
# a修改，d作为view之后的，也会跟着修改
d1 = a.view(1, 1, 1, 2, 3)
print('d1=', d1)

"""
`resize`是另一种可用来调整`size`的方法，但与`view`不同，它可以修改tensor的大小。
如果新大小超过了原大小，会自动分配新的内存空间，而如果新大小小于原大小，则之前的数据依旧会被保存。
"""
d1.resize_(3, 3)
print('d1=', d1)
d1.resize_(1, 3)
print('d1=', d1)
d1.resize_(2, 3)
print('d1=', d1)


