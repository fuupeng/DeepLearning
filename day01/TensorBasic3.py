#!/bin/usr/env python3
# -*- coding utf-8 -*-
# 将新版本的特性导入当前版本中,且必须放在第一行
from __future__ import print_function
# 导入torch库
import torch as t

# 2. 索引操作
"""
Tensor支持与numpy.ndarray类似的索引操作，语法上也类似，下面通过一些例子，讲解常用的索引操作。
如无特殊说明，索引出来的结果与原tensor共享内存，也即修改一个，另一个会跟着修改。
"""
tensor_a = t.randn(3, 4)
print('tensor_a=', tensor_a)
# 第0行(下标从0开始)
print('tensor_a[0]=', tensor_a[0])
#  第0列
print('tensor_a[:,0]', tensor_a[:, 0])
#  第0行第2个元素，等价于a[0, 2]
print('tensor_a[0][2]=', tensor_a[0][2])
print('tensor_a[0][2]=', tensor_a[0, 2])
# 第0行最后一个元素
print('tensor_a[0][-1]=', tensor_a[0, -1])
print('tensor_a[0][-1]=', tensor_a[0][-1])
# 前两行
print('tensor_a[:2]=', tensor_a[:2])
print('第0, 1行', tensor_a[t.LongTensor([0, 1])])

# 前两行第0,1列
print('tensor_a[:2, 0:2]=', tensor_a[:2, 0:2])

# 第0行，前两列
# 注意两者的区别：形状不同
print('第0行，前两列=', tensor_a[0:1, :2])
print('第0行，前两列', tensor_a[0, :2])

# None类似于np.newaxis, 为a新增了一个轴
# 等价于a.view(1, tensor_a.shape[0], tensor_a.shape[1])
print('添加前轴后a.shape=', tensor_a.shape)
print('tensor_a=', tensor_a)
print('添加新轴后a.shape=', tensor_a[None].shape)
print('tensor_a=', tensor_a)
print('添加新轴后a.shape=', tensor_a[:, None, :].shape)
print('添加新轴后a.shape=', tensor_a[:, None, None, :].shape)

# 选择
"""
常用的选择函数

函数	                                   功能
index_select(input, dim, index)	      在指定维度dim上选取，比如选取某些行、某些列
masked_select(input, mask)	          例子如上，a[a>0]，使用ByteTensor进行选取
non_zero(input)	                      非0元素的下标
gather(input, dim, index)	          根据index，在dim维度上选取数据，输出的size与index一样

gather是一个比较复杂的操作，对一个2维tensor，输出的每个元素如下：
out[i][j] = input[index[i][j]][j]  # dim=0
out[i][j] = input[i][index[i][j]]  # dim=1
三维tensor的`gather`操作同理.
"""
print('返回一个ByteTensor:\n', tensor_a > 1)
# 选择结果与原tensor不共享内存空间
print('tensor_a[tensor_a>1]=', tensor_a[tensor_a > 1])
print('tensor_a[tensor_a>1]=', tensor_a.masked_select(mask=tensor_a > 1))
# 第0行和第1行
print(' tensor_a[t.LongTensor([0,1])]=', tensor_a[t.LongTensor([0, 1])])

# gather
b = t.arange(3, 12).view(3, 3)
print('a=', b)
# indices必须与b的维度相同
# 索引出5 7 9
# dim=1, 即列方向的维度和index的维度是匹配的
# out[i][j] = input[i][index[i][j]]  # dim=1 （二维）
indices1 = t.tensor([[2], [1], [0]])
out1 = t.gather(input=b, dim=1, index=indices1)
print('索引出5 7 9 ：\n', out1)

# out[i][j] = input[index[i][j]][j]  # dim=0 （二维）
# dim=0, 即行方向的维度和index的维度是匹配的
indices0 = t.tensor([[2, 1, 0]])
out2 = t.gather(input=b, dim=0, index=indices0)
print('索引出5 7 9 ：\n', out2)

c = t.arange(0, 16).view(4, 4)
print('c=', c)
# 选取对角线元素
indices = t.LongTensor([[0, 1, 2, 3]])
out = t.gather(input=c, dim=0, index=indices)
print('索引出0 5 10 15：\n', out)
# 按行选取反对角线元素
indices_ = t.LongTensor([[3, 2, 1, 0]])
out_ = t.gather(input=c, dim=0, index=indices_)
print('索引出3 6 9 12：\n', out_)
# 按列选取反对角线元素
indices_ = t.LongTensor([[3], [2], [1], [0]])
out_1 = t.gather(input=c, dim=1, index=indices_)
print('索引出3 6 9 12：\n', out_1)
# 选取两个对角线上的元素
index = t.LongTensor([[0, 1, 2, 3], [3, 2, 1, 0]]).t()
b = c.gather(1, index)
print('两个对角线上的元素、\n', b)

# scatter_
"""
与`gather`相对应的逆操作是`scatter_`，`gather`把数据从input中按index取出，
而`scatter_`是把取出的数据再放回去。注意`scatter_`函数是inplace操作。

out = input.gather(dim, index)
-->近似逆操作
out = Tensor()
out.scatter_(dim, index)
"""
# 把两个对角线元素放回去到指定位置
c = t.zeros(4, 4)
c.scatter_(1, index, b.float())
print('放回两个对角线上的元素：\n', c)

# python对象
"""
对tensor的任何索引操作仍是一个tensor，想要获取标准的python对象数值，
需要调用`tensor.item()`, 这个方法只对包含一个元素的tensor适用
"""
# 依旧是tensor
print('c[0,0] =', c[0, 0])  # tensor(0.)
# python float
print('a[0,0].item()=', c[0, 0].item())  # 0

d = c[0:1, 0:1, None]
print('d.shape=', d.shape)
# 只包含一个元素的tensor即可调用tensor.item,与形状无关
print('d.item() =', d.item())




