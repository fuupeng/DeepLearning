#!/bin/usr/env python3
# -*- coding utf-8 -*-

from __future__ import print_function
import torch as t

# 5. 逐元操作
"""
这部分操作会对tensor的每一个元素(point-wise，又名element-wise)进行操作，此类操作的输入与输出形状一致.

常见的逐元素操作
函数	                                功能
abs/sqrt/div/exp/fmod/log/pow..	   绝对值/平方根/除法/指数/求余/求幂..
cos/sin/asin/atan2/cosh..	      相关三角函数
ceil/round/floor/trunc	         上取整/四舍五入/下取整/只保留整数部分
clamp(input, min, max)	         超过min和max部分截断
sigmod/tanh..	                激活函数
对于很多操作，例如div、mul、pow、fmod等，PyTorch都实现了运算符重载，所以可以直接使用运算符。如a ** 2 等价于torch.pow(a,2), a * 2等价于torch.mul(a,2)

其中clamp(x, min, max)的输出满足以下公式：
    min , if xi<min
y = xi, if min<=xi<=max
    max, if xi >max
clamp常用在某些需要比较大小的地方，如取一个tensor的每个元素与另一个数的较大/小值   
"""
a = t.arange(0, 6).view(2, 3).float()
# 余弦函数
print('t.cos(tensor_a)=', t.cos(a))
# 求余
print('t.fmod(tensor_a, 3)=', t.fmod(input=a, other=3))
print('tensor_a % 3 =', a % 3)
# 平方根
print('t.sqrt(tensor_a, 2)=', t.sqrt(a))
# 幂次方
print('tensor_a ** 2 =', a ** 2)
print('t.pow(tensor_a, 2)', t.pow(input=a, exponent=2))
# 取a中的每一个元素与3相比较大的一个 (小于3的截断成3)
print('tensor_a=', a)
print('a中大于3：', t.clamp(input=a, min=3, max=5))

# 效果同 a = a.sin();a=a ,但是更高效节省显存
b = a.sin_()
print('a=', b)
print('a=', a)

# 6.归并操作
"""
此类操作会使输出形状小于输入形状，并可以沿着某一维度进行指定操作。
如加法sum，既可以计算整个tensor的和，也可以计算tensor中每一行或每一列的和。常用的归并操作如下所示。

常用归并操作
函数	                       功能
mean/sum/median/mode	均值/和/中位数/众数
norm/dist	            范数/距离
std/var	                标准差/方差
cumsum/cumprod	        累加/累乘
以上大多数函数都有一个参数**dim**，用来指定这些操作是在哪个维度上执行的。
关于dim(对应于Numpy中的axis)的解释众说纷纭，这里提供一个简单的记忆方式：

假设输入的形状是(m, n, k)

如果指定dim=0，输出的形状就是(1, n, k)或者(n, k)
如果指定dim=1，输出的形状就是(m, 1, k)或者(m, k)
如果指定dim=2，输出的形状就是(m, n, 1)或者(m, n)
size中是否有"1"，取决于参数keepdim，keepdim=True会保留维度1。
注意，以上只是经验总结，并非所有函数都符合这种形状变化方式，如cumsum。
"""
b = t.ones(3, 3)
print('a=', b)
# 二维的轴0 或 -1
print('0 轴求和，保留形状')
b = b.sum(dim=0, keepdim=True)
print('a=', b)
print('0 轴求和，不保留形状')
b = b.sum(dim=0, keepdim=False)
print('a=', b)

print('1 轴求和，保留形状')
b = b.sum(dim=-1, keepdim=True)
print('a=', b)
print('1 轴求和，不保留形状')
b = b.sum(dim=-1, keepdim=False)
print('a=', b)

c = t.ones(3, 3, 3)
print('c=', c)
print('0 轴求和，保留形状')
print('c.sum=', c.sum(dim=0, keepdim=True))
print('1 轴求和，保留形状')
print('c.sum=', c.sum(dim=1, keepdim=True))
print('2 轴求和，保留形状')
print('c.sum=', c.sum(dim=2, keepdim=True))

d = t.arange(0, 6).view(2, 3)
print('d=', d)
# 沿着行累加
print('行-d.cumsum=', d.cumsum(dim=-1))
# 沿着列累加
print('列-d.cumsum=', d.cumsum(dim=0))

# 7.比较
"""
比较函数中有一些是逐元素比较，操作类似于逐元素操作，还有一些则类似于归并操作。

常用比较函数
函数	                功能
gt/lt/ge/le/eq/ne	大于/小于/大于等于/小于等于/等于/不等
topk	            最大的k个数
sort	            排序
max/min	            比较两个tensor最大最小值
表中第一行的比较操作已经实现了运算符重载，因此可以使用a>=a、a>a、a!=a、a==a，其返回结果是一个ByteTensor，可用来选取元素。
max/min这两个操作比较特殊，以max来说，它有以下三种使用情况：
t.max(tensor)：返回tensor中最大的一个数
t.max(tensor,dim)：指定维上最大的数，返回tensor和下标
t.max(tensor1, tensor2): 比较两个tensor相比较大的元素

至于比较一个tensor和一个数，可以使用clamp函数。
"""
e1 = t.linspace(0, 15, 6).view(2, 3)
print('e1=', e1)
e2 = t.linspace(15, 0, 6).view(2, 3)
print('e2=', e2)
# e1 e2大小比较
print('e1>e2:', e1 > e2)
# e1中大于e2的元素
print('e1>e2:', e1[e1 > e2])

print('e1中最大一个元素:', t.max(e1))
print('e1中按0轴方向最大元素\n:', t.max(e1, dim=0))  # 返回tensor和下标
print('e1中按1轴方向最大元素\n', t.max(e1, dim=1))
print('比较e1和10,得到较大的元素\n', t.clamp(e1, min=10))
print('比较e1 e2中较大的元素\n', t.max(e1, e2))

# 8.线性代数
"""
PyTorch的线性函数主要封装了Blas和Lapack，其用法和接口都与之类似.

常用的线性代数函数
函数	                                 功能
trace	                             对角线元素之和(矩阵的迹)
diag	                             对角线元素
triu/tril	                         矩阵的上三角/下三角，可指定偏移量
mm/bmm	                             矩阵乘法，batch的矩阵乘法
addmm/addbmm/addmv/addr/badbmm..	 矩阵运算
t	                                 转置
dot/cross	                         内积/外积
inverse	                             求逆矩阵
svd	                                 奇异值分解
需要注意的是, 矩阵的转置会导致存储空间不连续，需调用它的.contiguous方法将其转为连续。

奇异值分解：输出U S V.
U、V是两个正交矩阵，其中的每一列分别被称为左奇异向量和右奇异向量
他们和∑中对角线上的奇异值相对应，通常情况下我们只需要保留前k个奇异向量和奇异值即可，
其中U是m*k矩阵，V是n*k矩阵，∑是k*k方阵，从而达到减少存储空间的效果
经过SVD分解，矩阵的信息能够被压缩至更小的空间内进行存储，从而为PCA（主成分分析）、LSI(潜在语义索引）等算法做好了数学工具层面的铺垫.
"""

a = t.linspace(0, 15, 9).view(3, 3)
print('tensor_a=', a)
b = a.t()
print('a=', b)
print('b是否连续', b.is_contiguous())
b.contiguous()
print('b是否连续', b.is_contiguous())
print('对接线元素之和：', t.trace(b))
print('对接线元素之和：', t.trace(b))
print('矩阵的上三角：', t.triu(b))
print('矩阵的下三角：', t.tril(b))
print('奇异值分解：', t.svd(b))
