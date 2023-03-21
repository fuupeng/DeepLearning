#!/bin/usr/env python3
# -*- coding utf-8 -*-

# 将新版本的特性导入当前版本中,且必须放在第一行
from __future__ import print_function

# 导入torch库
import torch as t

# 3. 高级索引
"""
高级索引可以看成是普通索引操作的扩展，
但是高级索引操作的结果一般不和原始的Tensor共享内存
"""
x = t.arange(0, 27).view(3, 3, 3)
print('x=', x)

print('x[[1, 2], [1, 2], [2, 0]]=', x[[1, 2], [1, 2], [2, 0]])
# 上面等价于下面
print('x[1,1,2]=\n', x[1, 1, 2])
print('x[2,2,0]=\n', x[2, 2, 0])

# 4. Tensor类型
"""
tensor数据类型
Data type	                     dtype	                     CPU tensor	             GPU tensor
32-bit floating point	    torch.float32 or torch.float	torch.FloatTensor	     torch.cuda.FloatTensor
64-bit floating point	    torch.float64 or torch.double	torch.DoubleTensor	     torch.cuda.DoubleTensor
16-bit floating point	    torch.float16 or torch.half	    torch.HalfTensor	     torch.cuda.HalfTensor
8-bit integer (unsigned)	torch.uint8                  	torch.ByteTensor	     torch.cuda.ByteTensor
8-bit integer (signed)	    torch.int8	                    torch.CharTensor	     torch.cuda.CharTensor
16-bit integer (signed)	    torch.int16 or torch.short	    torch.ShortTensor	     torch.cuda.ShortTensor
32-bit integer (signed)	    torch.int32 or torch.int	    torch.IntTensor	         torch.cuda.IntTensor
64-bit integer (signed)	    torch.int64 or torch.long	    torch.LongTensor	     torch.cuda.LongTensor
各数据类型之间可以互相转换，type(new_type)是通用的做法，同时还有float、long、half等快捷方法。
CPU tensor与GPU tensor之间的互相转换通过tensor.cuda和tensor.cpu方法实现，此外还可以使用tensor.to(device)。
Tensor还有一个new方法，用法与t.Tensor一样，会调用该tensor对应类型的构造函数，生成与当前tensor类型一致的tensor。
torch.*_like(tensora) 可以生成和tensora拥有同样属性(类型，形状，cpu/gpu)的新tensor。 
tensor.new_*(new_shape) 新建一个不同形状的tensor。
"""
# 设置默认tensor，注意参数是字符串
t.set_default_tensor_type('torch.DoubleTensor')
a1 = t.Tensor(2, 3)
print('a1.dtype= ', a1.dtype)  # 现在a1是DoubleTensor,dtype是float64
# 恢复之前的默认设置
t.set_default_tensor_type('torch.FloatTensor')
# 把a转成FloatTensor，等价于b=a.type(t.FloatTensor)
b1 = a1.float()
print('b1.dtype=', b1.dtype)  # torch.float32

# 等价于torch.DoubleTensor(2,3)，建议使用a1.new_tensor
print('a1.new(3,2) =', a1.new(3, 2))
# 等价于t.zeros(a.shape,dtype=a.dtype,device=a.device)
print('t.zeros_like(a1)=\n', t.zeros_like(a1))
# 可以修改某些属性
print('t.zeros_like(a1, dtype=t.int16) =\n', t.zeros_like(a1, dtype=t.int16))
print()
print('t.rand_like(a1)=', t.rand_like(a1))
print('a1.new_ones(4,5, dtype=t.int)=\n', a1.new_ones(4, 5, dtype=t.int))
print('a1.new_tensor([3,4]) =\n', a1.new_tensor([3, 4]))
