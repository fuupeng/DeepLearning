#!/bin/usr/env python3
# -*- coding utf-8 -*-
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # ,假如有3块GPU可以用，对应真实gpu编号为1,2,3
import torch as t

# 10.CPU/GPU
"""
tensor可以很随意的在gpu/cpu上传输。
使用`tensor.cuda(device_id)`或者`tensor.cpu()`。
另外一个更通用的方法是`tensor.to(device)`.

注意
1.尽量使用tensor.to(device), 将device设为一个可配置的参数，这样可以很轻松的使程序同时兼容GPU和CPU.
2.数据在GPU之中传输的速度要远快于内存(CPU)到显存(GPU), 所以尽量避免频繁的在内存和显存中传输数据。
"""

use_gpu = [0,2] # 表示我想使用可用设备中的0,2号机器，对应真实gpu编号为1,3
if t.cuda.device_count() > 1:
    print('available gpus is ', t.cuda.device_count(), t.cuda.get_device_name())
    a = t.randn(3, 4, device=t.device('cuda:1'))  # 等价于a.t.randn(3,4).cuda(1)，但是比这个块
    print('a.device=', a.device)
else:
    print('no gpu is available!')
    a = t.randn(3, 4, device=t.device('cpu'))
    print('a.device=', a.device)

# 持久化
# Tensor的保存和加载十分的简单，使用t.save和t.load即可完成相应的功能。
# 在save/load时可指定使用的`pickle`模块，在load时还可将GPU tensor映射到CPU或其它GPU上。
if t.cuda.is_available():
    a = a.cuda(1)  # 把a转为GPU1上的tensor,
    t.save(a, 'a.pth')

    # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
    a = t.load('a.pth')
    # 加载为c, 存储于CPU
    c = t.load('a.pth', map_location=lambda storage, loc: storage)
    # 加载为d, 存储于GPU0上
    d = t.load('a.pth', map_location={'cuda:1': 'cuda:0'})

# 向量化
"""
向量化计算是一种特殊的并行计算方式，相对于一般程序在同一时间只执行一个操作的方式，
它可在同一时间执行多个操作，通常是对不同的数据执行同样的一个或一批指令，或者说把指令应用于一个数组/向量上。
向量化可极大提高科学运算的效率，Python本身是一门高级语言，使用很方便，但这也意味着很多操作很低效，尤其是`for`循环。
在科学计算程序中应当极力避免使用Python原生的`for循环`。
"""
import time

start1 = time.perf_counter()
def for_loop_add(x, y):
    result = []
    for i,j in zip(x, y):
        result.append(i + j)
    return t.Tensor(result)
end1 = time.perf_counter()
runTime1 = end1 -start1

start2 = time.perf_counter()
x = t.zeros(100)
y = t.ones(100)
end2 = time.perf_counter()
runTime2 = end2 - start2

print(runTime1)
print(runTime2)
print('runTime1-runTime2=', runTime1-runTime2)

"""
tips:
因此在实际使用中应尽量调用内建函数(buildin-function)，这些函数底层由C/C++实现，能通过执行底层优化实现高效计算。
因此在平时写代码时，就应养成向量化的思维习惯，千万避免对较大的tensor进行逐元素遍历。
"""

