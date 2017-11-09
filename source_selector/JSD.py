# -*- encoding:utf-8 -*-

import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


# 计算词频分布的KL时，分母不影响相对顺序，所以直接用词频向量就行
#inputDict输入的是gensim词频元组列表，否则输入的a和b是nparray，格式同cosine_similarity格式
def JSD_sims(a,b,inputDict = True):
    print 'computing JSD_sims'
    JSD_func = JSD_dict if inputDict else JSD
    res = []
    for i in a:
        temp = []
        num_b = len(b)
        div_b = num_b/10
        count = 0.0
        for j in b:
            count+=1
            temp.append(1 - JSD_func(i,j))
            if count%div_b==0:
                print '{0}%'.format('%.3f'%( count*100/num_b) )
        res.append(temp)
    return np.array(res)

# 输入的是gensim的词频元组形式
def JSD_dict(a,b):
    a,b = dict(a),dict(b)
    sum_a,sum_b = sum(a.values()),sum(b.values())
    # 注意必须转换成概率分布之后再求 (p+q)/2。直接词频相加再除是错误的 x/a+y/b != (x+y)/(a+b)
    for k in a.keys(): a[k] = float(a[k]) / sum_a
    for k in b.keys(): b[k] = float(b[k]) / sum_b
    avg_2ab = dict(Counter(a)+Counter(b))
    res_a,res_b = 0,0
    for k,v in avg_2ab.iteritems():
        if a.has_key(k):
            temp = a[k]
            res_a += temp*math.log(temp*2/v)
        if b.has_key(k):
            temp = b[k]
            res_b += temp*math.log(temp*2/v)
    return res_a*0.5+res_b*0.5

def JSD(a,b):
    a = np.array(a, dtype='float')
    b = np.array(b, dtype='float')
    avg_ab = 0.5 * (a+b)
    res = 0.5*asymmetricKL(a, avg_ab)+0.5*asymmetricKL(b, avg_ab)
    return res

# 这里注意要手动转换向量为概率分布，以及概率分布为0时的处理.对于p为0直接去掉或者eps
# (KL散度在Q出现0的时候可以加一个eps(较小值)解决, JSD算KL时输入的是avg_ab，本身就避免了q分布为0的情况，
# 但是矩阵很稀疏的时候所有都加入一个较小值极大增加存储空间，所以还是在计算的时候想办法
# http://blog.sina.com.cn/s/blog_4c9dc2a10102vlqp.html
def asymmetricKL(p, q):
    p = (p+1e-12)/sum(p)
    q = (q+1e-12)/sum(q)
    temp = (p/q)

    res = sum(  p * [ math.log(x) for x in temp ] ) # calculate the kl divergence between P and Q
    return res

def symmetricalKL(P, Q):
    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00