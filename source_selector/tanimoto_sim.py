# -*- encoding:utf-8 -*-

#计算谷本系数
def tanimoto(vec1, vec2):
    c1, c2, c3 = 0, 0, 0
    for i in range(len(vec1)):
        if vec1[i] == 1: c1 += 1
        if vec2[i] == 1: c2 += 1
        if vec1[i] == 1 and vec2[i] == 1: c3 += 1

    return c3 / (c1 + c2 - c3)