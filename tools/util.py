# -*- encoding:utf-8 -*-
from scipy.sparse import csr_matrix
from gensim.matutils import corpus2dense

# 转换出来列数只能等于元组列表中记录的最大列数，而实际总字典的列数可能不只，应该要输入一个总列数参数。
def dict2matrix(textDict,num_terms):
    print 'final_sa_method:dict2matrix'
    data = []
    rows = []
    cols = []
    line_count = 0
    maxCols = 0
    for line in textDict:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
            if elem[0]>maxCols: maxCols=elem[0]
        line_count += 1
    if maxCols<num_terms : # 没有包含最后列元素，在最后行加入一个最后列元素以使得矩阵大小满足要求
        rows.append(line_count-1)
        cols.append(num_terms)
        data.append(0)
    elif maxCols>num_terms :
        raise Exception('maxCols>num_terms ')
    sparse_matrix = csr_matrix((data, (rows, cols)))  # 稀疏向量
    #print len(textDict),len(sparse_matrix.toarray()),len(matutils.corpus2csc(textDict))
    return  sparse_matrix
    #thematrix = sparse_matrix.toarray()  # 密集向量

# gensim字典形式的特征向量列表转换成vsm稀疏特征矩阵.这个方法貌似有问题，出来的列和行是反的
def dict2matrix_2(textDict,num_terms):
    corpus2dense(textDict, num_terms)