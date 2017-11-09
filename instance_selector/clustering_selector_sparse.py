# -*- encoding:utf-8 -*-

from sklearn.cluster import KMeans,AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from gensim import corpora,models,similarities
from pprint import pprint # pprint(obj)格式化打印任意数据结构
import numpy as np
from collections import defaultdict
from datetime import datetime
# 按照聚类后，簇中的源域实例距离簇中的目标域实例的近似度(距离倒数)的和作为权重排序。
# 这样，所在类簇中有更多目标域实例的源域实例具有更高的权重，
# 并且距离类簇中的目标域实例更近的源域实例具有更高的权重。


# x=[x1,...], y=[y1,...]
def cluster_sim(x,y): #注意dist和sim是反过来的，这里只求sim就可以了
    return cosine_similarity(x,y)

# 得到距离矩阵
def get_affinity_matrix(allTextList,gensimDict = False):
    print 'clustering: get_affinity_matrix'
    start = datetime.now()
    # 构建字典
    if gensimDict is False:
        # 去停用词
        stoplist = set('for a of the and to in'.split())
        texts = [[word for word in text.lower().split() if word not in stoplist]
               for text in allTextList]
        # 去低频词
        k=2
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > k]
               for text in texts]
        dictionary=corpora.Dictionary(texts) # dictionary.doc2bow()将语料中的句子字符串转换为 由每个词对应的(id,freq)组成的列表
    else:
        dictionary = gensimDict
        texts = [ text.lower().split() for text in allTextList]
    corpus = [dictionary.doc2bow(text) for text in texts]

    featureNum = len(dictionary.token2id.keys())
    index = similarities.SparseMatrixSimilarity(corpus, num_features=featureNum)
    sims = [index[i] for i in corpus]
    print 'clustering: get_affinity_matrix end with ' , (datetime.now()- start).seconds, 's'
    return np.array( sims )

# 传入 df_target 格式为,df_source DataFrame
# 返回 select_num 个标注实例 #KMeans(n_clusters=theK, n_jobs= -1)
def select_by_KCluster(df_target,df_source,select_num,gensimDict = False, cluster_method = AgglomerativeClustering(affinity='precomputed', linkage='average') ):
    print 'select_by_KCluster'

    theK = len(df_target)
    cluster_method.set_params(n_clusters = theK)

    df_target['domain']= 1
    df_source['domain'] = 0

    df_final = pd.concat([df_target,df_source],axis=0,ignore_index=True) # 不设置 ignore_index 的话原来两个表中数据的索引号不变，设置了重新开始索引

    affinity_matrix = get_affinity_matrix(df_final['text'],gensimDict = gensimDict)

    theClass = cluster_method.fit_predict(affinity_matrix)

    df_final['theClass'] = theClass
    # 在df_final_target 中建立类簇索引。以方便取得某个类簇的所有目标域实例
    df_final_target_index =  df_final[df_final['domain']==1].set_index('theClass',append=True)

    # 这里记得必须reset_index,不然weightList直接赋值上去的时候会对齐原来的index。原来的index是concat之后的，是从len(target)开始的(如果source是在concat后面的话)。
    df_final_source_copy =  df_final[df_final['domain']==0].copy().reset_index()

    # 计算与簇内目标域点的相似度之和作为权重
    weightList = []

    targetClassIndex =  df_final[df_final['domain']==1]['theClass'].values
    # print 'targetClassIndex',targetClassIndex
    for index_i,row_i in  df_final_source_copy.iterrows():
        temp = 0
        if row_i['theClass']  in targetClassIndex: # 注意判断有没有目标域数据中没有某个class，即该类簇没有目标域数据的情况，则默认该域内实例权重为0
            # loc前面是行，后面是列
            # 这里发现个问题，loc出来只有一行的时候返回格式是series，而且series直接转为DataFrame的时候它的形式(列数)是不变的(需要把Series放在[]里).
            # 所以必须判断是否Series，然后调用to_frame方法统一转换
            df_inCluster_index = df_final_target_index.query('theClass=={}'.format(row_i['theClass'])).index.labels[0]
            if len(df_inCluster_index)==0:
                print row_i['theClass']
            temp = sum( affinity_matrix[index_i,df_inCluster_index ] )
            temp = sum( affinity_matrix[index_i,df_inCluster_index ] )
        weightList.append(temp)

    #print 'weightList',weightList
    df_final_source_copy['weight'] = pd.Series(weightList)

    # 这里记得reset_index。这样用 loc[0:select_num,featureCols] 取出来的才是排序后的top elect_num个。
    # 因为排序只改变现实顺序，不改变原有索引顺序。reset之后将重新排序顺序索引实例
    df_final_source_copy.sort_values('weight',ascending=False,inplace=True)
    df_final_source_copy.reset_index(inplace=True)

    # 返回筛选后实例的特征向量列
    return df_final_source_copy.loc[0:select_num-1,:] # pandas的loc方法中，both the start and the stop are included!.

def constrKMeans_selector(targetData,sourceData,select_num):
    pass



if __name__=='__main__':
    import datetime
    starttime = datetime.datetime.now()

    #359
    sent140 = pd.read_csv('D:/workspace/pycharmworkspace/PaperIMPL/data/dealed_sent140-noNeu.txt',
                          names=['topic', 'label', 'id', 'text'], delimiter="\t", quoting=3)
    sent140.drop(['topic', 'id'], axis=1, inplace=True)

    # #906
    sanders = pd.read_csv('D:/workspace/pycharmworkspace/PaperIMPL/data/sanders_noNeuNAIrr.txt',
                      names=['topic','label','id','text'], delimiter="\t", quoting=3)
    sanders.drop(['topic', 'id'], axis=1, inplace=True)
    #
    # #2034
    # sts = pd.read_csv('D:/workspace/pycharmworkspace/PaperIMPL/data/dealed_sts_gold_tweet.txt',
    #           names=['id','label','text'], delimiter="\t", quoting=3);sts['label']=sts['label'].replace([0,4],['negative','positive'] )
    # sts.drop(['id'], axis=1, inplace=True)
    # # 4558
    # se2016 = pd.read_csv('D:/workspace/pycharmworkspace/PaperIMPL/data/dealed2016all.txt',
    #                      names=['topic', 'label', 'id', 'text'], delimiter="\t", quoting=3)
    # se2016.drop(['topic', 'id'], axis=1, inplace=True)

    # large
    sent140_dist= pd.read_csv('D:/workspace/pycharmworkspace/PaperIMPL/data/dealed_distant-sent140-noNeu.txt',
              names=['topic','label','id','text'], delimiter="\t", quoting=3).sample(n=5000)
    sent140_dist.drop(['topic', 'id'], axis=1, inplace=True)


    #sent140_unigram = vectorizer.fit_transform(sent140['text']).toarray()
    #sent140_final = pd.concat([sent140, pd.DataFrame( sent14010_unigram ) ], axis=1)

    #se2016_unigram = vectorizer.transform(se2016['text']).toarray()
    #se2016_final = pd.concat([se2016, pd.DataFrame( se2016_unigram )], axis=1)

    df_target = sent140
    df_source =  sent140_dist #pd.concat( [sanders,sts,se2016],axis=0,ignore_index=True)
    select_num = len(df_target)*10

    cluster_method = AgglomerativeClustering(affinity='precomputed', linkage='average')

    selected_instances = select_by_KCluster(df_target, df_source, select_num, cluster_method=cluster_method)

    print selected_instances

    from sentiment_classify_method.ngram_sa_method import classify_test,csv_to_train_test

    print '----in domain----'
    # train_set, train_label, test_set, test_label = csv_to_train_test(df_target, df_target, ratio=3, times=1)
    # classify_test(train_set, train_label, test_set, test_label)
    test_set, test_label = df_target['text'],df_target['label']

    print '----out domain with TASC----'
    train_set, train_label = selected_instances['text'], selected_instances['label']
    classify_test(train_set, train_label, test_set, test_label)

    print '----out domain without TASC----'
    #train_set, train_label, test_set, test_label =  csv_to_train_test(df_source, df_target, ratio=3, times=1)
    train_set, train_label = df_source['text'], df_source['label']
    classify_test(train_set, train_label, test_set, test_label)

    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds,'s'

    # 接下来用 selected_instances 训练，在 se2016_final上测试

    #train_set, train_label, test_set, test_label = csv_to_train_test(se2016,sent140,ratio=3,times=1)
    #selected_instance = select_by_KCluster(sent140[['text','label']],se2016[['text','label']] )
