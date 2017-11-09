# -*- encoding:utf-8 -*-

from sklearn.cluster import KMeans,AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn.metrics.pairwise import paired_cosine_distances,cosine_similarity,cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
# 按照聚类后，簇中的源域实例距离簇中的目标域实例的近似度(距离倒数)的和作为权重排序。
# 这样，所在类簇中有更多目标域实例的源域实例具有更高的权重，
# 并且距离类簇中的目标域实例更近的源域实例具有更高的权重。

# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# print kmeans.labels_
# print kmeans.predict([[0, 0], [4, 4]])
# print kmeans.cluster_centers_


# 注意聚类用的是相似度，实例排序用的是权重
# x=[x1,...], y=[y1,...]
def cluster_sim(x,y): #注意dist和sim是反过来的，这里只求sim就可以了
    return cosine_similarity(x,y)


# 传入 df_target 格式为,df_source DataFrame
# 返回 select_num 个标注实例 #KMeans(n_clusters=theK, n_jobs= -1)
def select_by_KCluster(df_target,df_source,select_num,cluster_method = AgglomerativeClustering(affinity='cosine',linkage='average') ):

    theK = len(df_target)
    cluster_method.set_params(n_clusters = theK)

    df_target['domain']= 1
    df_source['domain'] = 0

    df_final = pd.concat([df_target,df_source],axis=0,ignore_index=True) # 不设置 ignore_index 的话原来两个表中数据的索引号不变，设置了重新开始索引
    featureCols = [i for i in df_final.columns if i not in ['domain', 'class', 'label', 'text','topic','id']]

    df_final['added'] = 1 #添加一列特征全为1(补充项)。防止相似度为0的情况(此时距离无限远infinite)无法聚类
    featureCols.append('added')
    df_final['theClass'] = cluster_method.fit_predict(df_final[featureCols])

    # 在df_final_target 中建立类簇索引。以方便取得某个类簇的所有目标域实例
    df_final_target_index =  df_final[df_final['domain']==1].set_index('theClass',append=True)

    # 这里记得必须reset_index,不然weightList直接赋值上去的时候会对齐原来的index。原来的index是concat之后的，是从len(target)开始的(如果source是在concat后面的话)。
    df_final_source_copy =  df_final[df_final['domain']==0].copy().reset_index()

    # 计算与簇内目标域点的相似度之和作为权重。这里想办法能不能直接从聚类器的距离矩阵中取出来
    weightList = []

    targetClassIndex = df_final[df_final['domain'] == 1]['theClass'].values
    print targetClassIndex
    for index_i,row_i in  df_final_source_copy.iterrows():
        temp = 0
        if row_i['theClass']  in targetClassIndex: # 注意判断没有目标域数据中没有某个class，即该类簇没有目标域数据的情况，则默认该域内实例权重为0
            # loc前面是行，后面是列
            # 这里发现个问题，loc出来只有一行的时候返回格式是series，而且series直接转为DataFrame的时候它的形式(列数)是不变的(需要把Series放在[]里).
            # 所以必须判断是否Series，然后调用to_frame方法统一转换
            df_inCluster = df_final_target_index.query('theClass=={}'.format(row_i['theClass'])) # 取出class索引为row_i['class']的行的原始索引
            if len(df_inCluster)>0:
                temp = sum( cosine_similarity( [row_i.loc[featureCols]],df_inCluster[featureCols])[0] )
            else:
                print 'len(df_inCluster)'
            # 或者循环一个一个的求，如果相似度计算需要自定义cluster_sim的话，就只能一个一个的求了
            # for index_j,row_j in df_inCluster.iterrows():
            #     # 这里的row是 Series，所以loc只能索引列
            #     # 注意cosine_similarity传入格式[a1,a2 ],[b1,b2]，返回a和b之间的两两相似度
            #     temp+= cluster_sim( [row_i.loc[featureCols]],[row_j.loc[featureCols]])[0][0]
        weightList.append(temp)
    # df_final[ df_final['domain']==1 ] .apply(lambda x: kmeans.transform( x[0:-2] )[0][x['class']],axis='columns')
    print weightList
    df_final_source_copy['weight'] = pd.Series(weightList)

    # 这里记得reset_index。这样用 loc[0:select_num,featureCols] 取出来的才是排序后的top elect_num个。
    # 因为排序只改变现实顺序，不改变原有索引顺序。reset之后将重新排序顺序索引实例
    df_final_source_copy.sort_values('weight',ascending=False,inplace=True)
    df_final_source_copy.reset_index(inplace=True)

    # 返回筛选后实例的特征向量列
    return df_final_source_copy.loc[0:select_num,:]

def constrKMeans_selector(targetData,sourceData,select_num):
    pass


if __name__=='__main__':
    import datetime
    starttime = datetime.datetime.now()

    sent140 = pd.read_csv('D:/workspace/pycharmworkspace/PaperIMPL/data/dealed_sent140-noNeu.txt',
                          names=['topic', 'label', 'id', 'text'], delimiter="\t", quoting=3)
    se2016 = pd.read_csv('D:/workspace/pycharmworkspace/PaperIMPL/data/dealed2016all.txt',
                         names=['topic', 'label', 'id', 'text'], delimiter="\t", quoting=3)
    sent140.drop(['topic', 'id'], axis=1, inplace=True)
    se2016.drop(['topic', 'id'], axis=1, inplace=True)

    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    sent140_unigram = vectorizer.fit_transform(sent140['text']).toarray()
    sent140_final = pd.concat([sent140, pd.DataFrame( sent140_unigram ) ], axis=1)

    se2016_unigram = vectorizer.transform(se2016['text']).toarray()
    se2016_final = pd.concat([se2016, pd.DataFrame( se2016_unigram )], axis=1)

    df_target = sent140_final
    df_source = se2016_final
    select_num = len(df_target)*4

    cluster_method = AgglomerativeClustering(affinity='cosine', linkage='average')

    selected_instances = select_by_KCluster(df_target, df_source, select_num, cluster_method=cluster_method)

    print selected_instances

    from sentiment_classify_method.ngram_sa_method import classify_test,csv_to_train_test
    print '----out domain without TASC----'
    train_set, train_label, test_set, test_label =  csv_to_train_test(df_source, df_target, ratio=3, times=1)
    classify_test(train_set, train_label, test_set, test_label)
    print '----out domain with TASC----'
    train_set, train_label, test_set, test_label = selected_instances['text'], selected_instances['label'], df_target['text'], df_target['label']
    classify_test(train_set, train_label, test_set, test_label)
    print '----in domain----'
    train_set, train_label, test_set, test_label = csv_to_train_test(df_target, df_target, ratio=3, times=1)
    classify_test(train_set, train_label, test_set, test_label)

    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds
