# -*- encoding:utf-8 -*-

import pandas as pd
from sklearn.cluster import KMeans,AgglomerativeClustering
import numpy as np

from source_selector import ranking_source_selector
from instance_selector import clustering_selector_sparse
from final.feature_handle import *
from gensim.models import KeyedVectors
from datetime import datetime

class TASC:
    instance_addr = ''
    para_w = [1,1,1]

    # 初始化，读取样本库特征文件
    def __init__(self,instance_addr='G:/data collection/TTSC/TTSC-paper data/processed Amazon Review Data/10w_sample_review_no3.txt',
                 vecModelAddr= 'G:/data collection/TTSC/TTSC-paper data/Word vector data/vectors/glove.twitter.27B/word2vec_glove.twitter.27B.100d.txt',
                 para_w = [1,1,1] ):
        self.instance_addr = instance_addr
        self.vecModelAddr = vecModelAddr
        self.allTermDict,self.gensimDict,self.allSourceDict = load_feature(self.instance_addr)
        self.vecModel = KeyedVectors.load_word2vec_format(self.vecModelAddr ,binary=False)
        self.para_w = para_w

    def re_init(self,instance_addr='G:/data collection/TTSC/TTSC-paper data/processed Amazon Review Data/10w_sample_review_no3.txt',
                 vecModelAddr= 'G:/data collection/TTSC/TTSC-paper data/Word vector data/vectors/glove.twitter.27B/word2vec_glove.twitter.27B.100d.txt',
                 para_w = [1,1,1]):
        self.instance_addr = instance_addr
        self.vecModelAddr = vecModelAddr
        self.allTermDict,self.gensimDict,self.allSourceDict = load_feature(self.instance_addr)
        self.vecModel = KeyedVectors.load_word2vec_format(self.vecModelAddr ,binary=False)
        self.para_w = para_w

    # 给定文件，生成特征
    @staticmethod
    def gen_featureFile(self,domain_addr,instance_addr,lowFreqK=50):
        gen_feature(domain_addr=domain_addr,
                    instance_addr=instance_addr,
                    lowFreqK=lowFreqK)

    # 为特定话题获取筛选数据。返回类型是 DataFrame
    def get_instance_TASC(self,targetTopic,targetTextAll,select_num, shortlist_num,pos_neg_ratio = 1,autoRatio = 0):
        print  'TASC getting instances for topic "{}", by set select_num={},shortlist_num = {},'\
            .format(targetTopic,select_num,shortlist_num)
        # if autoRatio>1:
        #     select_num = len(targetTextAll)*autoRatio
        #     shortlist_num = select_num*autoRatio

        firsttime = datetime.now()
        # 域筛选
        pos_sourceList,neg_sourceList = ranking_source_selector.simple_ranking_selector(targetTopic,targetTextAll,
                                                                    self.allSourceDict, self.gensimDict,shortlist_num,
                                                                     self.vecModel,self.para_w)
        secondtime = datetime.now()
        print 'ranking_source_selector running for ', (secondtime - firsttime).seconds, 's'
        lenpos = len(pos_sourceList)
        lenneg = len(neg_sourceList)

        print lenpos,lenneg

        ready_instances = []
        if lenpos>lenneg:
            for s in range(0,lenpos):
                ready_instances.extend( [ ins for ins in self.allTermDict[ pos_sourceList[s] ] if ins[0]=='1'] )
                if s<lenneg:
                    ready_instances.extend( [ ins for ins in self.allTermDict[ neg_sourceList[s] ] if ins[0]=='0'] )
        else:
            for s in range(0,lenneg):
                ready_instances.extend( [ ins for ins in self.allTermDict[ neg_sourceList[s] ] if ins[0]=='0'] )
                if s<lenpos:
                    ready_instances.extend( [ ins for ins in self.allTermDict[ pos_sourceList[s] ] if ins[0]=='1'] )
        print 'len(ready_instances): ',len(ready_instances)
        df_source = pd.DataFrame(ready_instances,columns=['label','text'])
        df_source['label'] = df_source['label'].astype(np.int64)
        df_target = pd.DataFrame(targetTextAll,columns=['text'])
        df_target.insert(0, 'label', pd.Series(np.ones(len(df_target)) * (-1), dtype=np.int64)) #必须跟df_source的列对齐

        # 聚类实例筛选
        cluster_method = AgglomerativeClustering(affinity='precomputed', linkage='average')
        gensimDict = False
        # selected_instances = clustering_selector_sparse.select_by_KCluster(df_target,df_source,select_num, gensimDict = False,cluster_method = cluster_method)
        # print 'by set gensimDict'
        selected_instances = clustering_selector_sparse.select_by_KCluster(df_target, df_source, select_num,
                                                                           gensimDict=self.gensimDict,
                                                                           cluster_method=cluster_method)

        thirdtime = datetime.now()
        print  'clustering_selector_sparse running for ', (thirdtime - secondtime).seconds, 's'

        return selected_instances[['label','text']]


        # 1、读取source，转化topic向量和词频分布，传入source_selector
        # 2、source_selector计算各个source的JSD和topic word sim，归一化相加，排序，

        # *、source的topic vec和 word distr，预先计算好存档。实验(如果实际应用的时候也应该是这样的)的时候直接读取
        #    因为要使用ngram-vsm，所以
        #           方案1：本地保存对所有source向量化的那个 vectorizer，然后所有的转化(对target和source)都使用这个vectorizer。
        #           方案2：对targetText向量化的 vectorizer，之后所有的转化(对source)都使用这个vectorizer。
        #               )
        #           (应该使用方案2，方案1在这个场景下没意义。但是要保存词分布的JSD，需要用所有样本实例的字典。
        #           。最后分类是把原始实例代进去，不保存unigram特征，因为分类特征和方法应该是可切换的)
        #    取得前K个source的实例满足总实例数>=shortListNum,输出k个source的实例所在地址，
        # 3、按名字地址读取其相应的instance，传入instance_selector
        # 4、instance_selector通过聚类过滤获取select_num个实例
        # 5、返回获取的实例

if __name__ == '__main__':
    # targetTopic = 'apple'
    # targetTextAll = ['Just apply for a job at @Apple, hope they call me lol',
    #  'Wow. Great deals on refurbed #iPad (first gen) models. RT: Apple offers great deals on refurbished 1st-gen iPads']
    # sent140 = pd.read_csv('D:/workspace/pycharmworkspace/PaperIMPL/data/dealed_sent140-noNeu.txt',
    #                       names=['topic', 'label', 'id', 'text'], delimiter="\t", quoting=3)
    # sent140.drop(['topic', 'id'], axis=1, inplace=True)
    #
    # df_target = sent140 #pd.DataFrame(targetTextAll)
    # df_target['label'] = df_source['label'].replace(['negative', 'positive'],[0, 1] ).astype(np.int64)
    #
    # test_set, test_label = df_target['text'],df_target['label']
    #
    # train_set, train_label = selected_instances['text'], selected_instances['label']
    # classify_test(train_set, train_label, test_set, test_label)
    pass