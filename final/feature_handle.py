# -*- encoding:utf-8 -*-
import cPickle,pickle
from gensim import corpora
from gensim.models import KeyedVectors
import numpy as np
import collections
from compiler.ast import flatten
import re
class Source:
    asin =''
    topic_vec = []
    term_dict = {}
    instance_avglen = 0
    pos_instance_num = 0
    neg_instance_num = 0
    titleList = []
    cateList = []
    # def __init__(self,topic_vec=[],term_dict={},instance_avglen=0):
    #     self.topic_vec = topic_vec
    #     self.term_dict = term_dict
    #     self.instance_avglen = instance_avglen


def gen_feature(domain_addr = 'G:/data collection/TTSC/TTSC-paper data/processed Amazon Review Data/sample_meta.txt',
                instance_addr = 'G:/data collection/TTSC/TTSC-paper data/processed Amazon Review Data/sample_review_no3.txt',
                lowFreqK = 50 ):

    ## 读取所有实例，构建字典 allTermDict 并保存
    print ' 读取所有实例，构建字典 allTermDict 并保存 '
    allTermDict = collections.defaultdict(list)
    for line in open(instance_addr, 'r'):
        line_split = line.strip().split('\t')
        asin = line_split[0]
        allTermDict[asin].append(line_split[1:])
    cPickle.dump(allTermDict, file(instance_addr + '_allTermDict', 'w'))  # 保存 allTermDict

    ## 构建gensim词字典并保存，不进行去停用词操作，去低频词
    print '构建gensim词字典'
    allTextList = flatten( [ [ i[1] for i in asinAllterm ]
                             for asinAllterm in allTermDict.values()]  )
    stoplist = set('for a of the and to in'.split())
    # allTexts = [[word for word in text.lower().split() if word not in stoplist]
    #             for text in allTextList]
    allTexts = [ text.lower().split() for text in allTextList]

    frequency = collections.defaultdict(int)
    for text in allTexts:
        for token in text:
            frequency[token] += 1
    allTexts = [[token for token in text if frequency[token] > lowFreqK]
             for text in allTexts]
    gensimDict = corpora.Dictionary(allTexts)
    # gensimDict.save(file(instance_addr + '_gensimDict', 'w'))
    cPickle.dump(gensimDict, file(instance_addr + '_gensimDict', 'w') )

    ## 为每个source生成词分布 和 平均实例长度，放入allSourceDict
    print '为每个source生成词分布 和 平均实例长度，放入allSourceDict'
    allSourceDict = collections.defaultdict(Source)
    for k,v in allTermDict.iteritems():
        allSourceDict[k].term_dict = gensimDict.doc2bow(
            flatten( [ i[1].split(' ') for i in v] )  )
        allSourceDict[k].instance_avglen = sum( [len(j[1]) for j in v] )/len(v)
        pos_shortlist_num = 0
        neg_shortlist_num = 0

        # 写成dictCount的形式实现，待实现
        for i in v:
            if i[0] == '1' :pos_shortlist_num += 1
            if i[0] == '0': neg_shortlist_num += 1
        allSourceDict[k].pos_instance_num = pos_shortlist_num
        allSourceDict[k].neg_instance_num = neg_shortlist_num

    del allTermDict
    del gensimDict

    ## 读取domain并生成 topicvec,然后保存allSourceDict
    print '读取domain并生成 topicvec,然后保存allSourceDict'
    model = KeyedVectors.load_word2vec_format(
        'G:/data collection/TTSC/TTSC-paper data/Word vector data/vectors/glove.twitter.27B/word2vec_glove.twitter.27B.100d.txt',
        binary=False)
    vector_size= model.vector_size
    vec_dimention = 100
    for dline in open(domain_addr,'r'):
        dline_split = dline.strip().split('\t')
        source = Source
        asin = dline_split[0]
        titleList = re.sub('[^a-zA-Z]', ' ', dline_split[1].strip()).split(' ')
        cateList = re.sub('[^a-zA-Z]', ' ', dline_split[2].strip()).split(' ')
        avgvec_title = np.zeros(vector_size)
        title_validCount = 0
        title_invalid_count = 0
        for i in titleList: #实际上 title 可以看作是一个短句。直接相加也是建模句子的简单实现。
            try:
                avgvec_title+=model.wv[i]
                title_validCount+=1
            except:
                title_invalid_count+=1
        avgvec_cate = np.zeros(vector_size) # 任何向量与零向量的余弦相似度都是0
        cate_validCount = 0
        cate_invalid_count = 0
        for i in cateList:
            try:
                avgvec_cate += model.wv[i]
                cate_validCount+=1
            except:
                cate_invalid_count+=1
        allSourceDict[asin].titleList = titleList
        allSourceDict[asin].cateList = cateList
        avgvec_title = avgvec_title / (title_validCount if title_validCount > 0 else 1)
        avgvec_cate = avgvec_title / (cate_validCount if cate_validCount > 0 else 1)

        allSourceDict[asin].topic_vec = (avgvec_title+avgvec_cate)/ \
                                        ( 2 if title_invalid_count==0 and cate_invalid_count==0
                                          else 1)

    cPickle.dump(allSourceDict, file(instance_addr + '_allSourceDict', 'w'))  # 保存 allTermDict

    print 'gen_feature finished'

    # model.wv['apple']: array([...,...])

def load_feature(instance_addr):
    allTermDict = cPickle.load(file(instance_addr+'_allTermDict','r')) # { asin: [ [label,text],...  ]  }
    gensimDict = cPickle.load(file(instance_addr+'_gensimDict','r'))# corpora.Dictionary.load(instance_addr+'_gensimDict')
    allSourceDict =  cPickle.load(file(instance_addr+'_allSourceDict','r')) # { asin: Source  }
    return allTermDict,gensimDict,allSourceDict

if __name__ =='__main__':
    domain_addr = 'G:/data collection/TTSC/TTSC-paper data/processed Amazon Review Data/sample_meta.txt'
    instance_addr = 'G:/data collection/TTSC/TTSC-paper data/processed Amazon Review Data/sample_review_no3.txt'
    gen_feature(domain_addr,instance_addr)
    allTermDict, gensimDict, allSourceDict = load_feature(instance_addr)

    domain_addr = 'G:/data collection/TTSC/TTSC-paper data/processed Amazon Review Data/10w_sample_meta.txt'
    instance_addr = 'G:/data collection/TTSC/TTSC-paper data/processed Amazon Review Data/10w_sample_review_no3.txt'
    gen_feature(domain_addr,instance_addr)
    allTermDict, gensimDict, allSourceDict = load_feature(instance_addr)


