# -*- encoding:utf-8 -*-
from gensim import corpora,models,similarities
from pprint import pprint # pprint(obj)格式化打印任意数据结构
import numpy as pd




# 输入的是初始语料是句子字符串组成的列表texts，去除停用词
# 输出时已经转变为token
def remove_stopwords_sentenceStr2Token(documents):
    stoplist=set('for a of the and to in'.split())
    res =[[word for word in texts.lower().split() if word not in stoplist]
            for texts in documents]
    return res
# 默认输入的句子字符串已经拆成一个一个词，去除只出现k次的词
def remove_ktimes(texts, k):
    from collections import defaultdict
    frequency=defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token]+=1
    res=[[token for token in text if frequency[token]>k]
            for text in texts]
    print res
    return res

def get_affinity_matrix():
    # 初始语料是句子字符串组成的列表texts，
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
    texts = remove_stopwords_sentenceStr2Token(documents)
    texts = remove_ktimes(texts,1)
    # 初始语料构建字典corpora.Dictionary(texts)，
    dictionary=corpora.Dictionary(texts)
    dictionary.save('./tmp/deerwester.dict') # 持久化字典

    dictionary_again = corpora.Dictionary().load('./tmp/deerwester.dict')
    # 测试新句
    new_doc="Human computer interaction"
    new_vec=dictionary_again.doc2bow(new_doc.lower().split())
    print 'new_vec:',new_vec

    # dictionary.doc2bow()将语料中的句子字符串转换为 由每个词对应的(id,freq)组成的列表
    corpus=[dictionary_again.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('./tmp/deerwester.mm',corpus) # 序列化存储corpus

    #重新加载
    corpus_again = corpora.MmCorpus('./tmp/deerwester.mm')
    # 用corpus训练一个tfidf模型
    featureNum=len(dictionary_again.token2id.keys())
    index = similarities.SparseMatrixSimilarity(corpus_again, num_features=featureNum)
    sims = [ index[i] for i in corpus_again]
    return sims

if __name__ =='__main__':
    pprint( get_affinity_matrix() )