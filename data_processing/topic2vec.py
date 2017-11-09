# -*- encoding:utf-8 -*-、

# -*- encoding:utf-8 -*-

import re
import gzip
import collections
from gensim.models import KeyedVectors

class Source:
    asin =''
    topic_vec = []
    term_dict = {}
    instance_avglen = 0
    def __init__(self,topic_vec,term_dict,instance_avglen):
        self.topic_vec = topic_vec
        self.term_dict = term_dict
        self.instance_avglen = instance_avglen



# 为元数据生成word vector
def genVector():
    res = collections.defaultdict(Source)
    model = KeyedVectors.load_word2vec_format(
        'G:/data collection/TTSC/TTSC-paper data/Word vector data/vectors/glove.twitter.27B/word2vec_glove.twitter.27B.100d.txt',
        binary=False)
    # model.wv['apple']: array([...,...])

def read2Dict():
    sourceDict


if __name__ == '__main__':
    pass