# -*- encoding:utf-8 -*-

from source_selector.JSD import JSD_sims
from tools.util import dict2matrix
from final.feature_handle import Source

from gensim import corpora
from gensim.models import KeyedVectors
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
import math

from compiler.ast import flatten

def simple_ranking_selector(targetTopic,targetTextAll,allSourceDict,gensimDict,shortlist_num,vecModel,para_w = [1,1,1]):
    # vecModel = KeyedVectors.load_word2vec_format(
    #     'G:/data collection/TTSC/TTSC-paper data/Word vector data/vectors/glove.twitter.27B/word2vec_glove.twitter.27B.100d.txt',
    #     binary=False)
    try:
        targetTopic_vec = vecModel.wv[targetTopic]
    except:
        print '**\nnot find, use zeros vector\n**'
        targetTopic_vec = np.zeros(model.vector_size)
        # 如果找不到？待完善
        pass
    # 注意cosine_similarity和term_JSD_sims的传入格式([a1,a2],[b1,b2])，以及输出格式  [ [],... ]
    topic_vec_sims = cosine_similarity( [targetTopic_vec],[ s.topic_vec for s in allSourceDict.values() ] )[0]
    term_JSD_sims = JSD_sims( [ gensimDict.doc2bow(   flatten( [ text.split() for text in targetTextAll] )  ) ],
                              [s.term_dict for s in allSourceDict.values()])[0]

    target_avglen = sum( [len(j) for j in targetTextAll] )/len(targetTextAll)
    avglen_diffs= [ abs(i.instance_avglen - target_avglen) for i in allSourceDict.values()]

    topic_vec_sims = minmax_scale(topic_vec_sims)
    term_JSD_sims = minmax_scale(term_JSD_sims)
    avglen_diffs = minmax_scale(avglen_diffs)

    # zip将若干个列表元素依次合并为一个元组列表，然后降序排序获得
    rankList = zip(*[allSourceDict.keys(),topic_vec_sims+term_JSD_sims+avglen_diffs])
    rankList.sort(key=lambda x:x[1],reverse=True)
    pos_res = []
    neg_res = []

    pos_count = 0
    neg_count = 0

    neg_shortlist_num = shortlist_num / 2
    pos_shortlist_num = shortlist_num - neg_shortlist_num

    for i in rankList:
        if pos_count < pos_shortlist_num:
            pos_count += allSourceDict[ i[0] ].pos_instance_num
            pos_res.append(i[0])
        if neg_count < neg_shortlist_num:
            neg_count += allSourceDict[ i[0] ].neg_instance_num
            neg_res.append(i[0])
        if pos_count >= pos_shortlist_num and neg_count >= neg_shortlist_num:
            break

    return pos_res,neg_res

def precised_topic_ranking_selector(targetTopic,targetTextAll,gensimDict,allSourceDict,shortListNum):
    pass

if __name__ =='__main__':
    pass
    # vecModel = KeyedVectors.load_word2vec_format(
    #     'G:/data collection/TTSC/TTSC-paper data/Word vector data/vectors/glove.twitter.27B/word2vec_glove.twitter.27B.100d.txt',
    #     binary=False)
    # targetTopic = 'apple'
    # targetTextAll = ['Just apply for a job at @Apple, hope they call me lol',
    #  'Wow. Great deals on refurbed #iPad (first gen) models. RT: Apple offers great deals on refurbished 1st-gen iPads']
    #
    # pass
