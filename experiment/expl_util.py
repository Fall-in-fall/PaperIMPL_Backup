# -*- encoding:utf-8 -*-
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.utils import shuffle
from gensim import corpora
from collections import defaultdict
from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix
import os

from tools.util import dict2matrix

# import sys
# sys.path.append("D:/workspace/pycharmworkspace/PaperIMPL")

# https://www.zhihu.com/question/33602314
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend([WORKING_DIR_AND_PYTHON_PATHS])


def readTopicData(topicDataDir  = 'G:/data collection/TTSC/TTSC-paper data/experiment collection/try data/topic data' ):
    topicFileList = os.listdir(topicDataDir)
    dataDict = defaultdict(pd.DataFrame)
    for f in topicFileList:
        if f[-1:] =='_':
            theData = pd.read_csv(topicDataDir+'/'+f, names=['label', 'text'], delimiter='\t', quoting=3)
            theData['label'] = theData['label'].astype(np.int64)
            keyName = f.split('.')[0] if f.__contains__('.') else f
            dataDict[keyName] = theData
    return dataDict

def readNonTopicText(addr = 'G:/data collection/TTSC/TTSC-paper data/experiment collection/try data/nontopicTrain.txt'):
    return pd.read_csv(addr, names=['label', 'text'], delimiter="\t", quoting=3)

def readRandom(addr = 'G:/data collection/TTSC/TTSC-paper data/experiment collection/try data/randomTrans.txt'):
    pass
# 如果 outTrain 和 target 一样(同一个引用)，就认为是in domain test，则对其按比例随机划分训练集和测试集，
# 否则认为 outTrain是训练集，target是测试集
# ratio 保证训练集是测试集的ratio倍，否则进行删减
# 这里要注意使用out和target时得到的测试集大小是不一样的
# 输入是DataFrame形式，至少包含text和label列
def csv_to_train_test(outTrain,target,ratio=4,times=1):
    if target is outTrain:
        print 'in domain'
        target=shuffle(target)
        trainSize=int(ratio/(ratio+1.0)*len(target))
        train_set = target["text"][:trainSize] ; train_label=target["label"][:trainSize ]
        test_set=target["text"][ trainSize:] ; test_label=target["label"][trainSize: ]
    else:
        if len(outTrain)/len(target)<ratio:
            trainSize=len(outTrain)
            testSize=len(outTrain)/ratio
        elif len(outTrain)/len(target)>ratio:
            trainSize=len(target)*ratio
            testSize=len(target)

        print 'cross domain'
        #test_reviews=target["text"].sample(testSize) ; test_label=target["label"].sample(testSize)
        target=shuffle(target)
        outTrain=shuffle(outTrain)

        sampleTrain=outTrain.sample(trainSize)
        sampleTest=target.sample(testSize)

        train_set = sampleTrain["text"] ; train_label = sampleTrain["label"]
        test_set = sampleTest["text"] ; test_label = sampleTest["label"]
    return train_set,train_label,test_set,test_label

def classificationTest(train_set,train_label,test_set,test_label,lowFreqK=2,classifier = MultinomialNB() ): # RandomForestClassifier(n_estimators=100)
    print 'classification test processing ...'
    all = train_set[:].tolist()
    all.extend(test_set)
    # 去停用词
    stoplist = set('for a of the and to in'.split())
    allTexts = [[word for word in text.lower().split() if word not in stoplist]
             for text in all]
    # 去低频词
    frequency = defaultdict(int)
    for text in allTexts:
        for token in text:
            frequency[token] += 1
    allTexts = [[token for token in text if frequency[token] > lowFreqK ]
             for text in allTexts]

    # 构建字典
    dictionary = corpora.Dictionary(allTexts[0:len(train_set)])

    # 怎么把dict转化为列表形式的向量 http://www.mamicode.com/info-detail-1518042.html
    num_terms = len(dictionary.keys())
    all_features = dict2matrix([dictionary.doc2bow(text) for text in allTexts],num_terms).toarray()
    train_data_features = all_features[0:len(train_set)]
    test_data_features = all_features[len(train_set):]

    classifier = classifier.fit( train_data_features , train_label  )
    result = classifier.predict(test_data_features)

    printlabels = [1, 0]  # 这个要对应实际的类别类型
    res = [accuracy_score(test_label, result),recall_score(test_label, result),f1_score(test_label, result),confusion_matrix(test_label, result,labels=printlabels)]
    return res



def drawHistogram(titleList,data):
    pass

def saveResult(data):
    pass

def oneTest():
    pass

if __name__ == '__main__':
    classificationTest()