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
from gensim import corpora,matutils
from collections import defaultdict

import re
from tools.util import dict2matrix
# 原始文本变为词列表-去非英文字符和停用词
def text_to_wordlist(oriText, remove_stopwords=True):
    # 1. Remove HTML
    pureText = BeautifulSoup(oriText,'lxml').get_text()
    #
    # 2. Remove non-letters
    pureText = re.sub("[^a-zA-Z!?.]", " ", pureText)
    #
    # 3. Convert words to lower case and split them
    words = pureText.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words(
            "english"))  # In Python, searching a set is much faster than searching a list, so convert the stop words to a set
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)

# 如果 outTrain 和 target 一样(同一个引用)，就认为是in domain test，则对其按比例随机划分训练集和测试集，
# 否则认为 outTrain是训练集，target是测试集
# ratio 保证训练集是测试集的ratio倍，否则进行删减
# 这里要注意使用out和target时得到的测试集大小是不一样的
def csv_to_train_test(outTrain,target,ratio=3,times=1):
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



def classify_test(train_set,train_label,test_set,test_label,reverseVetorize=False):
    return classify_test_21(train_set,train_label,test_set,test_label,reverseVetorize=False)

# 测试程序-使用gensim的字典再转换成矩阵-同时使用train和test构建字典
def classify_test_1(train_set,train_label,test_set,test_label,reverseVetorize=False):
    print 'final_sa_method:classify_test'
    all = train_set[:].tolist()
    all.extend(test_set)
    # 去停用词
    stoplist = set('for a of the and to in'.split())
    allTexts = [[word for word in text.lower().split() if word not in stoplist]
             for text in all]
    # 去低频词
    k = 2
    frequency = defaultdict(int)
    for text in allTexts:
        for token in text:
            frequency[token] += 1
    allTexts = [[token for token in text if frequency[token] > k]
             for text in allTexts]

    # 构建字典
    dictionary = corpora.Dictionary(allTexts)

    # 怎么把dict转化为列表形式的向量 http://www.mamicode.com/info-detail-1518042.html
    all_features = dict2matrix([dictionary.doc2bow(text) for text in allTexts]).toarray()
    train_data_features = all_features[0:len(train_set)]
    test_data_features = all_features[len(train_set):]

    #print train_data_features.toarray()

    classier = MultinomialNB() # RandomForestClassifier(n_estimators=100)
    classier = classier.fit( train_data_features , train_label  )

    print "Predicting test labels..."
    result = classier.predict(test_data_features)

    from sklearn.metrics import accuracy_score,confusion_matrix
    print accuracy_score(test_label, result)
    printlabels=['positive','negative']
    print printlabels
    print confusion_matrix(test_label, result,labels=printlabels)

# 测试程序-使用gensim的字典再转换成矩阵-只使用train构建字典
def classify_test_21(train_set,train_label,test_set,test_label,reverseVetorize=False):
    print 'final_sa_method:classify_test'
    all = train_set[:].tolist()
    all.extend(test_set)
    # 去停用词
    stoplist = set('for a of the and to in'.split())
    allTexts = [[word for word in text.lower().split() if word not in stoplist]
             for text in all]
    # 去低频词
    k = 2
    frequency = defaultdict(int)
    for text in allTexts:
        for token in text:
            frequency[token] += 1
    allTexts = [[token for token in text if frequency[token] > k]
             for text in allTexts]

    # 构建字典
    dictionary = corpora.Dictionary(allTexts[0:len(train_set)])

    # 怎么把dict转化为列表形式的向量 http://www.mamicode.com/info-detail-1518042.html
    num_terms = len(dictionary.keys())
    all_features = dict2matrix([dictionary.doc2bow(text) for text in allTexts],num_terms).toarray()
    train_data_features = all_features[0:len(train_set)]
    test_data_features = all_features[len(train_set):]

    #print train_data_features.toarray()

    classier = MultinomialNB() # RandomForestClassifier(n_estimators=100)
    classier = classier.fit( train_data_features , train_label  )

    print "Predicting test labels..."
    result = classier.predict(test_data_features)
    print 'result: ',result

    from sklearn.metrics import accuracy_score,confusion_matrix
    print accuracy_score(test_label, result)
    printlabels=[1,0] # 这个要对应实际的类别类型
    print printlabels
    print confusion_matrix(test_label, result,labels=printlabels)

# 测试程序-使用gensim的字典再转换成矩阵-只使用test构建字典
def classify_test_22(train_set,train_label,test_set,test_label,reverseVetorize=False):
    print 'final_sa_method:classify_test'
    all = train_set[:].tolist()
    all.extend(test_set)
    # 去停用词
    stoplist = set('for a of the and to in'.split())
    allTexts = [[word for word in text.lower().split() if word not in stoplist]
             for text in all]
    # 去低频词
    k = 2
    frequency = defaultdict(int)
    for text in allTexts:
        for token in text:
            frequency[token] += 1
    allTexts = [[token for token in text if frequency[token] > k]
             for text in allTexts]

    # 构建字典
    dictionary = corpora.Dictionary(allTexts[len(train_set):])

    # 怎么把dict转化为列表形式的向量 http://www.mamicode.com/info-detail-1518042.html
    all_features = dict2matrix([dictionary.doc2bow(text) for text in allTexts]).toarray()
    train_data_features = all_features[0:len(train_set)]
    test_data_features = all_features[len(train_set):]

    #print train_data_features.toarray()

    classier = MultinomialNB() # RandomForestClassifier(n_estimators=100)
    classier = classier.fit( train_data_features , train_label  )

    print "Predicting test labels..."
    result = classier.predict(test_data_features)

    from sklearn.metrics import accuracy_score,confusion_matrix
    print accuracy_score(test_label, result)
    printlabels=['positive','negative']
    print printlabels
    print confusion_matrix(test_label, result,labels=printlabels)

# 测试程序-使用sklearn的vectorizer
def classify_test_3(train_set,train_label,test_set,test_label,reverseVetorize=False):

    clean_train_set = []
    for i in train_set:
        clean_train_set.append(" ".join(text_to_wordlist(i, True)))
    clean_test_set = []
    print "Processing test set..."
    for i in test_set:
        clean_test_set.append(" ".join(text_to_wordlist(i, True)))
    print "Creating the bag of words..."
    # CountVectorizer scikit-learn 的词袋工具。遵循fit transform的操作使用
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
    if reverseVetorize:
        vectorizer.fit(clean_test_set)
        train_data_features = vectorizer.transform(clean_train_set).toarray()
    else:
        train_data_features = vectorizer.fit_transform(clean_train_set).toarray()

    test_data_features = vectorizer.transform(clean_test_set).toarray()
    print "Training the classifier ..."
    classier = MultinomialNB() # RandomForestClassifier(n_estimators=100)
    classier = classier.fit( train_data_features , train_label  )

    print "Predicting test labels..."
    result = classier.predict(test_data_features)

    from sklearn.metrics import accuracy_score,confusion_matrix
    print accuracy_score(test_label, result)
    printlabels=['positive','negative']
    print printlabels
    print confusion_matrix(test_label, result,labels=printlabels)

def classify_output(train_set,train_label,test_set):
    pass

if __name__ =='__main__':
    import datetime
    starttime = datetime.datetime.now()

    sent140 = pd.read_csv('../data/dealed_sent140-noNeu.txt',
                          names=['topic', 'label', 'id', 'text'], delimiter="\t", quoting=3)

    train_set, train_label, test_set, test_label = csv_to_train_test(sent140,sent140,ratio=3,times=1)
    classify_test_1(train_set, train_label, test_set, test_label)
    classify_test_21(train_set, train_label, test_set, test_label)
    classify_test_22(train_set, train_label, test_set, test_label)
    sencondtime = datetime.datetime.now()
    classify_test_3(train_set, train_label, test_set, test_label)
    thirdtime =  datetime.datetime.now()

    print (sencondtime-starttime).seconds,(thirdtime-sencondtime).seconds