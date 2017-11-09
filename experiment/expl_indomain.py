# -*- encoding:utf-8 -*-
from final import TASC,feature_handle
from sentiment_classify_method import ngram_sa_method
from experiment.expl_util import readTopicData,readNonTopicText,csv_to_train_test,classificationTest,saveResult

# 话题数据间互相训练测试
def topic_2_topic():
    topic
    resDict = {}
    topicData = readTopicData()
    for k1,v1 in topicData.iteritems():
        for k2, v2 in topicData.iteritems():
            train_set, train_label, test_set, test_label = csv_to_train_test(v1,v2,ratio=4,times=10)
            res = classificationTest(train_set, train_label,test_set, test_label)
            resDict[k1+'_'+k2] = res
    saveResult(res,saveAddr)

    return resDict

# 执行话题无关数据训练在各话题下测试
def nontopic_2_topic():
    nonTopicData = readNonTopicText()
    resDict = {}
    topicData = readTopicData()
    for k,v in topicData.iteritems():
        test_set, test_label = v['text'], v['label']
        train_set, train_label = nonTopicData['text'], nonTopicData['label']
        res = classificationTest(train_set, train_label,test_set, test_label)
        resDict[k] = res
    saveResult(res,saveAddr)
    return resDict


if __name__ == '__name__':
    # 1.一组话题无关数据
    # 2.测试话题及其数据
    # 3.加载TASC
    # 4.执行话题数据间互相训练测试
    # 5.执行话题无关数据训练在各话题下测试
    # 6.执行TASC训练在各话题下测试（测试不同的shortListNum参数，selectedNum参数）

    topicTestList = []
    textTestList = []
