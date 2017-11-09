# -*- encoding:utf-8 -*-
from collections import defaultdict
import pandas as pd

from final.TASC import TASC
from expl_util import readTopicData,readNonTopicText,csv_to_train_test,classificationTest,saveResult

def TASC_para_tuning():
    a,b,c = range(1,6),range(1,6),range(1,6)
    paraList = zip(*[a,b,c])
    maxRes = 0
    maxPara_w = 0
    for para_w in paraList:
        resDict = TASC_2_topic(para_w=para_w)
        temp =  sum([i[0] for i in resDict.values()])/len(resDict)
        if temp > maxRes:
            maxRes = temp
            maxPara_w = para_w
    print maxPara_w
    maxPara_w

# TASC方法特定话题下分类
def TASC_2_topic(saveAddr = '',para_w = [1,1,1]):
    tasc = TASC(para_w=para_w)
    resDict = {}
    topicData = readTopicData()
    for k,v in topicData.iteritems():
        selected_num = len(v) * 4
        shortlist_num = selected_num * 4
        selected_instances = tasc.get_instance_TASC(k,v,selected_num,shortlist_num)
        test_set, test_label = v['text'], v['label']
        train_set, train_label = selected_instances['text'], selected_instances['label']
        res = classificationTest(train_set, train_label,test_set, test_label)
        resDict[k] = res
    if saveAddr!='':
        saveResult(res,saveAddr)
    return resDict


# 混合TASC的实例和话题无关领域内实例训练
def mixdomain_2_topic():
    nonTopicData = readNonTopicText()
    tasc = TASC()
    resDict = {}
    topicData = readTopicData()
    for k,v in topicData.iteritems():
        selected_num = len(v) * 4
        shortlist_num = selected_num * 4
        selected_instances = tasc.get_instance_TASC(k,v,selected_num,shortlist_num)
        selected_instances = pd.concat([selected_instances,nonTopicData],axis=0)
        test_set, test_label = v['text'], v['label']
        train_set, train_label = selected_instances['text'], selected_instances['label']
        res = classificationTest(train_set, train_label,test_set, test_label)
        resDict[k] = res
    saveResult(res,saveAddr)
    return resDict


# 随机迁移特定话题下分类
def random_to_topic():
    randomData = readRandom()
    resDict = {}
    for k,v in nonTopicData.iteritems():
        test_set, test_label = v['text'], v['label']
        train_set, train_label = randomData['text'], randomData['label']
        res = classificationTest(train_set, train_label,test_set, test_label)
        resDict[k] = res
    saveResult(res,saveAddr)
    return resDict


    pass


if __name__ == '__name__':
    # 1.一组话题无关数据
    # 2.测试话题及其数据
    # 3.加载TASC
    # 4.执行话题数据间互相训练测试
    # 5.执行话题无关数据训练在各话题下测试
    # 6.执行TASC训练在各话题下测试（测试不同的shortListNum参数，selectedNum参数）

    # 输出测试指标：准确率，召回率，F1Score
    # 输出格式
    topicTestList = []
    textTestList = []
