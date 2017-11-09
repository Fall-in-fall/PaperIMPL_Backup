# -*- encoding:utf-8 -*-
from final.TASC import TASC
from final.feature_handle import Source
from sentiment_classify_method import ngram_sa_method
from experiment.expl_util import readTopicData,readNonTopicText,csv_to_train_test,classificationTest,saveResult


topicData = readTopicData()
# ------------ non to topic
nonTopicData = readNonTopicText()
resDict_non2topic = {}
for k, v in topicData.iteritems():
    print 'test in topic "{}"'.format(k)
    test_set, test_label = v['text'], v['label']
    train_set, train_label = nonTopicData['text'], nonTopicData['label']
    res = classificationTest(train_set, train_label, test_set, test_label)
    resDict_non2topic[k] = res
# ------------ transfer
tasc = TASC()  # tasc.get_instance_TASC('apple',topicData['pure_dealed2016all'],10000,15000)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

resDict_tasc1 = {}
for k, v in topicData.iteritems():
    print 'test in topic "{}"'.format(k)
    selected_num = len(v) * 4 if len(v)>1000 else 4000
    shortlist_num = selected_num * 4
    selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
    print 'len(selected_instances): ',len(selected_instances)
    test_set, test_label = v['text'], v['label']
    train_set, train_label = selected_instances['text'], selected_instances['label']
    res = classificationTest(train_set, train_label, test_set, test_label,classifier=MultinomialNB())#
    resDict_tasc1[k] = res

resDict_tasc2= {}
for k, v in topicData.iteritems():
    print 'test in topic "{}"'.format(k)
    selected_num = len(v) * 4
    shortlist_num = selected_num * 4
    selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
    print 'len(selected_instances): ',len(selected_instances)
    test_set, test_label = v['text'], v['label']
    train_set, train_label = selected_instances['text'], selected_instances['label']
    res = classificationTest(train_set, train_label, test_set, test_label,classifier =RandomForestClassifier(n_estimators=100))
    resDict_tasc2[k] = res


#----------------------------
k='twitter'
v=topicData[k]
resDict_tasc = {}
print 'test in topic "{}"'.format(k)
selected_num = len(v) * 4
shortlist_num = selected_num * 4
selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
print 'len(selected_instances): ', len(selected_instances)
test_set, test_label = v['text'], v['label']
train_set, train_label = selected_instances['text'], selected_instances['label']
res = classificationTest(train_set, train_label, test_set, test_label, classifier = RandomForestClassifier(n_estimators=100))
resDict_tasc[k] = res

