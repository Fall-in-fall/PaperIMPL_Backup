# -*- encoding:utf-8 -*-

import pandas as pd

def save_pure(fileAddr,newAddr):
    print 'processin '+fileAddr
    x = pd.read_csv(fileAddr,names=['topic','label','id','text'], delimiter="\t", quoting=3)
    x.drop(['topic', 'id'], axis=1, inplace=True)
    x['label'] = x['label'].replace( ['negative', 'positive'],[0, 1])
    x.to_csv(newAddr, index=False,header = False, sep='\t')

baseaddr = 'G:/data collection/TTSC/TTSC-paper data/experiment collection/try data/topic data/'
nameList = ['apple.txt', 'google.txt','microsoft.txt','twitter.txt']
for i in nameList:
    save_pure(baseaddr+i,baseaddr+i+'_')