# -*- encoding:utf-8 -*-
import re
import gzip
import random
import collections
# 除去某些列表的评论
def exceptSomeReview():
    metapath='G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\pruned_metadata_exceptSome.txt.gz'
    reviewpath = 'G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\pruned_kcore_5.txt.gz'
    g = gzip.open(metapath, 'r')
    allAsin = set()
    countMeta = 0
    for line in g:
        countMeta+=1
        if countMeta % 100000 == 0: print str(countMeta * 100 / 9430088.0) + '%'
        allAsin.add(line.strip().split('\t')[0])

    rf = file('G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\pruned_kcore_5_exceptSome.txt', 'w')
    g2 = gzip.open(reviewpath, 'r')
    countReview = 0
    for line in g2:
        theasin = line.strip().split('\t')[0]
        countReview += 1
        if countReview % 100000 == 0: print str(countReview * 100 /  41135700.0) + '%'
        if theasin in allAsin:
            rf.write(line)
    rf.close()

def genSampleData():
    metapath='G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\pruned_metadata_exceptSome.txt.gz'
    reviewpath = 'G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\pruned_kcore_5_exceptSome.txt.gz'

    rf_meta = file('G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\50w_sample_meta.txt', 'w')
    rf_review = file( 'G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\50w_sample_review_no3.txt', 'w')

    #thelen = len( gzip.open(metapath, 'r').readlines())
    #print thelen
    g = gzip.open(metapath, 'r')
    sampleAsin = {}
    countMeta = 0
    aboutMetaCount = 500000
    sep = 3913493/aboutMetaCount
    countWrite = 0
    for line in g:
        countMeta += 1
        linesplit = line.strip().split('\t')
        if len(linesplit)<3: continue
        if countMeta%sep==0:
            countWrite+=1
            sampleAsin[linesplit[0]]= [0,line.strip()+'\n']
            if countWrite > aboutMetaCount-1:
                print aboutMetaCount
                break

    g2 = gzip.open(reviewpath, 'r')
    count = 0
    # 如果再次取出掉thescore!=3的，存在可能有些域的实例全部被筛掉了，这些域保存在meta里面就没有意义，
    # 后续提取的时候也容易出现问题，需要进行判断处理，写完实例后，再写入meta，把不包含实例的meta去掉
    sampleAsinKeySet = set(sampleAsin.keys())
    reviewCount = 0
    for line in g2:
        linesplit = line.strip().split('\t')
        theasin = linesplit[0]
        thescore = int(float(linesplit[1]))
        if theasin in sampleAsinKeySet:
            if thescore!=3:
                label = '1' if thescore>3 else '0'
                rf_review.write( '\t'.join([theasin,label,linesplit[2] ])+'\n' )
                sampleAsin[theasin][0] += 1
                reviewCount+=1
        count += 1
        if count % 100000 == 0:
            print str(count * 100 / 41135700.0) + '%'
    metaCount = 0
    for k,v in sampleAsin.iteritems():
        if v[0]>0:
            rf_meta.write(v[1])
            metaCount+=1
    rf_meta.close()
    print 'actually meta num and review num: ',metaCount,reviewCount

# 去掉只有3分实例的meta后 704494 18219578
# 不排除部分类别，去掉只有3分实例的meta后，805732 19960376
# 100w : 200547 5199670
# 50w : 94971 2472485
if __name__ == '__main__':
    genSampleData()


# meta数据格式 ['0001042335', 'Hamlet: Complete &amp; Unabridged', 'Books\r\r\n']
# review 数据格式 ['0000031887','3.0', '... .\r\n']