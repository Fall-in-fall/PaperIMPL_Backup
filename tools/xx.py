# -*- encoding:utf-8 -*-
from gensim import corpora,models,similarities


#nine documents,each consisting of only a single sentence
documents = ["Human machine interface for lab abc computer applications",
 			"A survey of user opinion of computer system response time",
 			"The EPS user interface management system",
			"System and human system engineering testing of EPS",
 			"Relation of user perceived response time to error measurement",
			"The generation of random binary unordered trees",
			"The intersection graph of paths in trees",
 			"Graph minors IV Widths of trees and well quasi ordering",
 			"Graph minors A survey"]

#remove commen words and tokenize
stoplist=set('for a of the and to in'.split())
texts=[[word for word in document.lower().split() if word not in stoplist]
		for document in documents]

#remove words that appear only once
print texts
from collections import defaultdict
frequency=defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts=[[token for token in text if frequency[token]>1]
		for text in texts]

print texts

from pprint import pprint
pprint(texts)

dictionary=corpora.Dictionary(texts)
dictionary.save('./tmp/deerwester.dict') #store the dictionary,for future reference

print(dictionary)
print(dictionary.token2id)

new_doc="Human computer interaction"
#the word 'interaction' does not appear in the dictionary and is ignored
new_vec=dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

corpus=[dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('./tmp/deerwester.mm',corpus) #store to dist,for later use
pprint(corpus)


#set tfidf model
tfidf=models.TfidfModel(corpus)

featureNum=len(dictionary.token2id.keys())
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=featureNum)
sims=index[tfidf[new_vec]]

print(sims)