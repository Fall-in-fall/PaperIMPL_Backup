# -*- encoding:utf-8 -*-

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        self.n_synth = int((self.N / 100.0) * self.n_samples)
        self.synthetic = np.zeros((self.n_synth, self.n_attrs))

        rand_indexes = np.random.permutation(self.n_samples)
        if self.N > 100:
            self.N = np.ceil(self.N / 100)
            for i in range(int(self.N) - 1):
                rand_indexes = np.append(rand_indexes, np.random.permutation(self.n_samples))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in rand_indexes[:self.n_synth]:
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self, i, nnarray):
        ## Choose a random number between 0 and k
        nn = np.random.randint(0, self.k)
        while nnarray[nn] == i:
            nn = np.random.randint(0, self.k)

        dif = self.samples[nnarray[nn]] - self.samples[i]
        gap = np.random.rand(1, self.n_attrs)

        self.synthetic[self.newindex] = self.samples[i] + gap.flatten() * dif
        self.newindex += 1
        return
# a=np.array([[1,0,3],[4,0,6],[2,0,1],[2,0,2],[2,0,4],[2,0,4]])
# s=Smote(a,N=20)
# print s.over_sampling()

def random_oversampling():
    pass
