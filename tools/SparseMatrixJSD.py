# -*- encoding:utf-8 -*-
import logging
import itertools
import os
import heapq

import numpy
import scipy.sparse

from gensim import interfaces, utils, matutils
from six.moves import map as imap, xrange, zip as izip

#基于稀疏矩阵计算JSD，待实现
class SparseMatrixJSD(interfaces.SimilarityABC):
    """
    Compute similarity against a corpus of documents by storing the sparse index
    matrix in memory. The similarity measure used is cosine between two vectors.

    Use this if your input corpus contains sparse vectors (such as documents in
    bag-of-words format) and fits into RAM.

    The matrix is internally stored as a `scipy.sparse.csr` matrix. Unless the entire
    matrix fits into main memory, use `Similarity` instead.

    Takes an optional `maintain_sparsity` argument, setting this to True
    causes `get_similarities` to return a sparse matrix instead of a
    dense representation if possible.

    See also `Similarity` and `MatrixSimilarity` in this module.
    """
    def __init__(self, corpus, num_features=None, num_terms=None, num_docs=None, num_nnz=None,
                 num_best=None, chunksize=500, dtype=numpy.float32, maintain_sparsity=False):
        self.num_best = num_best
        self.normalize = True
        self.chunksize = chunksize
        self.maintain_sparsity = maintain_sparsity

        if corpus is not None:
            logger.info("creating sparse index")

            # iterate over input corpus, populating the sparse index matrix
            try:
                # use the more efficient corpus generation version, if the input
                # `corpus` is MmCorpus-like (knows its shape and number of non-zeroes).
                num_terms, num_docs, num_nnz = corpus.num_terms, corpus.num_docs, corpus.num_nnz
                logger.debug("using efficient sparse index creation")
            except AttributeError:
                # no MmCorpus, use the slower version (or maybe user supplied the
                # num_* params in constructor)
                pass
            if num_features is not None:
                # num_terms is just an alias for num_features, for compatibility with MatrixSimilarity
                num_terms = num_features
            if num_terms is None:
                raise ValueError("refusing to guess the number of sparse features: specify num_features explicitly")
            corpus = (matutils.scipy2sparse(v) if scipy.sparse.issparse(v) else
                      (matutils.full2sparse(v) if isinstance(v, numpy.ndarray) else
                       matutils.unitvec(v)) for v in corpus)
            self.index = matutils.corpus2csc(
                corpus, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                dtype=dtype, printprogress=10000).T

            # convert to Compressed Sparse Row for efficient row slicing and multiplications
            self.index = self.index.tocsr()  # currently no-op, CSC.T is already CSR
            logger.info("created %r", self.index)

    def __len__(self):
        return self.index.shape[0]

    def get_similarities(self, query):
        """
        Return similarity of sparse vector `query` to all documents in the corpus,
        as a numpy array.

        If `query` is a collection of documents, return a 2D array of similarities
        of each document in `query` to all documents in the corpus (=batch query,
        faster than processing each document in turn).

        **Do not use this function directly; use the self[query] syntax instead.**

        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.T  # convert documents=rows to documents=columns
            elif isinstance(query, numpy.ndarray):
                if query.ndim == 1:
                    query.shape = (1, len(query))
                query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T
            else:
                # default case: query is a single vector, in sparse gensim format
                query = matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)

        # compute cosine similarity against every other document in the collection
        result = self.index * query.tocsc()  # N x T * T x C = N x C
        if result.shape[1] == 1 and not is_corpus:
            # for queries of one document, return a 1d array
            result = result.toarray().flatten()
        elif self.maintain_sparsity:
            # avoid converting to dense array if maintaining sparsity
            result = result.T
        else:
            # otherwise, return a 2d matrix (#queries x #index)
            result = result.toarray().T
        return result
#endclass SparseMatrixJSD