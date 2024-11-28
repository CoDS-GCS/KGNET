from __future__ import annotations
from ampligraph.latent_features import ComplEx as ampligraphComplEx
from ampligraph.latent_features import TransE as ampligraphTransE
from ampligraph.latent_features import DistMult as ampligraphDistMult
from ampligraph.latent_features import HolE as ampligraphHolE
from ampligraph.latent_features import ConvE as ampligraphConvE
from ampligraph.latent_features import ConvKB as ampligraphConvKB
##########################
from ampligraph.evaluation import train_test_split_no_unseen 
from typing import List
import attr
import numpy as np
import pandas as pd
############################
from gensim.models.word2vec import Word2Vec as W2V
# from pyrdf2vec.embedders import Embedder
# from pyrdf2vec.typings import Embeddings, Entities, SWalk
# from pyrdf2vec import RDF2VecTransformer
# from pyrdf2vec.walkers import RandomWalker
# from pyrdf2vec.samplers import PageRankSampler
# from pyrdf2vec.graphs import KG

# class MyWord2Vec(Embedder):
#     """Defines the Word2Vec embedding technique.
#     SEE: https://radimrehurek.com/gensim/models/word2vec.html
#     Attributes:
#         _model: The gensim.models.word2vec model.
#             Defaults to None.
#         kwargs: The keyword arguments dictionary.
#             Defaults to { min_count=0, negative=20, vector_size=500 }.
#     """
#
#     kwargs = attr.ib(init=False, default=None)
#     _model = attr.ib(init=False, type=W2V, default=None, repr=False)
#
#     def __init__(self, **kwargs):
#         self.kwargs = {
#             "min_count": 0,
#             "negative": 20,
#             "vector_size": 500,
#             **kwargs,
#         }
#         self._model = W2V(**self.kwargs)
#
#     def fit(
#         self, walks: List[List[SWalk]], is_update: bool = False
#     ) -> Embedder:
#         """Fits the Word2Vec model based on provided walks.
#         Args:
#             walks: The walks to create the corpus to to fit the model.
#             is_update: True if the new walks should be added to old model's
#                 walks, False otherwise.
#                 Defaults to False.
#         Returns:
#             The fitted Word2Vec model.
#         """
#         corpus = [walk for entity_walks in walks for walk in entity_walks]
#         self._model.build_vocab(corpus, update=is_update)
#         self._model.train(
#             corpus,
#             total_examples=self._model.corpus_count,
#             epochs=self._model.epochs,
#         )
#         return self
#
#     def transform(self, entities: Entities) -> Embeddings:
#         """The features vector of the provided entities.
#             Args:
#                 entities: The entities including test entities to create the
#                 embeddings. Since RDF2Vec is unsupervised, there is no label
#                 leakage.
#         Returns:
#             The features vector of the provided entities.
#         """
#         if not all([entity in self._model.wv for entity in entities]):
#             raise ValueError(
#                 "The entities must have been provided to fit() first "
#                 "before they can be transformed into a numerical vector."
#             )
#         return [self._model.wv.get_vector(entity) for entity in entities]


class GRL:
    def __init__(self):
        self.supportedGRL="ComplEx,TransE"  
        self.kgPath="/"
        self.datasetName=""
    def generateEmbeddings(self):
        return None
# class RDF2Vec(GRL):
#     def __init__(self,vectorSize=50):
#         GRL.__init__(self)
#         self.supportedGRL="RDF2Vec"
#         self.entities=[]
#         self.vectorSize=vectorSize
#         self.transformer=None
#         self.kg=None
#
#     def generateEmbeddings(self):
#             self.kg = KG(self.kgPath,fmt="ttl")
#             self.transformer = RDF2VecTransformer(
#                             MyWord2Vec(vector_size=self.vectorSize),
#                             walkers = [RandomWalker(4, 20, PageRankSampler())])
#             embeddings, literals = self.transformer.fit_transform(self.kg,self.entities)
#             df_embd=pd.DataFrame(embeddings)
#             df_embd.insert(0,"entity",self.entities)
#             df_embd=df_embd.dropna()
#             return df_embd
        
class ComplEx(GRL):
    def __init__(self,df_spo,k=50):
        GRL.__init__(self)
        self.supportedGRL="ComplEx"  
        self.datasetName=""
        self.k=k
        self.spo=df_spo
        self.entities=df_spo["subject"].drop_duplicates().tolist()
        self.model =None
    def generateEmbeddings(self):
            X_train=np.array(self.spo)
#             print(X_train)
            self.model = ampligraphComplEx(batches_count=30,epochs=50,k=int(self.k/2),eta=30,
                        optimizer='adam', optimizer_params={'lr':0.015},loss='multiclass_nll',regularizer='LP', 
                        regularizer_params={'p':3, 'lambda':0.0015}, seed=0, verbose=True)
            self.model.fit(X_train)
            df_embd=pd.DataFrame(self.model.get_embeddings(self.entities))
            df_embd.insert(0,"entity",self.entities)
            df_embd=df_embd.dropna()
            return df_embd

class TransE(GRL):
    def __init__(self,df_spo,k=50):
        GRL.__init__(self)
        self.supportedGRL="TransE"  
        self.datasetName=""
        self.k=k
        self.spo=df_spo
        self.entities = df_spo["subject"].drop_duplicates().tolist()
        self.model =None
    def generateEmbeddings(self):
            X_train=np.array(self.spo)
#             print(X_train)
            self.model = ampligraphTransE(batches_count=30,epochs=50,k=self.k,eta=30,
                    optimizer='adam', optimizer_params={'lr':0.015},loss='multiclass_nll',regularizer='LP', 
                    regularizer_params={'p':3, 'lambda':0.0015}, seed=0, verbose=True)
            self.model.fit(X_train)
            df_embd=pd.DataFrame(self.model.get_embeddings(self.entities))
            df_embd.insert(0,"entity",self.entities)
            df_embd=df_embd.dropna()
            return df_embd
                                          
class DistMult(GRL):
    def __init__(self,df_spo,k=50):
        GRL.__init__(self)
        self.supportedGRL="DistMult"  
        self.datasetName=""
        self.k=k
        self.spo=df_spo
        self.entities = df_spo["subject"].drop_duplicates().tolist()
        self.model =None
    def generateEmbeddings(self):
            X_train=np.array(self.spo)
#             print(X_train)
            self.model = ampligraphDistMult(batches_count=30,epochs=50,k=self.k,eta=30,
                    optimizer='adam', optimizer_params={'lr':0.015},loss='multiclass_nll',regularizer='LP', 
                    regularizer_params={'p':3, 'lambda':0.0015}, seed=0, verbose=True)
            self.model.fit(X_train)
            df_embd=pd.DataFrame(self.model.get_embeddings(self.entities))
            df_embd.insert(0,"entity",self.entities)
            df_embd=df_embd.dropna()
            return df_embd

class HolE(GRL):
    def __init__(self,df_spo,k=50):
        GRL.__init__(self)
        self.supportedGRL="HolE"  
        self.datasetName=""
        self.k=k
        self.spo=df_spo
        self.entities = df_spo["subject"].drop_duplicates().tolist()
        self.model =None
    def generateEmbeddings(self):
            X_train=np.array(self.spo)
#             print(X_train)
            self.model = ampligraphHolE(batches_count=30,epochs=50,k=int(self.k/2),eta=30,
                    optimizer='adam', optimizer_params={'lr':0.015},loss='multiclass_nll',regularizer='LP', 
                    regularizer_params={'p':3, 'lambda':0.0015}, seed=0, verbose=True)
            self.model.fit(X_train)
            df_embd=pd.DataFrame(self.model.get_embeddings(self.entities))
            df_embd.insert(0,"entity",self.entities)
            df_embd=df_embd.dropna()
            return df_embd
                                          
class ConvE(GRL):
    def __init__(self,df_spo,k=50):
        GRL.__init__(self)
        self.supportedGRL="ConvE"  
        self.datasetName=""
        self.k=k
        self.spo=df_spo
        self.entities = df_spo["subject"].drop_duplicates().tolist()
        self.model =None
    def generateEmbeddings(self):
            X_train=np.array(self.spo)
#             print(X_train)
            self.model = ampligraphConvE(batches_count=30,epochs=50,k=int(self.k/2),eta=30,
                    optimizer='adam', optimizer_params={'lr':0.015},loss='multiclass_nll',regularizer='LP', 
                    regularizer_params={'p':3, 'lambda':0.0015}, seed=0, verbose=True)
            self.model.fit(X_train)
            df_embd=pd.DataFrame(self.model.get_embeddings(self.entities))
            df_embd.insert(0,"entity",self.entities)
            df_embd=df_embd.dropna()
            return df_embd  
                                          
class ConvKB(GRL):
    def __init__(self,df_spo,k=50):
        GRL.__init__(self)
        self.supportedGRL="ConvKB"  
        self.datasetName=""
        self.k=k
        self.spo=df_spo
        self.entities = df_spo["subject"].drop_duplicates().tolist()
        self.model =None
    def generateEmbeddings(self):
            X_train=np.array(self.spo)
#             print(X_train)
            self.model = ampligraphConvKB(batches_count=30, seed=22, epochs=10, k=self.k, eta=50,
               embedding_model_params={'num_filters': 32, 'filter_sizes': [1], 'dropout': 0.3},
               optimizer='adam', optimizer_params={'lr': 0.01},
               loss='pairwise', loss_params={}, verbose=True)
            self.model.fit(X_train)
            df_embd=pd.DataFrame(self.model.get_embeddings(self.entities))
            df_embd.insert(0,"entity",self.entities)
            df_embd=df_embd.dropna()
            return df_embd
                                          
                                          
                                          