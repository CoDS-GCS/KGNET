import numpy as np
import pickle
from .similarityMetrics  import cosineSimilarity
from itertools import islice
import itertools
import faiss
class embeddingStore:
    def __init__(self):
        self.supportedStore="pickle,Faiss"  
        self.dataPath="/"
        self.datasetName=""
        self.embDic={}
class pickleInMemoryStore(embeddingStore):
    def __init__(self):
        embeddingStore.__init__(self)
        self.supportedStore="pickle"
    def add(self,embTechnique,picklePath):
        self.embDic[embTechnique]=pickle.load(open( self.dataPath+str(picklePath), "rb" ))
        print("pickleInMemoryStore added "+embTechnique+" for "+str(len(self.embDic[embTechnique]))+ " entities to dataset " +self.datasetName ) 
    def get(self,embTechnique):
        return self.embDic[embTechnique]
    def getEmbeddingVector(self,embTechnique,entity):
        return self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity]["emb"].values[0]
    def getSimilartyScore(self,embTechnique,entity1,entity2,similarityMetric):
        emb1=self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity1]["emb"].values[0]
        emb2=self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity2]["emb"].values[0]
        return similarityMetric.get(emb1, emb2)
    
    def searchTopSimilarEntites(self,embTechnique,entity1,similarityMetric,top=10):
        emb1=self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity1]["emb"]
#         print(emb1)
        dic={}
        for idx,row in self.embDic[embTechnique].iterrows():
            entity_name=row['entity']
            emb2=self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity_name]["emb"]
            Max_Sim=-2    
            for emb1_ins in emb1:
                for emb2_ins in emb2:        
                    result=similarityMetric.get(emb1_ins, emb2_ins)
                    if result>Max_Sim:
                        Max_Sim=result     

            dic[entity_name.replace("'","").replace("\"","").replace("","")]=Max_Sim   

        sorted_dict = {}
        sorted_keys = sorted(dic, key=dic.get,reverse=True)  # sort dic descending
        for w in sorted_keys:
            sorted_dict[w] = dic[w]
    #         print(sorted_dict) 
        sorted_dict=dict(itertools.islice(sorted_dict.items(), top+1)) #get frist top elements
        sorted_dict.pop(entity1,None)
        return sorted_dict

class FaissInMemoryStore(embeddingStore):
    def __init__(self):
        embeddingStore.__init__(self)
        self.supportedStore="Faiss"
        self.emb2dArrDic={}
        self.IndexFlatL2Dic={}
        self.IndexIVFFlat={}
        self.IndexIVFPQ={}
        
    def loadIndexFlatL2DFromFile(self,filePath,embTechnique,emb_df_dic):
        if embTechnique not in self.embDic:
            self.embDic[embTechnique] = emb_df_dic
        if embTechnique not in self.IndexFlatL2Dic:
            self.IndexFlatL2Dic[embTechnique] = faiss.read_index(filePath)
        print("Faiss IndexFlatL2 loaded from files for embedding " + embTechnique + " with " + str(
            self.IndexFlatL2Dic[embTechnique].ntotal) + " entity of dataset " + self.datasetName)

    def addToIndexFlatL2Dic(self,embTechnique,picklePath):
        if embTechnique not in self.embDic:
            self.embDic[embTechnique]=pickle.load(open( self.dataPath+str(picklePath), "rb" ))
            self.embDic[embTechnique]["emb_np"]=self.embDic[embTechnique]['emb'].apply(lambda x: np.array(x))    
        if embTechnique not in self.emb2dArrDic:
            self.emb2dArrDic[embTechnique]=self.embDic[embTechnique]["emb_np"].to_numpy()
            self.emb2dArrDic[embTechnique]=np.stack(self.emb2dArrDic[embTechnique])
        if embTechnique not in self.IndexFlatL2Dic:
            self.IndexFlatL2Dic[embTechnique] = faiss.IndexFlatL2(self.emb2dArrDic[embTechnique].shape[1])   # build the index
#         print("Faiss Index Trained=",self.faissIndexDic[key].is_trained)
            self.IndexFlatL2Dic[embTechnique].add(self.emb2dArrDic[embTechnique])                  # add vectors to the index
        print("Faiss IndexFlatL2 added "+embTechnique+" for "+ str(self.IndexFlatL2Dic[embTechnique].ntotal) +" entity to dataset " +self.datasetName )

    def addToIndexFlatL2Dic_df(self, embTechnique, emb_df):
            if embTechnique not in self.embDic:
                self.embDic[embTechnique] = emb_df
                self.embDic[embTechnique]["emb_np"] = self.embDic[embTechnique]['emb'].apply(lambda x: np.array(x))
            if embTechnique not in self.emb2dArrDic:
                self.emb2dArrDic[embTechnique] = self.embDic[embTechnique]["emb_np"].to_numpy()
                self.emb2dArrDic[embTechnique] = np.stack(self.emb2dArrDic[embTechnique])
            if embTechnique not in self.IndexFlatL2Dic:
                self.IndexFlatL2Dic[embTechnique] = faiss.IndexFlatL2(
                    self.emb2dArrDic[embTechnique].shape[1])  # build the index
                #         print("Faiss Index Trained=",self.faissIndexDic[key].is_trained)
                self.IndexFlatL2Dic[embTechnique].add(self.emb2dArrDic[embTechnique])  # add vectors to the index
            print("Faiss IndexFlatL2 added " + embTechnique + " for " + str(
                self.IndexFlatL2Dic[embTechnique].ntotal) + " entity to dataset " + self.datasetName)
    def addToIndexIVFFlatDic(self,embTechnique,picklePath):
        self.addToIndexFlatL2Dic(embTechnique,picklePath)
        if embTechnique not in self.IndexIVFFlat:
            nlist = 50
            self.IndexIVFFlat[embTechnique] = faiss.IndexIVFFlat(self.IndexFlatL2Dic[embTechnique],self.emb2dArrDic[embTechnique].shape[1],nlist)   # build the index
            self.IndexIVFFlat[embTechnique].train(self.emb2dArrDic[embTechnique])                  # add vectors to the index
            self.IndexIVFFlat[embTechnique].add(self.emb2dArrDic[embTechnique]) 
        print("Faiss IndexIVFFlat added "+embTechnique+" for "+ str(self.IndexIVFFlat[embTechnique].ntotal) +" entity to dataset " +self.datasetName )
    def addToIndexIVFFlatDic_df(self,embTechnique,emb_df):
        self.addToIndexFlatL2Dic_df(embTechnique,emb_df)
        if embTechnique not in self.IndexIVFFlat:
            nlist = 50
            self.IndexIVFFlat[embTechnique] = faiss.IndexIVFFlat(self.IndexFlatL2Dic[embTechnique],self.emb2dArrDic[embTechnique].shape[1],nlist)   # build the index
            self.IndexIVFFlat[embTechnique].train(self.emb2dArrDic[embTechnique])                  # add vectors to the index
            self.IndexIVFFlat[embTechnique].add(self.emb2dArrDic[embTechnique])
        print("Faiss IndexIVFFlat added "+embTechnique+" for "+ str(self.IndexIVFFlat[embTechnique].ntotal) +" entity to dataset " +self.datasetName )
    
    def getSimilartyScore(self,embTechnique,entity1,entity2,similarityMetric):
        emb1=self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity1]["emb_np"].values[0]
        emb2=self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity2]["emb_np"].values[0]
        return similarityMetric.get(emb1, emb2)
    
    def get(self,embTechnique):
        return self.faissIndexDic[embTechnique]
    def getIndexFlatL2EmbeddingVector(self,embTechnique,entity_uri):
        index = self.embDic[embTechnique].index
        condition = self.embDic[embTechnique]["entity"]==entity_uri
        condition_indices = index[condition]
        print(type(condition_indices))
        entity_idx=condition_indices.values[0]
        return self.IndexFlatL2Dic[embTechnique].reconstruct(int(entity_idx))

    def searchTopSimilarEntites_FlatL2D(self,embTechnique,entity1,similarityMetric,top=10):
        # entity_emb=self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity1]["emb_np"]
        # entity_emb= np.reshape(entity_emb.tolist()[0],(1,200))
        entity_emb=self.getIndexFlatL2EmbeddingVector(embTechnique,entity1)
        entity_emb = np.array([entity_emb])
        # print(type(entity_emb))
        D, I = self.IndexFlatL2Dic[embTechnique].search(entity_emb[:1], top+1)  # search
        dic={}
        for idx in I[0]:
            dis=similarityMetric.get(entity_emb[0].tolist(), self.emb2dArrDic[embTechnique][idx].tolist())
#             lst_scores.append(dis)
#             dic[self.embDic[embTechnique].iloc[idx]["entity"].replace("'","").replace("\"","").replace("","")]=dis
            dic[self.embDic[embTechnique].iloc[idx]["entity"]] = dis
        sorted_keys = sorted(dic, key=dic.get,reverse=True)  # sort dic descending
        sorted_dict = {}
        for w in sorted_keys:
#             print(w)
            sorted_dict[w] = dic[w]
        sorted_dict.pop(entity1,None)
        return sorted_dict
    def searchTopSimilarEntites_IVFFlat(self,embTechnique,entity1,top=10):
        entity_emb=self.embDic[embTechnique][self.embDic[embTechnique]["entity"]==entity1]["emb_np"]
        entity_emb= np.reshape(entity_emb.tolist()[0],(1,200))
        if key in self.IndexIVFFlat:
            D, I = self.IndexIVFFlat[embTechnique].search(entity_emb[:1], top+1)  # search
            dic={}
            for idx in I[0]:
                dis=self.cosineSimilarityObj.get(entity_emb[0].tolist(), self.emb2dArrDic[embTechnique][idx].tolist())
    #             lst_scores.append(dis)
                dic[self.embDic[embTechnique].iloc[idx]["entity"].replace("'","").replace("\"","").replace("","")]=dis
            sorted_keys = sorted(dic, key=dic.get,reverse=True)  # sort dic descending
            sorted_dict = {}
            for w in sorted_keys:
    #             print(w)
                sorted_dict[w] = dic[w]
            sorted_dict.pop(entity1,None)
            return sorted_dict
        return None
    
    
 
      
    
 

