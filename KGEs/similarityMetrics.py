from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import numpy as np
import torch
import torch.nn.functional as F

class similarityMetric:
    def __init__(self):
        self.supportedMetric="similarity metric"
    def get(self,v1,v2):
        return 0    

class cosineSimilarity(similarityMetric):
    def __init__(self):
        similarityMetric.__init__(self)
        self.supportedMetric="cosine similarity"
    def scipy(self,v1,v2):
        return 1 - spatial.distance.cosine(v1, v2)    
    def get(self,v1,v2):
        return self.scipy(v1, v2)    
    def numpy(self,v1,v2):
        return  dot(v1, v2)/(norm(v1)*norm(v2))
    def sklearn(self,v1,v2):
        v1=np.array(v1)
        v2=np.array(v2)
        return cosine_similarity(v1.reshape(1,-1),v2.reshape(1,-1))
    def torch(self,v1,v2):
        v1=torch.FloatTensor(v1)
        v2=torch.FloatTensor(v2)
        return F.cosine_similarity(v1, v2, dim=0)
    
class euclideanDistance(similarityMetric):
    def __init__(self):
        similarityMetric.__init__(self)
        self.supportedMetric="euclidean distance"
    def scipy(self,v1,v2):
        return spatial.distance.euclidean(v1, v2)   
    def get(self,v1,v2):
        return self.scipy(v1, v2)    
    
 

