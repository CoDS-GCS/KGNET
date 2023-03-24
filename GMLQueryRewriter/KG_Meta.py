from rdflib import Graph
import rdflib
import os
import pandas as pd
import re

def kgnet_getModelURI(query,PATH_TO_KG = os.path.join('.','GMLQueryRewriter','KGNET_MetaGraph.ttl')):
    g = Graph()
    g.parse(PATH_TO_KG,format=rdflib.util.guess_format(PATH_TO_KG, fmap=None))
    results = g.query(query)
    for row in results:
        return (row[0])

class KgMeta():
    def __init__(self,PATH_TO_KG = os.path.join('.','GMLQueryRewriter','KGNET_MetaGraph.ttl')):
        self.PATH_TO_KG = PATH_TO_KG 
        self.g = Graph()
        self.g.parse(self.PATH_TO_KG)
        
    def query(self,user_query):
        results = self.g.query(user_query)
        col_pattern = re.search(r'select\s+((\?)?[a-zA-Z_]+(\s+(\?)?[a-zA-Z_]+)*)\s+where', user_query,re.IGNORECASE)
        columns = re.findall(r'(?<=\?)[\w]+', col_pattern.group())
        return pd.DataFrame(results,columns=columns)

    
    