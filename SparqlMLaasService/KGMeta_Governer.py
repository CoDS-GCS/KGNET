from rdflib import Graph
import rdflib
import os
import pandas as pd
import re
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
class KGMeta_Governer_rdflib():
    def __init__(self,ttl_file_path ='KGNET_MetaGraph.ttl',KGMeta_URI="http://kgnet"):
        self.ttl_file_path = ttl_file_path
        self.g = Graph()
        self.g.parse(self.ttl_file_path)
        self.KGMeta_URI=KGMeta_URI
    def executeSparqlquery(self,query):
        results = self.g.query(query)
        col_pattern = re.search(r'select\s+((\?)?[a-zA-Z_]+(\s+(\?)?[a-zA-Z_]+)*)\s+where', query,re.IGNORECASE)
        columns = re.findall(r'(?<=\?)[\w]+', col_pattern.group())
        return pd.DataFrame(results,columns=columns)


class KGMeta_Governer(sparqlEndpoint):
    def __init__(self,endpointUrl,KGMeta_URI="http://kgnet"):
        sparqlEndpoint.__init__(self, endpointUrl)
        self.KGMeta_URI = KGMeta_URI
    def insertTriples(self,triples_lst):
        """not implemented yet"""
    def deleteTriples(self,triples_lst):
        """not implemented yet"""
class KGMeta_OntologyManger(sparqlEndpoint):
    def __init__(self,endpointUrl,KGMeta_URI="http://kgnet"):
        sparqlEndpoint.__init__(self, endpointUrl)
        self.KGMeta_URI = KGMeta_URI
    def insertTriples(self,triples_lst):
        """not implemented yet"""
    def deleteTriples(self,triples_lst):
        """not implemented yet"""
if __name__ == '__main__':
    # kgmeta_govener = KGMeta_Governer_rdflib(ttl_file_path ='KGNET_MetaGraph.ttl')
    kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.98.118:8890/sparql',KGMeta_URI="http://kgnet")
    query = """ 
        SELECT distinct ?LinkPredictor ?gmlModel ?mID ?apiUrl
        WHERE
        {
            ?LinkPredictor <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <kgnet:types/LinkPredictor> .
            ?LinkPredictor <kgnet:GML/SourceNode> <dblp:author> .
            ?LinkPredictor <kgnet:GML/DestinationNode> <dblp:Affiliation> .
            ?LinkPredictor <kgnet:term/uses> ?gmlModel .
            ?gmlModel <kgnet:GML_ID> ?mID .
            ?mID <kgnet:API_URL> ?apiUrl .   
        }"""
    res_df = kgmeta_govener.executeSparqlquery(query)
    print(res_df)
