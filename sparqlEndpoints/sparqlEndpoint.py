import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON

class sparqlEndpoint:
    def __init__(self,endpointUrl="http://206.12.98.118:8890/sparql"):
        # self.endpointUrl="http://localhost:6190/sparql"
        self.endpointUrl = endpointUrl
#             self.endpoint="http://192.168.79.140:8890/sparql"            

#    Returns SparqlQuery As Dataframe
    def executeSparqlQuery(self,Sparql_query):    
        sparql = SPARQLWrapper(self.endpointUrl)
        sparql.setQuery(Sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # print(results)
        res_val=[]
        lst_columns=[]
        if len(results["results"]["bindings"])>0:
            lst_columns=results["results"]["bindings"][0].keys()
            # print(lst_columns)
            # df=pd.DataFrame(columns=lst_columns)
            # print("df=",df)
            for result in results["results"]["bindings"]:
                lst_values=[]
                for col in lst_columns:
                    lst_values.append(result[col]["value"])  
                res_val.append(lst_values)
        return pd.DataFrame(res_val, columns = lst_columns) 