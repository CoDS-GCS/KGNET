import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON

class sparqlEndpoint:
    def __init__(self):
        self.endpointUrl="http://localhost:6190/sparql"
#             self.endpoint="http://192.168.79.140:8890/sparql"            

#    Returns SparqlQuery As Dataframe
    def executeSparqlQuery(self,Sparql_query):    
        sparql = SPARQLWrapper(self.endpointUrl)
        sparql.setQuery(Sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # print(results)
        lst_values=[]
        df=pd.DataFrame()
        if len(results["results"]["bindings"])>0:
            lst_columns=results["results"]["bindings"][0].keys()
        #     print(lst_columns)
            df=pd.DataFrame(columns=lst_columns)

            for result in results["results"]["bindings"]:
                for col in lst_columns:
                    lst_values.append(result[col]["value"])  
                zipped = zip(lst_columns, lst_values)
                a_dictionary = dict(zipped)
                lst_values=[]
        #         print(a_dictionary)
                df=df.append(a_dictionary,ignore_index=True)

        return df 