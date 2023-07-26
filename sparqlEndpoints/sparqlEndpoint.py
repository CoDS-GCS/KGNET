import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
import datetime
import requests
class sparqlEndpoint:
    """sparql Endpoint class to excute sparql query on a virtuoso RDF endpoint"""
    def __init__(self,endpointUrl="http://206.12.98.118:8890/sparql"):
        self.endpointUrl = endpointUrl
#    Returns SparqlQuery As Dataframe
    def executeSparqlWrapperQuery(self,Sparql_query):
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

    #    Returns SparqlQuery As Dataframe
    def executeSparqlQuery_dopost(self,query,firstRowHeader=True):
        """
        Execute sparql query through dopost and return results in form of datafarme.
        :param query:the sparql query string.
        :type query: str
        :param firstRowHeader: wherther to assume frist line as the dataframe columns header.
        :type firstRowHeader: boolean

        """
        body = {'query': query}
        headers = {
            # 'Content-Type': 'application/sparql-update',
            'Content-Type': "application/x-www-form-urlencoded",
            'Accept-Encoding': 'gzip',
            'Accept': 'text/tab-separated-values; charset=UTF-8'}
        r = requests.post(self.endpointUrl, data=body, headers=headers)
        if firstRowHeader:
            return pd.DataFrame([x.split('\t') for x in r.text.split('\n')[1:] if x],columns=r.text.split('\n')[0].split('\t'))
        else:
            return pd.DataFrame([x.split('\t') for x in r.text.split('\n')])
if __name__ == '__main__':
    e=sparqlEndpoint(endpointUrl="http://206.12.98.118:8890/sparql")
    res_df=e.executeSparqlQuery_dopost("select ?s ?p ?o from <https://linkedmdb.org> where {?s ?p ?o} limit 10")
    print(res_df)