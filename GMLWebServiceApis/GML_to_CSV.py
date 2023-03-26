# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:54:04 2023

@author: walee
"""
import sys
import pandas as pd
# import GMLQueryRewriter.gmlRewriter as qw
import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
import os
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
                  try:
                    lst_values.append(result[col]["value"])  
                  except:
                    lst_values.append(None)  
                res_val.append(lst_values)
        return pd.DataFrame(res_val, columns = lst_columns) 

def mapVenues(filename,labels_path=r'./data/labelidx2labelname.csv'):

    label_info = pd.read_csv(labels_path)
    predictions = pd.read_csv(filename)#.iloc[:,:]
    intersection = pd.merge(label_info,predictions,left_on='label idx',right_on = 'venue')
    predictions['venue'] = intersection['label name']
    predictions.to_csv(filename,index=False)

def sparqlToCSV(query,filename):
    s_endpoint = sparqlEndpoint()
    df = s_endpoint.executeSparqlQuery(query)
    df.to_csv(filename,index=False)
    return

def sparqlTodf(query):
    s_endpoint = sparqlEndpoint()
    df = s_endpoint.executeSparqlQuery(query)
    return df

def csvToHTML(filename):
    df = pd.read_csv(filename)
    df = df.to_html()
    df = df.replace('NaN','-')
    return df
# filename = os.path.join('.','test_results.csv')
# sparqlToCSV(query, filename)
