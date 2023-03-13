# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:54:04 2023

@author: walee
"""
import sys
sys.path.append("..")
import sparqlEndpoints.sparqlEndpoint as se
import pandas as pd
# import GMLQueryRewriter.gmlRewriter as qw

import os

def sparqlToCSV(query,filename):
    s_endpoint = se.sparqlEndpoint()
    df = s_endpoint.executeSparqlQuery(query)
    df.to_csv(filename,index=False)
    return

def csvToHTML(filename):
    df = pd.read_csv(filename)
    df = df.to_html()
    return df
# filename = os.path.join('.','test_results.csv')
# sparqlToCSV(query, filename)