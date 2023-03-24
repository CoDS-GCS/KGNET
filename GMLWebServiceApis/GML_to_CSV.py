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

def mapVenues(filename,labels_path=r'./data/labelidx2labelname.csv'):
    label_info = pd.read_csv(labels_path)
    predictions = pd.read_csv(filename)#.iloc[:,:]
    intersection = pd.merge(label_info,predictions,left_on='label idx',right_on = 'venue')
    predictions['venue'] = intersection['label name']
    predictions.to_csv(filename,index=False)

def sparqlToCSV(query,filename):
    s_endpoint = se.sparqlEndpoint()
    df = s_endpoint.executeSparqlQuery(query)
    df.to_csv(filename,index=False)
    return

def csvToHTML(filename):
    df = pd.read_csv(filename)
    df = df.to_html()
    df = df.replace('NaN','-')
    return df
# filename = os.path.join('.','test_results.csv')
# sparqlToCSV(query, filename)