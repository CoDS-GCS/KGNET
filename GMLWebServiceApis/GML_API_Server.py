# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:13:15 2023

@author: walee
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import pandas as pd
import json
import os

PATH_CSV = os.path.join('.','data','DBLP_Paper_Venue_FM_Literals2Nodes_SY1900_EY2021_50Class_GA_0_GSAINT_50_run2_output.csv')
HOSTNAME = '127.0.0.1'
PORT = 64646
PAGE = 1
PAGE_SIZE = 100
data  = pd.read_csv(PATH_CSV)

def gen_keyVal(y_pred,key,value):
    """ Takes pandas dataframe as input and returns a dictionary where 'ent name'
        is the key and 'y_pred' is the value."""
        
    return y_pred.set_index(key)[value].to_dict()

def single_pred (uri):
    """ Takes a single uri and searches its y_prediction from the provided CSV"""
    # data = pd.read_csv(PATH_CSV)
    y_pred = data [data['ent name'] == uri][['y_pred','ent name']]
    return gen_keyVal(y_pred)

def multi_pred (list_uri,page=PAGE,size=PAGE_SIZE):
    """ Takes a list of uris and searches their y_predictions from the provided CSV.
        page and size are optional parameters to limit the size of returned items"""
    # data = pd.read_csv(PATH_CSV)
    PATH_CSV = os.path.join('.','data','DBLP_Paper_Venue_FM_Literals2Nodes_SY1900_EY2021_50Class_GA_0_GSAINT_50_run2_output.csv')
    data = pd.read_csv(PATH_CSV)
    start = (page -1) * size
    end = start + size
    lis_y_pred = data[data['ent name'].isin(list_uri)][['y_pred','ent name']][start:end]#.values
    return gen_keyVal(lis_y_pred)

def all_pred (page=PAGE,size=PAGE_SIZE):
    """ Returns all the y_predictions from the csv. Page and size are optional parameters to 
        limit the size of returned items"""
    # data = pd.read_csv(PATH_CSV)
    start = (page - 1) * size
    end = start + size
    all_y_pred = data[['y_pred','ent name']][start:end]#.values
    return gen_keyVal(y_pred=all_y_pred,key='ent name',value='y_pred')

def DBLP_AF (page=PAGE,size=PAGE_SIZE):
    """ For DBLP Author Affiliation prediction """
    #TODO Page implementation
    PATH_CSV = os.path.join('.','data','AuthorsPrimaryAffaliations_LP.csv')
    # PATH_LABELS = os.path.join('.','data','AuthorsPrimaryAffaliations_LP.csv')
    data = pd.read_csv(PATH_CSV)
    # label_info = pd.read_csv(PATH_LABELS)
    # label_info['label_idx'] = label_info['label_idx'].astype('str')
    # data['author'] = data['author'].astype('str')
    
    # intersection = pd.merge(label_info,data,left_on='label_idx',right_on = 'author')
    # data['author']

    dblp_af_pred = data[['author','affiliation']]
    return gen_keyVal(y_pred=dblp_af_pred,key='author',value='affiliation')

def MAG_PV (page=PAGE,size=PAGE_SIZE):
    """ For MAG Paper Venue prediction """
    PATH_CSV = os.path.join('.','data','mag_papers.csv')
    data = pd.read_csv(PATH_CSV)
    mag_pv_pred = data[['paper','y_pred']]
    return gen_keyVal(y_pred=mag_pv_pred,key='paper',value='y_pred')

def IEEECIS (page=PAGE,size=PAGE_SIZE):
    """ For IEEE Fraud Transaction prediction """
    PATH_CSV = os.path.join('.','data','CIS_Transactions2.csv')
    data = pd.read_csv(PATH_CSV)
    ieeecis_pred = data[['Transaction','ypred']]
    return gen_keyVal(y_pred=ieeecis_pred,key='Transaction',value='ypred')

def YAGO (page=PAGE,size=PAGE_SIZE):
    """ For IEEE Fraud Transaction prediction """
    PATH_CSV = os.path.join('.','data','yago3-10_CA.csv')
    data = pd.read_csv(PATH_CSV)
    dblp_pv_pred = data[['Source_Node','Edge_type','Destination_Node']]
    return gen_keyVal(y_pred=dblp_pv_pred,key='',value='')

class GML_Server(BaseHTTPRequestHandler):
    
    def _set_response(self):
        self.send_response(200)
        self.end_headers()
    
    def _send_JSONresponse(self,y_pred):
        self._set_response()
        self.wfile.write(json.dumps(y_pred).encode('utf-8'))  
               
    def do_POST(self):
        parsed_url = urlparse(self.path)
        endpoint = parsed_url.path[1:]
        if self.headers['Content-Length'] is not None:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode()
            print(post_data)
            data = json.loads(post_data)
            papers = data.get("paper")
            page = int(data.get("page",PAGE))
            size = int(data.get("size",PAGE_SIZE))
        
        if endpoint == 'single':
            # print (f'input = {papers}')
            y_pred = single_pred(papers,)
            self._send_JSONresponse(y_pred)
        
        elif endpoint == 'multi':       
            # print(f'page = {page} size = {size}')
            y_pred = multi_pred(papers,page=page,size=size)
            self._send_JSONresponse(y_pred)
         
        elif endpoint == 'all':
            y_pred = all_pred(page=page,size=size)
            self._send_JSONresponse(y_pred)
        
        elif endpoint == 'DBLP_AF':
            y_pred = DBLP_AF(page=page,size=size)
            self._send_JSONresponse(y_pred)
    
        elif endpoint == 'MAG_PV':
            y_pred = MAG_PV(page=page,size=size)
            self._send_JSONresponse(y_pred)
            
        elif endpoint == 'IEEECIS':
            y_pred = IEEECIS(page=page,size=size)
            self._send_JSONresponse(y_pred)


if __name__ == "__main__":        
    webServer = HTTPServer((HOSTNAME, PORT), GML_Server)
    print("Server started http://%s:%s" % (HOSTNAME, PORT))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

        