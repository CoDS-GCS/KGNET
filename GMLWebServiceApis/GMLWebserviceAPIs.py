# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import logging
from urllib.parse import urlparse,parse_qs
import pickle
import json
import requests
from scipy import spatial
import pandas as np
# !pip install validators
import validators
from sklearn.metrics.pairwise import cosine_similarity

class kgnet_embedding: 
    def __init__(self):
        self.kgnet_Companies_emb_ComplEx_df=pickle.load( open( "kgnet_Companies_emb_ComplEx_df.pickle", "rb" ))
        self.kgnet_Companies_emb_ConvE_df=pickle.load( open( "kgnet_Companies_emb_ConvE_df.pickle", "rb" ))
        self.kgnet_Companies_emb_ConvKB_df=pickle.load( open( "kgnet_Companies_emb_ConvKB_df.pickle", "rb" ))
        self.kgnet_Companies_emb_DistMult_df=pickle.load( open( "kgnet_Companies_emb_DistMult_df.pickle", "rb" ))
        self.kgnet_Companies_emb_HolE_df=pickle.load( open( "kgnet_Companies_emb_HolE_df.pickle", "rb" ))
        self.kgnet_Companies_emb_Rdf2vec_df=pickle.load( open( "kgnet_Companies_emb_Rdf2vec_df.pickle", "rb" ))
        self.kgnet_Companies_emb_TransE_df=pickle.load( open( "kgnet_Companies_emb_TransE_df.pickle", "rb" ))
        self.emd_df=self.kgnet_Companies_emb_DistMult_df
        self.emd_df["entity"]=self.emd_df["entity"].str.lower()
#         self.emd_df_kgnet_rdf2vec = np.DataFrame (self.kgnet_rdf2vec_faces_emb,columns=['company','Market_Value_class','emb'])
        print("kgnet emb count=",len(self.emd_df))        
        print('Done loading')
        
    def set_emb_model(self,emb_model="DistMult"):
            if emb_model.lower() =="DistMult".lower():
                self.emd_df=self.kgnet_Companies_emb_DistMult_df
            elif emb_model.lower() =="ConvE".lower():
                self.emd_df=self.kgnet_Companies_emb_ConvE_df
            elif emb_model.lower() =="ConvKB".lower():
                self.emd_df=self.kgnet_Companies_emb_ConvKB_df
            elif emb_model.lower() =="ComplEx".lower():
                self.emd_df=self.kgnet_Companies_emb_ComplEx_df
            elif emb_model.lower() =="HolE".lower():
                self.emd_df=self.kgnet_Companies_emb_HolE_df     
            elif emb_model.lower() =="Rdf2vec".lower():
                self.emd_df=self.kgnet_Companies_emb_Rdf2vec_df   
            elif emb_model.lower() =="TransE".lower():
                self.emd_df=self.kgnet_Companies_emb_TransE_df   
            else:
                emb_model="DistMult"
                self.emd_df=self.kgnet_Companies_emb_DistMult_df
                
            self.emd_df["entity"]=self.emd_df["entity"].str.lower()    
            print("emb_model=",emb_model)
            
    def get_Semantic_Affinity(self,str_company1, str_company2):    
        emb1=None
        emb2=None   
        emb1=self.emd_df[self.emd_df["entity"]==str_company1.lower()]["emb"]
#         print("emb1",type(emb1),emb1)
        if len(emb1)>0:    
            emb1=emb1.tolist()
        else:
            emb1=None

        emb2=self.emd_df[self.emd_df["entity"]==str_company2.lower()]["emb"]
#         print("emb2",type(emb2),emb2)
        if len(emb2)>0:    
            emb2=emb2.tolist()
        else:
            emb2=None


        if emb1 is None :
            print("emb1 is None")
            return -2
        if emb2 is None :
            print("emb2 is None")
            return -2
#         print("emb1=",emb1)

        Max_Sim=-2    
        for emb1_ins in emb1:
            for emb2_ins in emb2:        
                result=1 - spatial.distance.cosine(emb1_ins, emb2_ins)
                if result>Max_Sim:
                    Max_Sim=result


        return Max_Sim


obj_kgnet_embedding=kgnet_embedding()
hostName = "0.0.0.0"
serverPort = 6101

class MyServer(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
#         self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):               
        logging.info("GET request to kgnet Embedding Server,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))        
        o = urlparse(self.path)
        qs_dict=parse_qs(o.query)
        answer="NONE"
        ##############################################
        print(qs_dict)
        if not qs_dict.get("comp1"):
            comp1="NULL"
        else:
            comp1 = qs_dict.get("comp1")[0]
        print("comp1=",comp1)
        
        if not qs_dict.get("comp2"):
            comp2="NULL"
        else:
            comp2 = qs_dict.get("comp2")[0]            
        print("comp2=",comp2)
        
        if not qs_dict.get("emb_model"):
            emb_model="NULL"
        else:
            emb_model = qs_dict.get("emb_model")[0]              
#         print("emb_model=",emb_model)        
        
        obj_kgnet_embedding.set_emb_model(emb_model)
        if comp2!="NULL" and comp1 !="NULL":
            answer = obj_kgnet_embedding.get_Semantic_Affinity(comp1, comp2)
        else:
            answer=-2
            
        print("semantic_affinity=", answer)
        self._set_response()
        self.wfile.write(json.dumps({'semantic_affinity': answer}).encode())

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
