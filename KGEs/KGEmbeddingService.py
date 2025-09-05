from .embeddingStore  import FaissInMemoryStore
from .similarityMetrics  import cosineSimilarity
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import time
import logging
from urllib.parse import urlparse,parse_qs
import pickle
import json
import cgi
import requests
import threading
class embeddingAsaService(BaseHTTPRequestHandler):
    embeddingStores={}
    serviceName=""          
    def addEmbeddingStore(self,key,store):
        embeddingStores[key]=store
    def _set_response(self):
        self.send_response(200)
        self.end_headers()
    def do_POST(self):
        print("doPost request Received at Server="+self.serviceName)
        ctype, pdict = cgi.parse_header(self.headers.get('Content-Type'))
        # refuse to receive non-json content
        if ctype != 'application/json':
            self._set_response()    
            self.wfile.write('POST APIs only accepts JSON body'.encode('utf-8'))
            return
        # read the message and convert it into a python dictionary
        content_len = int(self.headers.get('Content-Length'))
#         print("length=",content_len)
        post_body = self.rfile.read(content_len)
#         print('post_body=',post_body)
        data = json.loads(post_body)            
        self._set_response()    
#         print('self.path=',self.path)
##########################read Params ################################3
        if 'emb_technique' in data:
                emb_technique=data['emb_technique']
        else:
                emb_technique="ComplEx" 
        if 'entity' in data:
                entity=data['entity']
        else:
                entity=None
        if 'entity1' in data:
                entity1=data['entity1']
        else:
                entity1=None
        if 'entity2' in data:
                entity2=data['entity2']
        else:
                entity2=None            
        if 'top' in data:
                top=int(data['top'])
        else:
                top=10
        if 'targetFeature' in data:
                targetFeature=data['targetFeature']
        else:
                targetFeature=None
                
        answer=None
        print("path=",self.path," data=",data)
        cs=cosineSimilarity()
#########################################################################
        if self.path.endswith("/getTopSimilarEntitiesFaiss"): 
            if entity !=None:
                l3=self.embeddingStores["faiss"].searchTopSimilarEntites_FlatL2D(emb_technique,entity,cs,top)
                json_obj="{\"api\":\"getTopSimilarCompaniesFaiss\",\"parameters\":{\"entity\":\""+entity+"\",\"top\":\""+str(top) +"\",\"emb_technique\":\""+emb_technique+"\"},\"results\":"+ str(l3).replace("'","\"") +",\"success\":true,\"code\":200}"
                self.wfile.write(json_obj.encode('utf-8'))
            else:
                answer=-2
                print("semantic_affinity=", answer)
                self._set_response()
                self.wfile.write(json.dumps({'semantic_affinity': answer}).encode())
        #####################################################################        
        elif self.path.endswith("/getTopSimilarEntities"):
            if entity !=None:
                l3=self.embeddingStores["pickle"].searchTopSimilarEntites(emb_technique,entity,cs,top)
                json_obj="{\"api\":\"getTopSimilarEntities\",\"parameters\":{\"entity\":\""+entity+"\",\"top\":\""+str(top) +"\",\"emb_technique\":\""+emb_technique+"\"},\"results\":"+ str(l3).replace("'","\"") +",\"success\":true,\"code\":200}"
                self.wfile.write(json_obj.encode('utf-8'))
            else:
                answer=-2
                print("semantic_affinity=", answer)
                self._set_response()
                self.wfile.write(json.dumps({'semantic_affinity': answer}).encode())
        ####################################################################                
        elif self.path.endswith("/getSimilartyScore"): 
            if entity1 !=None and entity2 !=None:
                score=self.embeddingStores["pickle"].getSimilartyScore(emb_technique,entity1,entity2,cs)
                json_obj="{\"api\":\"getSimilartyScore\",\"parameters\":{\"entity1\":\""+entity1+"\",\"entity2\":\""+ entity2+"\",\"top\":\""+str(top) +"\",\"emb_technique\":\""+emb_technique+"\"},\"score\":"+ str(score) +",\"success\":true,\"code\":200}"
                self.wfile.write(json_obj.encode('utf-8'))
            else:
                answer=-2
                print("semantic_affinity=", answer)
                self._set_response()
                self.wfile.write(json.dumps({'semantic_affinity': answer}).encode())
        #######################################################################      
        elif self.path.endswith("/getEmbeddingVector"): 
            if entity !=None :
                emb=self.embeddingStores["pickle"].getEmbeddingVector(emb_technique,entity)
                json_obj="{\"api\":\"getEmbeddingVector\",\"parameters\":{\"entity\":\""+entity+"\",\"top\":\""+str(top) +"\",\"emb_technique\":\""+emb_technique+"\"},\"embeddingVector\":"+str(emb) +",\"success\":true,\"code\":200}"
                self.wfile.write(json_obj.encode('utf-8'))
            else:
                answer=-2
                print("semantic_affinity=", answer)
                self._set_response()
                self.wfile.write(json.dumps({'semantic_affinity': answer}).encode())
        #######################################################################      
        elif self.path.endswith("/getNearOptimalEmbeddingTechnique"): 
            if targetFeature !=None :
                nearOptimalEmbeddingTechnique=self.embeddingStores["pickle"].getEmbeddingVector(emb_technique,targetFeature)
                json_obj="{\"api\":\"getNearOptimalEmbeddingTechnique\",\"parameters\":{\"entity\":\""+entity+"\",\"top\":\""+str(top) +"\",\"emb_technique\":\""+emb_technique+"\"},\"nearOptimalEmbeddingTechnique\":"+nearOptimalEmbeddingTechnique +",\"success\":true,\"code\":200}"
                self.wfile.write(json_obj.encode('utf-8'))
            else:
                answer=-2
                print("semantic_affinity=", answer)
                self._set_response()
                self.wfile.write(json.dumps({'semantic_affinity': answer}).encode())
        else:
                self._set_response()
                self.wfile.write(json.dumps({'method':'None'}).encode())
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass
    
def startHttpServer(server,serviceName,Port=8080,hostName="0.0.0.0"):                                
    webServer = ThreadedHTTPServer((hostName, Port), server)
    print("Embedding Service "+serviceName+" started at http://%s:%s" % (hostName, Port))
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
    
def startHttpServerThread(server,serviceName,Port=8080,hostName="0.0.0.0"):
    daemon = threading.Thread(name=serviceName+'_daemon_server',
                  target=startHttpServer,
                  args=(server,serviceName,Port,hostName))
    daemon.setDaemon(True) # Set as a daemon so it will be killed once the main thread is dead.
    daemon.start()
    
    


