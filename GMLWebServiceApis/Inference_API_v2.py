
import sys
import os
kgnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inference_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(kgnet_dir)
sys.path.append(inference_dir)

from Constants import *
from GMLaaS.Inference_pipeline_v2 import wise_inference
from GMLaaS.Inference_pipeline import perform_inference
from fastapi import FastAPI, File,UploadFile,HTTPException,Depends,Form
from uvicorn import run
from typing import List
from pydantic import BaseModel
import pandas as pd
import json
from fastapi.responses import JSONResponse
import traceback

app = FastAPI()
# HOST = '127.0.0.1'
HOST = '0.0.0.0'
# PORT = KGNET_Config.GML_Inference_PORT
PORT = 64648

# class InferenceRequest(BaseModel):
#     model_id : str
#     named_graph_uri : str
#     sparqlEndpointURL : str
#     RDFEngine : str
#     dataQuery : List[str]=None
#     targetNodesQuery : str=None
#     targetNodesList:List[str] =None
#     TOSG_Pattern:str=TOSG_Patterns.d1h1
#     topk: int = 1

# class InferenceRequest(BaseModel):
#     model_id : str
#     named_graph_uri : str
#     sparqlEndpointURL : str
#     RDFEngine : str
#     dataQuery : List[str]=None
#     targetNodesQuery : str=None
#     targetNodesList:List[str] =None
#     TOSG_Pattern:str=TOSG_Patterns.d1h1
#     topk: int = 1

class InferenceRequest(BaseModel):
    model_id: str
    named_graph_uri: str
    sparqlEndpointURL: str = 'http://206.12.98.118:8890/sparql'
    RDFEngine: str = 'OpenlinkVirtuoso'
    dataQuery: List[str] = None
    targetNodesQuery: str = None
    targetNodesList: List[str] = None
    TOSG_Pattern: str = TOSG_Patterns.d1h1
    topk: int = 1

    @classmethod
    def as_form(cls,model_id: str = Form(...),named_graph_uri: str = Form(...),sparqlEndpointURL: str = Form(...),
                RDFEngine: str = Form(...),dataQuery: List[str] = Form(None),targetNodesQuery: str = Form(None),
                targetNodesList: List[str] = Form(None),TOSG_Pattern: str = Form(TOSG_Patterns.d1h1),
                topk: int = Form(1)) :
        return cls(model_id=model_id,named_graph_uri=named_graph_uri,sparqlEndpointURL=sparqlEndpointURL,RDFEngine=RDFEngine,
                   dataQuery=dataQuery,targetNodesQuery=targetNodesQuery,targetNodesList=targetNodesList,TOSG_Pattern=TOSG_Pattern,
                   topk=topk)


@app.post("/fullbatch_wise/mid/{mid}")
async def run_fullbatch_inference(mid:str,
                              inference_request:InferenceRequest = Depends(),
                              file:UploadFile = File(...)
                              ):

    model_id = inference_request.model_id if inference_request.model_id is not None else mid
    named_graph_uri = inference_request.named_graph_uri
    sparqlEndpointURL = inference_request.sparqlEndpointURL
    RDFEngine = inference_request.RDFEngine
    dataQuery = inference_request.dataQuery
    targetNodesQuery = inference_request.targetNodesQuery
    targetNodesList=inference_request.targetNodesList
    topk = inference_request.topk
    TOSG_Pattern=inference_request.TOSG_Pattern
    #print(file.filename)
    targetNodesList = []
    file_targets = file.file.read()
    targets = file_targets.decode().split('\n')
    # print("file_targets=", targets)
    for target in targets:
        if target.strip() and len(target.strip())>0:
            targetNodesList.append(target.strip().replace("\"",""))

    dic_results = wise_inference(model_id = model_id, named_graph_uri = named_graph_uri,#target_rel=target_rel,
                                    dataQuery= dataQuery, sparqlEndpointURL = sparqlEndpointURL,
                                    targetNodesQuery = targetNodesQuery,
                                    topk = topk,RDFEngine=RDFEngine,targetNodesList=targetNodesList,TOSG_Pattern=TOSG_Pattern)
    return dic_results#['y_pred']

@app.post("/wise_inference/mid/{mid}")
async def run_wise_inference(mid:str,inference_request: InferenceRequest):
    model_id = inference_request.model_id if inference_request.model_id is not None else mid
    named_graph_uri = inference_request.named_graph_uri
    sparqlEndpointURL = inference_request.sparqlEndpointURL
    RDFEngine = inference_request.RDFEngine
    dataQuery = inference_request.dataQuery
    targetNodesQuery = inference_request.targetNodesQuery
    targetNodesList=inference_request.targetNodesList
    topk = inference_request.topk
    TOSG_Pattern=inference_request.TOSG_Pattern
    #target_rel = inference_request.target_rel
    dic_results = wise_inference(model_id = model_id, named_graph_uri = named_graph_uri,#target_rel=target_rel,
                                    dataQuery= dataQuery, sparqlEndpointURL = sparqlEndpointURL,
                                    targetNodesQuery = targetNodesQuery,
                                    topk = topk,RDFEngine=RDFEngine,targetNodesList=targetNodesList,TOSG_Pattern=TOSG_Pattern)
    return dic_results#['y_pred']

@app.post("/gml_inference/mid/{mid}")
async def run_inference(mid:str,inference_request: InferenceRequest):
    model_id = inference_request.model_id if inference_request.model_id is not None else mid
    named_graph_uri = inference_request.named_graph_uri
    sparqlEndpointURL = inference_request.sparqlEndpointURL
    RDFEngine = inference_request.RDFEngine
    dataQuery = inference_request.dataQuery
    targetNodesQuery = inference_request.targetNodesQuery
    targetNodesList=inference_request.targetNodesList
    topk = inference_request.topk
    TOSG_Pattern=inference_request.TOSG_Pattern

    dic_results = perform_inference(model_id = model_id, named_graph_uri = named_graph_uri,
                                    dataQuery= dataQuery, sparqlEndpointURL = sparqlEndpointURL,
                                    targetNodesQuery = targetNodesQuery,
                                    topk = topk,RDFEngine=RDFEngine,targetNodesList=targetNodesList,TOSG_Pattern=TOSG_Pattern)
    return dic_results#['y_pred']


if __name__ == "__main__":
    run(app,host=HOST,port=PORT)

