
import sys
import os
kgnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inference_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(kgnet_dir)
sys.path.append(inference_dir)

from Constants import *
from GMLaaS.Inference_pipeline import perform_inference
from fastapi import FastAPI
from uvicorn import run
from typing import List
from pydantic import BaseModel
import traceback

app = FastAPI()
HOST = '127.0.0.1'
# HOST = '0.0.0.0'
PORT = KGNET_Config.GML_Inference_PORT

class InferenceRequest(BaseModel):
    model_id : int
    named_graph_uri : str
    sparqlEndpointURL : str
    dataQuery : List[str]
    targetNodesQuery : str
    topk: int = 1


@app.post("/gml_inference/mid/{mid}")
async def run_inference(mid:int,inference_request: InferenceRequest):
    model_id = inference_request.model_id if inference_request.model_id is not None else mid
    named_graph_uri = inference_request.named_graph_uri
    sparqlEndpointURL = inference_request.sparqlEndpointURL
    dataQuery = inference_request.dataQuery
    targetNodesQuery = inference_request.targetNodesQuery
    # try:
    #     dic_results = perform_inference(model_id = model_id,named_graph_uri = named_graph_uri,
    #                   targetNode_filter_statements = dataQuery,
    #                   sparqlEndpointURL = sparqlEndpointURL)
    # except  Exception as e :
    #     print (e)
    #     print(traceback.print_stack())
    #     return {"Exception":str(e)}

    dic_results = perform_inference(model_id = model_id, named_graph_uri = named_graph_uri,
                                    dataQuery= dataQuery, sparqlEndpointURL = sparqlEndpointURL,
                                    targetNodesQuery = targetNodesQuery)
    return dic_results#['y_pred']
    # return {
    #     "model_id": model_id,
    #     "named_graph_uri": named_graph_uri,
    #     "target_rel_uri": sparqlEndpointURL,
    #     "targetNode_filter_statements": dataQuery
    # }


if __name__ == "__main__":
    run(app,host=HOST,port=PORT)

