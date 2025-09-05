
import sys
import os
kgnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inference_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(kgnet_dir)
sys.path.append(inference_dir)

from Constants import *
from GMLaaS.Inference_pipeline_v2 import wise_inference
from GMLaaS.Inference_pipeline import perform_inference
from GMLaaS.models.EXPERIMENTS_wise_ssaint import inference_API
from GMLaaS.models.graph_saint_Shadow_FAISS import inference_baseline_API
from fastapi import FastAPI, File,UploadFile,HTTPException,Depends,Form,Request
from uvicorn import run
from typing import List
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime as dt
from resource import *
import asyncio
from fastapi.staticfiles import StaticFiles

import pandas as pd
import json
from fastapi.responses import JSONResponse
import traceback
from functools import partial
from codecarbon import EmissionsTracker
app = FastAPI()
# HOST = '127.0.0.1'
HOST = '0.0.0.0'
# PORT = KGNET_Config.GML_Inference_PORT
PORT = 7770
# app.mount("/static", StaticFiles(directory="static"), name="static")
global LOADED_MODEL_NAME, LOADED_MODEL_OBJ
LOADED_MODEL_NAME = None
LOADED_MODEL_OBJ = None
# Create a global process pool
process_pool = ProcessPoolExecutor()


class InferenceRequest(BaseModel):
    model_id: str = ''
    dataset_name: str
    num_workers: int = 4
    named_graph_uri: str = ''
    sparqlEndpointURL: str = 'http://206.12.98.118:8890/sparql'
    RDFEngine: str = 'OpenlinkVirtuoso'
    dataQuery: List[str] = None
    targetNodesQuery: List[str] = None
    targetNodesList: List[str] = None
    TOSG_Pattern: str = TOSG_Patterns.d1h1
    topk: int = 1

    @classmethod
    def as_form(cls,model_id: str = Form(...),dataset_name = Form(...),num_workers:int = Form(...),named_graph_uri: str = Form(...),sparqlEndpointURL: str = Form(...),
                RDFEngine: str = Form(...),dataQuery: List[str] = Form(None),targetNodesQuery:List[str] = Form(None),
                targetNodesList: List[str] = Form(None),TOSG_Pattern: str = Form(TOSG_Patterns.d1h1),
                topk: int = Form(1)) :
        return cls(model_id=model_id,dataset_name=dataset_name,num_workers=num_workers,named_graph_uri=named_graph_uri,sparqlEndpointURL=sparqlEndpointURL,RDFEngine=RDFEngine,
                   dataQuery=dataQuery,targetNodesQuery=targetNodesQuery,targetNodesList=targetNodesList,TOSG_Pattern=TOSG_Pattern,
                   topk=topk)


def split_list(target_list, num_chunks):
    chunk_size = len(target_list) // num_chunks
    remainder = len(target_list) % num_chunks
    tasks = []
    start_idx = 0
    for i in range(num_chunks):
        # If there's a remainder, add an extra element to the current chunk
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        tasks.append(target_list[start_idx:end_idx])
        start_idx = end_idx

    print(f'\t****** NUMBER OF TASKS = {len(tasks)}\n\n\t****** NUMBER OF WORKERS = {num_chunks}')
    return tasks

@app.post("/fullbatch_wise/mid/{mid}")  ### KG-WISE PARALLEL Full BATCH
async def run_fullbatch_inference(mid: str,
                                  inference_request: InferenceRequest = Depends(),
                                  file: UploadFile = File(...)
                                  ):
    model_id = inference_request.model_id if inference_request.model_id is not None else mid
    dataset_name = inference_request.dataset_name
    num_workers = 1#inference_request.num_workers
    targetNodesList = []
    file_targets = file.file.read()
    targets = file_targets.decode().split('\n')
    time_start = dt.now()
    for target in targets:
        if target.strip() and len(target.strip()) > 0:
            targetNodesList.append(target.strip().replace("\"", ""))
    # target_chunks = split_list(targetNodesList, num_workers)
    results = []
    # inference_API(dataset_name, targetNodesList, model_id)

    ### Track emissions ###
    # tracker = EmissionsTracker()#measure_power_secs=5
    # tracker.start()
    loop = asyncio.get_event_loop()
    partial_wise_inference = partial(inference_API, dataset_name=dataset_name, targetNodesList=targetNodesList,
                                model_id=model_id,)
    dic_results = await loop.run_in_executor(process_pool,partial_wise_inference,)
    emissions: float = tracker.stop()

    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(
    #         inference_API, dataset_name, chunk, model_id
    #     ) for chunk in target_chunks]
    # for future in futures:
    #     print('result = ', future)
    #     result = future.result()
    #     results.append(result)
    time_end = (dt.now() - time_start).total_seconds()
    max_mem = (getrusage(RUSAGE_SELF).ru_maxrss) / (1024 * 1024)
    # dic_results = {}
    total_mem = dic_results['maxMem']
    acc = dic_results['accuracy']


    # final_acc = (sum(acc) / len(acc))

    dic_results['Total time'] = time_end
    dic_results['Max Memory Usage'] = total_mem
    dic_results['accuracy'] = acc#final_acc
    dic_results['num workers'] = num_workers
    dic_results['Emissions'] = emissions
    print(
        f'\n\nDataset Name = {dataset_name}\n\nTOTAL TIME = {time_end}\n\nMax Memory usage = {total_mem} GB\n\nAccuracy = {acc}\n\nNum target = {len(targetNodesList)}\n\nEmissions = {emissions}\n\n*********')
    return dic_results  # ['y_pred']



async def baseline_inference_wrapper(dataset_name,target_nodes,model_id,return_model):
    loop = asyncio.get_event_loop()
    partial_inference = partial(inference_baseline_API,dataset_name=dataset_name,targetNodesList=target_nodes,model_id=model_id,return_Model=return_model)
    return await loop.run_in_executor(process_pool,
                                      partial_inference)


@app.post("/full_batch_BASELINE/mid/{mid}")
async def run_fullbatch_Baseline_inference(mid: str,
                                  inference_request: InferenceRequest = Depends(),
                                  file: UploadFile = File(...)
                                  ):
    model_id = inference_request.model_id if inference_request.model_id is not None else mid
    dataset_name = inference_request.dataset_name
    num_workers = inference_request.num_workers
    targetNodesList = []
    file_targets = file.file.read()
    targets = file_targets.decode().split('\n')
    time_start = dt.now()
    for target in targets:
        if target.strip() and len(target.strip()) > 0:
            targetNodesList.append(target.strip().replace("\"", ""))
    target_chunks = split_list(targetNodesList, num_workers)
    results = []
    # For Parallel Execution
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(
    #         inference_baseline_API, dataset_name, chunk, model_id
    #     ) for chunk in target_chunks]
    # for future in futures:
    #     print('result = ', future)
    #     result = future.result()
    #     results.append(result)

    ### Track Emissions ###
    # tracker = EmissionsTracker() #measure_power_secs=5
    # tracker.start()
    global LOADED_MODEL_NAME, LOADED_MODEL_OBJ
    if (LOADED_MODEL_NAME is None) or (LOADED_MODEL_NAME != dataset_name) :
        #dic_results,LOADED_MODEL_OBJ = inference_baseline_API(dataset_name=dataset_name,targetNodesList=targetNodesList,model_id=model_id,return_Model=True)
        dic_results, LOADED_MODEL_OBJ = await baseline_inference_wrapper(dataset_name=dataset_name,
                                                               target_nodes=targetNodesList, model_id=model_id,
                                                               return_model=True)
        LOADED_MODEL_NAME = dataset_name
    else: # <--- For caching the model
        dic_results = inference_baseline_API(dataset_name=dataset_name,
                                                               targetNodesList=targetNodesList, model_id=model_id,
                                                               return_Model=False,model_obj=LOADED_MODEL_OBJ)
    emissions:float = tracker.stop()

    time_end = (dt.now() - time_start).total_seconds()
    max_mem = (getrusage(RUSAGE_SELF).ru_maxrss) / (1024 * 1024)
    print(f'--> server side calculated memory = {max_mem}')
    total_mem = dic_results['maxMem']
    acc = dic_results['accuracy']
    # for x in dic_results:
    #     total_mem += x['maxMem']
    #     acc.append(x['accuracy'])

    # final_acc = (sum(acc) / len(acc))

    dic_results['Total time'] = time_end
    dic_results['Max Memory Usage'] = total_mem
    dic_results['Emissions'] = emissions
    # dic_results['accuracy'] = final_acc
    # dic_results['num workers'] = num_workers
    print(
        f'\n\nDataset = {dataset_name} \n\nTOTAL TIME = {time_end}\n\nMax Memory usage = {total_mem} GB\n\nAccuracy = {acc}\n\nEmissions = {emissions}\n\n*********')
    return dic_results  # ['y_pred']

#
# @app.post("/gml_inference/mid/{mid}")
# async def run_inference(mid:str,inference_request: InferenceRequest):
#     model_id = inference_request.model_id if inference_request.model_id is not None else mid
#     named_graph_uri = inference_request.named_graph_uri
#     sparqlEndpointURL = inference_request.sparqlEndpointURL
#     RDFEngine = inference_request.RDFEngine
#     dataQuery = inference_request.dataQuery
#     targetNodesQuery = inference_request.targetNodesQuery
#     targetNodesList=inference_request.targetNodesList
#     topk = inference_request.topk
#     TOSG_Pattern=inference_request.TOSG_Pattern
#
#     dic_results = wise_SHsaint(dataset_name=dataset_name,targetNodesList = targetNodesList,target_rel_uri=target_rel_uri,
#                                ds_types='mag', graph_uri=graph_uri,)
#     return dic_results#['y_pred']
#
#
# # def inference_API(dataset_name, model_id, num_workers):
# #     return {"dataset": dataset_name, "model_id": model_id, "num_workers": num_workers}
#
# @app.post("/submit-inference")
# async def handle_inference(request: Request):
#     form_data = await request.form()
#
#     # Extract number of instances dynamically based on form keys
#     num_instances = len([key for key in form_data if key.startswith("dataset_name_")])
#
#     inference_requests = []
#     for i in range(num_instances):
#         dataset_name = form_data[f"dataset_name_{i}"]
#         model_id = form_data.get(f"model_id_{i}", "")
#         num_workers = int(form_data.get(f"num_workers_{i}", 4))
#
#         inference_requests.append(InferenceRequest(
#             model_id=model_id,
#             dataset_name=dataset_name,
#             num_workers=num_workers
#         ))
#
#     # Run all inferences independently in parallel
#     results = []
#     with ProcessPoolExecutor() as executor:
#         futures = [executor.submit(inference_API, req.dataset_name, req.model_id, req.num_workers) for req in inference_requests]
#         for future in futures:
#             results.append(future.result())
#
#     return {"results": results}
if __name__ == "__main__":
    run(app,host=HOST,port=PORT)

