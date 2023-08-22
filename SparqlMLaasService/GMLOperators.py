import sys
import os
import pandas as pd
from Constants import utils as kgnet_utils, GML_Operator_Types,GML_Query_Types,KGNET_Config
from GMLaaS.run_pipeline import run_training_pipeline
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
from SparqlMLaasService.gmlRewriter import gmlQueryParser,gmlQueryRewriter
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_query as get_NC_d1h1_query
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d2h1_query as get_NC_d2h1_query
from SparqlMLaasService.TaskSampler.TOSG_Extraction_LP import write_d2h1_TOSG,get_LP_d1h1_query,get_LP_d2h1_query
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
class gmlOperator():
    def __init__(self,KG_sparqlEndpoint):
        self.KG_sparqlEndpoint = KG_sparqlEndpoint
        self.GML_Query_Type = None
class gmlInsertOperator(gmlOperator):
    def __init__(self,KGMeta_Governer_obj,KG_sparqlEndpoint):
        self.KGMeta_Governer_obj = KGMeta_Governer_obj
        self.KG_sparqlEndpoint = KG_sparqlEndpoint
        self.GML_Query_Type = GML_Query_Types.Insert
    def sample_Task_Subgraph(self, query_dict,output_path,TOSG="d1h1"):
        task_var = query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1]
        tragetNode_filter_statments=None
        if  "targetNodeFilters" in query_dict["insertJSONObject"]["GMLTask"]:
            for filter in query_dict["insertJSONObject"]["GMLTask"]["targetNodeFilters"]:
                tragetNode_filter_statments=""
                filter_vals=query_dict["insertJSONObject"]["GMLTask"]["targetNodeFilters"][filter]
                tragetNode_filter_statments+="?s" + filter_vals[0]+" "+ filter_vals[1]+" .\n"
                for idx in range(2,len(filter_vals)):
                    tragetNode_filter_statments+=filter_vals[idx]+".\n"

        if task_var == GML_Operator_Types.NodeClassification:
            target_rel_uri=query_dict["insertJSONObject"]["GMLTask"]["targetEdge"]
            named_graph_uri=query_dict["insertJSONObject"]["GMLTask"]["namedGraphURI"]
            if TOSG=="d1h1":
                query=[get_NC_d1h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)]
            elif TOSG=="d2h1":
                query=get_NC_d2h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)
            self.KG_sparqlEndpoint.execute_sparql_multithreads(query,output_path,start_offset=0, batch_size=10**5, threads_count=16,rows_count=None)
        if task_var == GML_Operator_Types.LinkPrediction:
            target_rel_uri=query_dict["insertJSONObject"]["GMLTask"]["targetEdge"]
            named_graph_uri=query_dict["insertJSONObject"]["GMLTask"]["namedGraphURI"]
            if TOSG=="d1h1":
                query=[get_LP_d1h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)]
            elif TOSG=="d2h1":
                query=get_LP_d2h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)
            self.KG_sparqlEndpoint.execute_sparql_multithreads(query,output_path,start_offset=0, batch_size=10**5, threads_count=16,rows_count=None)
    def create_train_pipline_json(self,query_dict):
        task_uri, task_exist = self.getTaskUri(query_dict)
        next_model_id = self.KGMeta_Governer_obj.getNextGMLModelID()
        model_model_uri = "kgnet:GMLModel/mid-" + kgnet_utils.getIdWithPaddingZeros(next_model_id)
        ds_name="mid-" + kgnet_utils.getIdWithPaddingZeros(next_model_id)
        train_pipeline_dict={
            "transformation": {
                "operatorType": query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1],
                "target_rel": query_dict["insertJSONObject"]["GMLTask"]["targetEdge"],
                "dataset_name": ds_name,
                "dataset_name_csv": ds_name,
                "dataset_types":query_dict["insertJSONObject"]["GMLTask"]["datasetTypesFilePath"],
                "test_size": 0.1,
                "valid_size": 0.1,
                "MINIMUM_INSTANCE_THRESHOLD": 6,
                "output_root_path": KGNET_Config.datasets_output_path
            },
            "training":
                {"dataset_name": ds_name,
                 "n_classes": 1000,
                 "root_path":  KGNET_Config.datasets_output_path,
                 "GNN_Method":query_dict["insertJSONObject"]["GMLTask"]["GNNMethod"],
                 }
        }
        return train_pipeline_dict

    def getTaskUri(self,query_dict):
        task_type = query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1]
        tid= self.KGMeta_Governer_obj.getGMLTaskID(query_dict)
        if tid:
            return "kgnet:GMLTask/tid-"+kgnet_utils.getIdWithPaddingZeros(tid),True
        else:
            next_tid= self.KGMeta_Governer_obj.getNextGMLTaskID()
            return "kgnet:GMLTask/tid-"+ kgnet_utils.getIdWithPaddingZeros(next_tid),False
    def UpdateKGMeta(self,query_dict,transform_results_dict,train_results_dict):
        task_uri, task_exist = self.getTaskUri(query_dict)
        print("task_uri=", task_uri)
        next_model_id = self.KGMeta_Governer_obj.getNextGMLModelID()
        model_uri = "kgnet:GMLModel/mid-" + kgnet_utils.getIdWithPaddingZeros(next_model_id)
        print("model_uri=",model_uri)
        res=self.KGMeta_Governer_obj.insertGMLModel(query_dict,task_uri,task_exist,next_model_id,model_uri,transform_results_dict,train_results_dict)
        result_dict={}
        result_dict["task_uri"]=task_uri
        result_dict["task_exist"] = task_exist
        result_dict["model_uri"] = model_uri
        result_dict["insetred_triples_msg"] = res
        return result_dict
    def executeQuery(self,query_dict):
        train_pipline_json = self.create_train_pipline_json(query_dict)
        print("################# TOSG Sampling ###########################")
        KG_PrimePath=train_pipline_json["transformation"]["output_root_path"]+train_pipline_json["transformation"]["dataset_name"]+".tsv"
        print("KG' path=",KG_PrimePath)
        self.sample_Task_Subgraph(query_dict,KG_PrimePath,
                                  query_dict["insertJSONObject"]["GMLTask"]["TOSG"])
        print("################# Start GNN Task Training  ###########################")
        transform_results_dict,train_results_dict=run_training_pipeline(json_args=train_pipline_json)
        res=self.UpdateKGMeta(query_dict,transform_results_dict,train_results_dict)
        return res,transform_results_dict,train_results_dict

class gmlDeleteOperator(gmlOperator):
    def __init__(self,KGMeta_Governer_obj ):
        self.KGMeta_Governer_obj = KGMeta_Governer_obj
        self.GML_Query_Type=GML_Query_Types.Delete

class gmlInferenceOperator(gmlOperator):
    def __init__(self,KGMeta_Governer_obj,KG_sparqlEndpoint):
        self.KGMeta_Governer_obj = KGMeta_Governer_obj
        self.KG_sparqlEndpoint = KG_sparqlEndpoint
        self.GML_Query_Type = GML_Query_Types.Inference
    def executeQuery(self, query):
        gmlqp = gmlQueryParser(query)
        dataInferQ,dataQ,tragetNodesq, kmetaq,model_id = gmlQueryRewriter(gmlqp.extractQueryStatmentsDict(), self.KGMeta_Governer_obj).rewrite_gml_query()
        # print("KGMeta task select query= \n",kmetaq)
        # print("SPARQL candidate query form 2= \n",dataInferQ)
        # print("SPARQLdata only Query=\n", dataQ)
        return dataInferQ,dataQ,tragetNodesq,kmetaq,model_id

if __name__ == '__main__':
    ""
    KG_sparqlEndpoint = sparqlEndpoint(endpointUrl='http://206.12.98.118:8890/sparql/')
    gml_operator=gmlOperator(KG_sparqlEndpoint=KG_sparqlEndpoint)
    df=gml_operator.getKGNodeEdgeTypes(namedGraphURI="http://www.aifb.uni-karlsruhe.de")