import sys
import os
import Constants
import pandas as pd
from Constants import *
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
    def getKGNodeEdgeTypes(self,namedGraphURI=None,prefix=None):
        predicate_types_query="select distinct ?p \n"
        predicate_types_query+= "" if namedGraphURI is None else  "from <"+namedGraphURI+"> \n"
        predicate_types_query+= " where {?s ?p ?o.} "
        predicate_types_df=self.KG_sparqlEndpoint.executeSparqlquery(predicate_types_query)
        edge_types_lst=predicate_types_df["p"].apply(lambda x:x.replace("\"","")).tolist()
        KG_types_lst=[]
        for edgeType in edge_types_lst:
            if edgeType !="http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                s_type_query="select IF(STRLEN(xsd:string(?s_type))>0,xsd:string(?s_type),\""+edgeType.split("/")[-1]+"_Subj\") as ?s_type count(*) as ?count \n"
                s_type_query += "" if namedGraphURI is None else "from <" + namedGraphURI + "> \n"
                s_type_query += "where { ?s <"+edgeType+"> ?o. \n"
                s_type_query += " OPTIONAL {?s a ?s_type.} } \n group by  ?s_type \n order by desc(count(*))"

                o_type_query = "select IF(STRLEN(xsd:string(?o_type))>0,xsd:string(?o_type),\"" + edgeType.split("/")[-1] + "_Obj\") as ?o_type count(*) as ?count \n"
                o_type_query += "" if namedGraphURI is None else "from <" + namedGraphURI + "> \n"
                o_type_query += "where { ?s <" + edgeType + "> ?o. \n"
                o_type_query += " OPTIONAL {?o a ?o_type.} } \n group by  ?o_type \n order by desc(count(*))"

                s_types_df = self.KG_sparqlEndpoint.executeSparqlquery(s_type_query)
                o_types_df = self.KG_sparqlEndpoint.executeSparqlquery(o_type_query)
                KG_types_lst.append([s_types_df["s_type"].values[0].replace("\"","").split("/")[-1],edgeType.split("/")[-1],o_types_df["o_type"].values[0].replace("\"","").split("/")[-1]])
            else:
                KG_types_lst.append(["entity", edgeType.split("/")[-1],"type"])
        kg_types_df=pd.DataFrame(KG_types_lst)
        kg_types_df.to_csv(Constants.KGNET_Config.datasets_output_path+ (namedGraphURI.split(".")[1] if prefix is None else prefix) +"_Types.csv",header=None, index=None)
        return kg_types_df



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

        if task_var == Constants.GML_Operator_Types.NodeClassification:
            target_rel_uri=query_dict["insertJSONObject"]["GMLTask"]["targetEdge"]
            named_graph_uri=query_dict["insertJSONObject"]["GMLTask"]["namedGraphURI"]
            if TOSG=="d1h1":
                query=[get_NC_d1h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)]
            elif TOSG=="d2h1":
                query=get_NC_d2h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)
            self.KG_sparqlEndpoint.execute_sparql_multithreads(query,output_path,start_offset=0, batch_size=10**5, threads_count=16,rows_count=None)
        if task_var == Constants.GML_Operator_Types.LinkPrediction:
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
        model_model_uri = "kgnet:GMLModel/mid-" + str(int(next_model_id)).zfill(7)
        ds_name="mid-" + str(int(next_model_id)).zfill(7)
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
                "output_root_path": Constants.KGNET_Config.datasets_output_path
            },
            "training":
                {"dataset_name": ds_name,
                 "n_classes": 1000,
                 "root_path":  Constants.KGNET_Config.datasets_output_path,
                 "GNN_Method":query_dict["insertJSONObject"]["GMLTask"]["GNNMethod"],
                 }
        }
        return train_pipeline_dict
    def getTaskUri(self,query_dict):
        task_type = query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1]
        tid= self.KGMeta_Governer_obj.getGMLTaskID(query_dict)
        if tid:
            return "kgnet:GMLTask/tid-"+ str(int(tid)).zfill(7),True
        else:
            next_tid= self.KGMeta_Governer_obj.getNextGMLTaskID()
            return "kgnet:GMLTask/tid-"+ str(int(next_tid)).zfill(7),False
    def UpdateKGMeta(self,query_dict,transform_results_dict,train_results_dict):
        task_uri, task_exist = self.getTaskUri(query_dict)
        print("task_uri=", task_uri)
        next_model_id = self.KGMeta_Governer_obj.getNextGMLModelID()
        model_uri = "kgnet:GMLModel/mid-" + str(int(next_model_id)).zfill(7)
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
        self.sample_Task_Subgraph(query_dict,
                                  train_pipline_json["transformation"]["output_root_path"]+train_pipline_json["transformation"]["dataset_name"]+".tsv",
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
    def __init__(self,KGMeta_Governer_obj ):
        self.KGMeta_Governer_obj = KGMeta_Governer_obj
        self.GML_Query_Type = GML_Query_Types.Inference

if __name__ == '__main__':
    ""
    KG_sparqlEndpoint = sparqlEndpoint(endpointUrl='http://206.12.98.118:8890/sparql/')
    gml_operator=gmlOperator(KG_sparqlEndpoint=KG_sparqlEndpoint)
    df=gml_operator.getKGNodeEdgeTypes(namedGraphURI="http://www.aifb.uni-karlsruhe.de")