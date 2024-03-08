import Constants
from Constants import utils as kgnet_utils, GML_Operator_Types,GML_Query_Types,KGNET_Config
from GMLaaS.run_pipeline import run_training_pipeline
from SparqlMLaasService.gmlRewriter import gmlQueryParser,gmlQueryRewriter
from SparqlMLaasService.QueryFormatter import gmlQueryFormatter
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_KG_entity_types as get_KG_entity_types
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_query as get_NC_d1h1_query
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d2h1_query as get_NC_d2h1_query
from SparqlMLaasService.TaskSampler.TOSG_Extraction_LP import get_LP_d1h1_query,get_LP_d2h1_query
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from datetime import datetime
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
        tragetNode_filter_statments=[]
        if  "targetNodeFilters" in query_dict["insertJSONObject"]["GMLTask"]:
            for filter in query_dict["insertJSONObject"]["GMLTask"]["targetNodeFilters"]:
                filter_vals=query_dict["insertJSONObject"]["GMLTask"]["targetNodeFilters"][filter]
                tragetNode_filter_statments.append("?s " + filter_vals[0]+" "+ filter_vals[1]+" .")
                for idx in range(2,len(filter_vals)):
                    tragetNode_filter_statments.append(filter_vals[idx]+".")

        if task_var == GML_Operator_Types.NodeClassification:
            stype=query_dict["insertJSONObject"]["GMLTask"]["targetNode"]
            otype = query_dict["insertJSONObject"]["GMLTask"]["labelNode"] if "labelNode" in query_dict["insertJSONObject"]["GMLTask"].keys() else None
            target_rel_uri = query_dict["insertJSONObject"]["GMLTask"]["targetEdge"]
            named_graph_uri=query_dict["insertJSONObject"]["GMLTask"]["namedGraphURI"]
            types_query=get_KG_entity_types(graph_uri=named_graph_uri)
            if TOSG=="d1h1":
                query_spo,query_o_types=get_NC_d1h1_query(graph_uri=named_graph_uri, target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments, stype=stype, otype=otype,prefixs=query_dict['prefixes'])
                query=[query_spo,query_o_types]
            elif TOSG=="d2h1":
                query=get_NC_d2h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments,stype=stype,otype=otype)
            # query.append(types_query)
            self.KG_sparqlEndpoint.execute_sparql_multithreads(query,output_path,start_offset=0, batch_size=10**5, threads_count=16,rows_count=None)
        if task_var == GML_Operator_Types.LinkPrediction:
            target_rel_uri=query_dict["insertJSONObject"]["GMLTask"]["targetEdge"]
            named_graph_uri=query_dict["insertJSONObject"]["GMLTask"]["namedGraphURI"]
            if TOSG=="d1h1":
                query=[get_LP_d1h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)]
            elif TOSG=="d2h1":
                query=get_LP_d2h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)
            self.KG_sparqlEndpoint.execute_sparql_multithreads(query,output_path,start_offset=0, batch_size=10**5, threads_count=16,rows_count=None)
    def get_next_model_id(self,query_dict,mode='HASH'):
        if mode == 'HASH':
            # next_model_id = "Model->" + query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1] + "->" + \
            #                 query_dict["insertJSONObject"]["GMLTask"]["namedGraphURI"] + "->" + \
            #                 query_dict["insertJSONObject"]["GMLTask"]["targetEdge"] + "->" + \
            #                 query_dict["insertJSONObject"]["GMLTask"]["GNNMethod"] + "->" +\
            #                 str(datetime.datetime.now().timestamp())
            next_model_id = str(datetime.now().timestamp())
            next_model_id = kgnet_utils.get_sha256(next_model_id)
        else:
            next_model_id = self.KGMeta_Governer_obj.getNextGMLModelID()
            next_model_id = kgnet_utils.getIdWithPaddingZeros(next_model_id)
        return next_model_id
    def get_next_task_id(self,query_dict,mode='HASH'):
        if mode=='HASH':
            # next_task_id = "Model->" + query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1] + "->" + \
            #                 query_dict["insertJSONObject"]["GMLTask"]["namedGraphURI"] + "->" + \
            #                 query_dict["insertJSONObject"]["GMLTask"]["targetEdge"]+"->"+ \
            #                str(datetime.datetime.now().timestamp())
            next_task_id = str(datetime.now().timestamp())
            next_task_id = kgnet_utils.get_sha256(next_task_id)
        else:
            next_task_id = self.KGMeta_Governer_obj.getNextGMLTaskID()
            next_task_id = kgnet_utils.getIdWithPaddingZeros(next_task_id)
        return next_task_id
    def create_train_pipline_json(self,query_dict):
        task_uri, task_exist = self.getTaskUri(query_dict)
        ds_name="mid-"+self.get_next_model_id(query_dict)
        # model_model_uri = "kgnet:GMLModel/mid-" + kgnet_utils.getIdWithPaddingZeros(next_model_id)
        # ds_name="mid-" + kgnet_utils.getIdWithPaddingZeros(next_model_id)
        train_pipeline_dict={
            "transformation": {
                "operatorType": query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1],
                "target_rel": query_dict["insertJSONObject"]["GMLTask"]["targetEdge"].split(":")[-1],
                "dataset_name": ds_name,
                "dataset_name_csv": ds_name,
                "dataset_types":query_dict["insertJSONObject"]["GMLTask"]["datasetTypesFilePath"],
                "test_size": 0.1,
                "valid_size": 0.1,
                "MINIMUM_INSTANCE_THRESHOLD": int(query_dict["insertJSONObject"]["GMLTask"]["MinInstancesPerLabel"]) if "MinInstancesPerLabel" in query_dict["insertJSONObject"]["GMLTask"].keys() else 21,
                "output_root_path": KGNET_Config.datasets_output_path,
                "target_node_type": query_dict["insertJSONObject"]["GMLTask"]["targetNode"].split(":")[-1],
                "label_node_type": (query_dict["insertJSONObject"]["GMLTask"]["labelNode"].split(":")[-1] if "labelNode" in query_dict["insertJSONObject"]["GMLTask"].keys() else None),
                "labels_count": (query_dict["insertJSONObject"]["GMLTask"]["labelsCount"] if "labelsCount" in query_dict["insertJSONObject"]["GMLTask"].keys() else None)
            },
            "training":
                {"dataset_name": ds_name,
                 "n_classes": 1000,
                 "root_path":  KGNET_Config.datasets_output_path,
                 "GNN_Method":query_dict["insertJSONObject"]["GMLTask"]["GNNMethod"],
                 "epochs": (query_dict["insertJSONObject"]["GMLTask"]["epochs"] if "epochs" in query_dict["insertJSONObject"]["GMLTask"].keys() else None),
                 "embSize": (query_dict["insertJSONObject"]["GMLTask"]["embSize"] if "embSize" in query_dict["insertJSONObject"]["GMLTask"].keys() else None)
                 }
        }
        return train_pipeline_dict

    def getTaskUri(self,query_dict):
        task_type = query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1]
        tid = self.KGMeta_Governer_obj.getGMLTaskID(query_dict)
        if tid:
            if Constants.utils.is_number(tid):
                return "kgnet:GMLTask/tid-" + Constants.utils.getIdWithPaddingZeros(tid), True
            else:
                return "kgnet:GMLTask/tid-"+tid,True
        else:
            next_tid= self.get_next_task_id(query_dict)
            return "kgnet:GMLTask/tid-"+next_tid,False
    def UpdateKGMeta(self,query_dict,transform_results_dict,train_results_dict):
        task_uri, task_exist = self.getTaskUri(query_dict)
        print("task_uri=", task_uri)
        next_model_id =train_results_dict["dataset_name"]
        model_uri = "kgnet:GMLModel/" + next_model_id
        print("model_uri=",model_uri)
        res=self.KGMeta_Governer_obj.insertGMLModel(query_dict,task_uri,task_exist,next_model_id.split('mid-')[1],model_uri,transform_results_dict,train_results_dict)
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
    def executeQuery(self, query,in_pipline=True):
        gmlqp = gmlQueryParser(query)
        start_time=datetime.now()
        if in_pipline==True:
            res_df,exectuted_Queries,=gmlqp.exec_query_plan()
            return res_df,exectuted_Queries,(datetime.now()-start_time).total_seconds()
        else:
            st = datetime.now()
            q_stmt=gmlqp.extractQueryStatmentsDict()
            formatted_gml_query = gmlQueryFormatter.format_gml_query_tree(gmlqp.query_statments)
            print("formatted_gml_query=", formatted_gml_query)
            dataInferQ,dataQ,tragetNodesq, kgmeta_model_queries_dict,model_ids = gmlQueryRewriter(q_stmt, self.KGMeta_Governer_obj).rewrite_gml_query()
            print(f"Query Parse and Rewrite Time:{(datetime.now()-st).total_seconds()} Sec")
            st=datetime.now()
            df_res = self.KG_sparqlEndpoint.executeSparqlquery(dataInferQ)
            print(f"SPARQL Query Exec Time:{(datetime.now() - st).total_seconds()} Sec")
            # print("KGMeta task select query= \n",kmetaq)
            # print("SPARQL candidate query form 2= \n",dataInferQ)
            # print("SPARQLdata only Query=\n", dataQ)
            df_res=df_res.applymap(lambda x: str(x)[1:-1])
            return df_res,dataInferQ,dataQ,tragetNodesq,kgmeta_model_queries_dict,model_ids,(datetime.now()-start_time).total_seconds()

# if __name__ == '__main__':
#     ""
#     KG_sparqlEndpoint = sparqlEndpoint(endpointUrl='http://206.12.98.118:8890/sparql/')
#     gml_operator=gmlOperator(KG_sparqlEndpoint=KG_sparqlEndpoint)
#     df=gml_operator.getKGNodeEdgeTypes(namedGraphURI="http://www.aifb.uni-karlsruhe.de")