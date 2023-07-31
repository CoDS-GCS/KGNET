import sys
import os
from Constants import *
from GMLaaS.run_pipeline import run_training_pipeline
from KGMeta_Governer import KGMeta_Governer
from gmlRewriter import gmlQueryParser,gmlQueryRewriter
from TaskSampler.TOSG_Extraction_NC import get_d1h1_query as get_NC_d1h1_query
from TaskSampler.TOSG_Extraction_NC import get_d2h1_query as get_NC_d2h1_query
from TaskSampler.TOSG_Extraction_LP import write_d2h1_TOSG
class GML_Operator():
    def __init__(self ):
        self.KGMeta_Governer_obj =None
        self.GML_Query_Type = None
class GML_Insert_Operator(GML_Operator):
    def __init__(self,KGMeta_Governer_obj ):
        self.KGMeta_Governer_obj = KGMeta_Governer_obj
        self.GML_Query_Type = GML_Query_Types.Insert
    def check_gml_task_exist(self,query_dict):
        task_exist_query = ""
        for key in query_dict["prefixes"].keys():
            task_exist_query += "PREFIX " + str(key) + ":<" + query_dict["prefixes"][key] + ">\n"

        task_exist_query += """SELECT
        distinct ?model  ?p ?o
        WHERE
        {\n"""
        task_var = "" + query_dict["Insert_JSON_Object"]["GML-Task"]["TaskType"].split(":")[1]

        if task_var == "NodeClassifier":
            task_exist_query += "?" + task_var + " a <kgnet:types/" + task_var + ">.\n"
            task_exist_query += "?" + task_var + " <kgnet:GML/TargetNode> <" + query_dict["Insert_JSON_Object"]["GML-Task"][
                "TargetNode"] + ">.\n"
            task_exist_query += "?" + task_var + " <kgnet:GML/NodeLabel> <" + query_dict["Insert_JSON_Object"]["GML-Task"][
                "NodeLabel"] + ">.\n"
            task_exist_query += "?" + task_var + """ <kgnet:term/uses>	?task.
                ?task <kgnet:GML_ID> ?model.
                ?model <kgnet:GMLMethod> <kgnet:GML/Method/""" + query_dict["Insert_JSON_Object"]["GML-Task"]["GNN_Method"] + """>.
                ?model ?p ?o.
                }"""
        task_res_df = self.KGMeta_Governer_obj.executeSparqlquery(task_exist_query)
        return True if len(task_res_df) > 0 else False, task_res_df, task_exist_query
    def sample_Task_Subgraph(self, query_dict,output_path,TOSG="d1h1"):
        task_var = query_dict["Insert_JSON_Object"]["GML-Task"]["TaskType"].split(":")[1]
        tragetNode_filter_statments=None
        if  "TargetNodeFilters" in query_dict["Insert_JSON_Object"]["GML-Task"]:
            for filter in query_dict["Insert_JSON_Object"]["GML-Task"]["TargetNodeFilters"]:
                tragetNode_filter_statments=""
                filter_vals=query_dict["Insert_JSON_Object"]["GML-Task"]["TargetNodeFilters"][filter]
                tragetNode_filter_statments+="?s" + filter_vals[0]+" "+ filter_vals[1]+" .\n"
                for idx in range(2,len(filter_vals)):
                    tragetNode_filter_statments+=filter_vals[idx]+".\n"

        if task_var == "NodeClassifier":
            target_rel_uri=query_dict["Insert_JSON_Object"]["GML-Task"]["LabelPredicate"]
            named_graph_uri=query_dict["Insert_JSON_Object"]["GML-Task"]["named_graph_uri"]
            if TOSG=="d1h1":
                query=[get_NC_d1h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)]
            elif TOSG=="d2h1":
                query=get_NC_d2h1_query(graph_uri=named_graph_uri,target_rel_uri=target_rel_uri,tragetNode_filter_statments=tragetNode_filter_statments)
            self.KGMeta_Governer_obj.execute_sparql_multithreads(query,output_path,start_offset=0, batch_size=10**5, threads_count=16,rows_count=None)
    def create_train_pipline_json(self,query_dict):
        train_pipeline_dict={
            "transformation": {
                "target_rel": query_dict["Insert_JSON_Object"]["GML-Task"]["LabelPredicate"],
                "dataset_name": query_dict["Insert_JSON_Object"]["Name"],
                "dataset_name_csv": query_dict["Insert_JSON_Object"]["Name"],
                "dataset_types":query_dict["Insert_JSON_Object"]["GML-Task"]["dataset_types_path"],
                "test_size": 0.1,
                "valid_size": 0.1,
                "MINIMUM_INSTANCE_THRESHOLD": 6,
                "output_root_path": "/media/hussein/UbuntuData/GithubRepos/KGNET/Datasets/"
            },
            "training":
                {"dataset_name": query_dict["Insert_JSON_Object"]["Name"],
                 "n_classes": 1000,
                 "root_path": "/media/hussein/UbuntuData/GithubRepos/KGNET/Datasets/"
                 }
        }
        return train_pipeline_dict
    def executeQuery(self,query_dict):
        exist,task_res_df,task_exist_query=self.check_gml_task_exist(query_dict)
        gml_model_status=""
        transform_results_dict={}
        train_results_dict={}
        if not exist:
            train_pipline_json = self.create_train_pipline_json(query_dict)
            print("################# TOSG Sampling ###########################")
            self.sample_Task_Subgraph(query_dict,
                                      train_pipline_json["transformation"]["output_root_path"]+train_pipline_json["transformation"]["dataset_name"]+".tsv",
                                      query_dict["Insert_JSON_Object"]["GML-Task"]["TOSG"])
            print("################# Start GNN Task Training  ###########################")
            transform_results_dict,train_results_dict=run_training_pipeline(json_args=train_pipline_json)
            gml_model_status="SPARQL-ML Insert Operator Executed Scussfully"
        else:
            gml_model_status = "There is GML model exist with URI:"+task_res_df["model"].values[0].replace("\"","")
        return gml_model_status,transform_results_dict,train_results_dict

class GML_Delete_Operator(GML_Operator):
    def __init__(self,KGMeta_Governer_obj ):
        self.KGMeta_Governer_obj = KGMeta_Governer_obj
        self.GML_Query_Type=GML_Query_Types.Delete

class GML_Inference_Operator(GML_Operator):
    def __init__(self,KGMeta_Governer_obj ):
        self.KGMeta_Governer_obj = KGMeta_Governer_obj
        self.GML_Query_Type = GML_Query_Types.Inference
if __name__ == '__main__':
    dblp_LP = """
       prefix dblp:<https://dblp.org/rdf/schema#>
       prefix kgnet: <https://www.kgnet.ai/>
       select ?author ?affiliation
       where {
       ?author a dblp:Person.    
       ?author ?LinkPredictor ?affiliation.    
       ?LinkPredictor  a <kgnet:types/LinkPredictor>.
       ?LinkPredictor  <kgnet:GML/SourceNode> <dblp:author>.
       ?LinkPredictor  <kgnet:GML/DestinationNode> <dblp:Affiliation>.
       ?LinkPredictor <kgnet:term/uses> ?gmlModel .
       ?gmlModel <kgnet:GML_ID> ?mID .
       ?mID <kgnet:API_URL> ?apiUrl.
       ?mID <kgnet:GMLMethod> <kgnet:GML/Method/MorsE>.
       }
       limit 10
       offset 0
       """
    dblp_NC = """
          prefix dblp:<https://dblp.org/rdf/schema#>
          prefix kgnet: <https://www.kgnet.ai/>
          select ?paper ?title ?venue 
          where {
          ?paper a dblp:Publication.
          ?paper dblp:title ?title.
          ?paper <https://dblp.org/rdf/schema#publishedIn> ?o.
          ?paper <https://dblp.org/has_gnn_model> 1.
          ?paper ?NodeClassifier ?venue.
          ?NodeClassifier a <kgnet:types/NodeClassifier>.
          ?NodeClassifier <kgnet:GML/TargetNode> <dblp:Publication>.
          ?NodeClassifier <kgnet:GML/NodeLabel> <dblp:venue>.
          }
          limit 100
          """

    DBLP_Insert_Query = """
           prefix dblp:<https://www.dblp.org/>
           prefix kgnet:<https://www.kgnet.com/>
           Insert into <kgnet>
           where{
               select * from kgnet.TrainGML(
               {
               "Name":"DBLP_Year_lt_1960_Paper_Venue_Classifer",
               "GML-Task":{
                   "TaskType":"kgnet:NodeClassifier",
                   "TargetNode":"dblp:Publication",
                   "NodeLabel":"dblp:venue2",
                   "LabelPredicate":"https://dblp.org/rdf/schema#publishedInJournal",
                   "named_graph_uri":"http://dblp.org",
                   "GNN_Method":\""""+GNN_Methods.Graph_SAINT+"""\",
                   "dataset_types_path":"/media/hussein/UbuntuData/GithubRepos/KGNET/Datasets/DBLP_Types.csv",
                   "TOSG":\""""+TOSG_Patterns.d1h1+"""\",
                   "TargetNodeFilters":{
                "filter1":["<https://dblp.org/rdf/schema#yearOfPublication>", "?year","filter(xsd:integer(?year)<1960)"]
                    }
                   },
               "TaskBudget":{
                   "MaxMemory":"50GB",
                   "MaxTime":"1h",
                   "Priority":"ModelScore"}
               })}"""
    LinkedIMDB_Insert_Query = """
               prefix kgnet:<https://www.kgnet.com/>
               Insert into <kgnet>
               where{
                   select * from kgnet.TrainGML(
                   {
                   "Name":"LIMDB_Year_lte_1990_Film_Language_Classifer",
                   "GML-Task":{
                       "TaskType":"kgnet:NodeClassifier",
                       "TargetNode":"imdb:Film",
                       "NodeLabel":"imdb:Language",
                       "LabelPredicate":"http://data.linkedmdb.org/resource/movie/language",
                       "named_graph_uri":"https://linkedimdb",
                       "GNN_Method":\""""+GNN_Methods.Graph_SAINT+"""\",
                       "dataset_types_path":"/media/hussein/UbuntuData/GithubRepos/KGNET/Datasets/LinkedIMDB_Types.csv",
                       "TOSG":\""""+TOSG_Patterns.d1h1+"""\",
                       "TargetNodeFilters":{
                "filter1":["<http://purl.org/dc/terms/date>", "?year","filter(xsd:integer(?year)<=1990)"]
                    }
                       },
                   "TaskBudget":{
                       "MaxMemory":"50GB",
                       "MaxTime":"1h",
                       "Priority":"ModelScore"}
                   })}"""
    MAG_Insert_Query = """
                   prefix mag:<http://mag.org/>
                   prefix kgnet:<https://www.kgnet.com/>
                   Insert into <kgnet>
                   where{
                       select * from kgnet.TrainGML(
                       {
                       "Name":"MAG_Year_lte_2011_Paper_Venue_Classifer",
                       "GML-Task":{
                           "TaskType":"kgnet:NodeClassifier",
                           "TargetNode":"mag:Paper",
                           "NodeLabel":"mag:Venue",
                           "LabelPredicate":"https://makg.org/property/appearsInJournal",
                           "named_graph_uri":"http://mag.org",
                           "GNN_Method":\""""+GNN_Methods.Graph_SAINT+"""\",
                           "dataset_types_path":"/media/hussein/UbuntuData/GithubRepos/KGNET/Datasets/MAG_Types.csv",
                           "TOSG":\""""+TOSG_Patterns.d1h1+"""\",
                           "TargetNodeFilters":{
                    "filter1":["<https://makg.org/publish_year>", "?year","filter(xsd:integer(?year)<=2011)"]
                        }
                           },
                       "TaskBudget":{
                           "MaxMemory":"50GB",
                           "MaxTime":"1h",
                           "Priority":"ModelScore"}
                       })}"""


    # Insert_Query="""prefix dblp:<https://www.dblp.org/>
    #     prefix kgnet:<https://www.kgnet.com/>
    #     #Insert {GRAPH <kgnet> {?s ?p ?o}}
    #     select *
    #     where
    #     {
    #         GRAPH <http://dblp.org>
    #         {
    #             SELECT ?s ?p ?o
    #             { ?s ?p ?o.}
    #             limit 1
    #         }
    #     }
    #     """
    # kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.98.118:8890/sparql', KGMeta_URI="http://kgnet")
    # kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.99.253:8890/sparql/', KGMeta_URI="http://kgnet")
    kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.97.2:8890/sparql/', KGMeta_URI="http://kgnet")
    # gmlqp = gmlQueryParser(dblp_NC)
    # (dataq,kmetaq)=gmlQueryRewriter(gmlqp.extractQueryStatmentsDict(),kgmeta_govener).rewrite_gml_query()

    # insert_task_dict = gmlQueryParser(DBLP_Insert_Query).extractQueryStatmentsDict()
    # insert_task_dict = gmlQueryParser(LinkedIMDB_Insert_Query).extractQueryStatmentsDict()
    insert_task_dict = gmlQueryParser(MAG_Insert_Query).extractQueryStatmentsDict()
    gml_insert_op=GML_Insert_Operator(kgmeta_govener)
    print(gml_insert_op.executeQuery(insert_task_dict))