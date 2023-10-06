from rdflib import Graph
import datetime
import rdflib
import os
import pandas as pd
import re

import Constants
from Constants import RDFEngine
from Constants import  KGNET_Config, GML_Operator_Types,GNN_Methods
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from SparqlMLaasService.ModelSelector import ModelSelector
class KGMeta_Governer_rdflib():
    def __init__(self,ttl_file_path ='KGNET_MetaGraph.ttl',KGMeta_URI="http://kgnet"):
        self.ttl_file_path = ttl_file_path
        self.g = Graph()
        self.g.parse(self.ttl_file_path)
        self.KGMeta_URI=KGMeta_URI
    def executeSparqlquery(self,query):
        results = self.g.query(query)
        col_pattern = re.search(r'select\s+((\?)?[a-zA-Z_]+(\s+(\?)?[a-zA-Z_]+)*)\s+where', query,re.IGNORECASE)
        columns = re.findall(r'(?<=\?)[\w]+', col_pattern.group())
        return pd.DataFrame(results,columns=columns)


def append_triple(Insert_Triples,model_model_uri,triple,insert_dict,key):
    if isinstance(insert_dict,dict) and str(key) in insert_dict:
        Insert_Triples += "<" + model_model_uri + "> "+triple+" "+ str(insert_dict[key])+" . \n"
    return Insert_Triples

class KGMeta_Governer(sparqlEndpoint):
    def __init__(self,endpointUrl=KGNET_Config.KGMeta_endpoint_url,KGMeta_URI="http://kgnet",RDFEngine=RDFEngine.OpenlinkVirtuoso):
        sparqlEndpoint.__init__(self, endpointUrl,RDFEngine=RDFEngine)
        self.KGMeta_URI = KGMeta_URI
    def insertKGMetadata(self,sparqlendpoint,prefix,namedGraphURI,name,description,domain):
        max_g_query="select max(?gid)+1 from <"+KGNET_Config.KGMeta_IRI+">\n where {"
        max_g_query+="?g	a	<kgnet:type/graph>. ?g  <kgnet:graph/id> ?gid.}"""
        max_gid=self.ExecScalarQuery(max_g_query)
        insertQuery="Insert into <"+KGNET_Config.KGMeta_IRI+">\n"
        gid_uri= "kgnet:graph/gid-"+Constants.utils.getIdWithPaddingZeros(7)
        insertQuery+="""
            { 	 
                <"""+gid_uri+""">	a	<kgnet:type/graph> 	 .
                <"""+gid_uri+""">	<kgnet:graph/id>	?max_gid	 .
                <"""+gid_uri+""">	<kgnet:graph/createdby>	<kgnet:user/uid-0000001> 	 .
                <"""+gid_uri+""">	<kgnet:graph/domain>	?domain	 .
                <"""+gid_uri+""">	<kgnet:graph/name>	?name 	 .
                <"""+gid_uri+""">	<kgnet:graph/nodeCount>	0 	 .
                <"""+gid_uri+""">	<kgnet:graph/nodeType>    0 	 .
                <"""+gid_uri+""">	<kgnet:graph/prefix> ?prefix	 .
                <"""+gid_uri+""">	<kgnet:graph/sparqlendpoint>	?sparqlendpoint .
                <"""+gid_uri+""">	<kgnet:user/datecreated>  "" 	 .
                <"""+gid_uri+""">	<kgnet:graph/description>  	 ?description .
                <"""+gid_uri+""">	<kgnet:graph/edgeCount>	0 	 .
                <"""+gid_uri+""">	<kgnet:graph/edgeTypes>   0 	 .
                <"""+gid_uri+""">	<kgnet:graph/namedGraphURI>  ?namedGraphURI 	 .
            }
            """
        insertQuery=insertQuery.replace("?sparqlendpoint","\""+sparqlendpoint+"\"")
        insertQuery = insertQuery.replace("?namedGraphURI", "\"" + namedGraphURI + "\"")
        insertQuery = insertQuery.replace("?prefix", "\"" + prefix + "\"")
        insertQuery = insertQuery.replace("?name", "\"" + name + "\"")
        insertQuery = insertQuery.replace("?description", "\"" + description + "\"")
        insertQuery = insertQuery.replace("?domain", "\"" + domain + "\"")
        insertQuery = insertQuery.replace("?max_gid", max_gid)
        res=self.executeSparqlquery(insertQuery)
        return max_gid,res[res.columns[0]].values[0].split(",")[1]
    def getModelKGMetadata(self, mid):
        kg_metadata_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                            select ?g ?p ?o
                            from <""" + KGNET_Config.KGMeta_IRI + """> 
                            where { ?m <kgnet:GMLModel/id> """+str(mid)+""" .
                            ?t <kgnet:GMLTask/modelID> ?m .
                            ?t <kgnet:GMLTask/appliedOnGraph> ?g .
                            ?g ?p ?o. 
                            }"""
        kg_df = self.executeSparqlquery(kg_metadata_query)
        return kg_df
    def getNextGMLModelID(self):
        next_tid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                            select max(?mid)+1 as ?mid
                             from <""" + KGNET_Config.KGMeta_IRI + """> where { ?model <kgnet:GMLModel/id> ?mid.}"""
        model_df = self.executeSparqlquery(next_tid_query)
        if len(model_df) > 0:
            return int(model_df["mid"].values[0])
        else:
            return 1
    def getNextGMLTaskID(self):
        next_tid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                        select max(?tid)+1 as ?tid
                        from <""" + KGNET_Config.KGMeta_IRI + """> where { ?task <kgnet:GMLTask/id> ?tid.}"""
        task_df = self.executeSparqlquery(next_tid_query)
        if len(task_df)>0:
            return int(task_df["tid"].values[0])
        else:
            return 1
    def getGMLTaskInfoByID(self,tid):
        tid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                        select ?s ?p ?o 
                        from <""" + KGNET_Config.KGMeta_IRI + """> where { ?s <kgnet:GMLTask/id> """+str(tid)+" . \n ?s ?p ?o .}"
        return self.executeSparqlquery(tid_query)
    def getGMLTaskBasicInfoByID(self,tid):
        t_Query= """PREFIX kgnet: <https://www.kgnet.com/>  
            select *
            where 
            {
              {
                select ?s ?p ?o
                from <"""+ KGNET_Config.KGMeta_IRI+ """>
                where {
                   ?s  <kgnet:GMLTask/id> """+str(tid)+""" .
                   ?s ?p ?o .
                  }
              }
            union
              {
                select  ?g as ?s ?g_p as ?p ?g_o as ?o
                from <"""+ KGNET_Config.KGMeta_IRI+ """> 
                {
                  ?s  <kgnet:GMLTask/id> """+str(tid)+""" .
                  ?s <kgnet:GMLTask/appliedOnGraph>	?g.
                  ?g ?g_p ?g_o.
                }
              }
            }
            """
        res_df= self.executeSparqlquery(t_Query)
        if len(res_df)==0:
            return None
        basic_info={}
        res_df["p"]=res_df["p"].apply(lambda x: x.replace("\"",""))
        res_df["o"]=res_df["o"].apply(lambda x: x.replace("\"", ""))
        p_list=res_df["p"].unique().tolist()
        basic_info["Task ID"] = res_df[res_df["p"] == "kgnet:GMLTask/id"]["o"].values[0]
        basic_info["Task Name"]=res_df[res_df["p"]=="kgnet:GMLTask/name"]["o"].values[0]
        basic_info["Task Type"] = res_df[res_df["p"] == "kgnet:GMLTask/taskType"]["o"].values[0]
        basic_info["KG name"] = res_df[res_df["p"] == "kgnet:graph/name"]["o"].values[0]
        basic_info["KG description"] = res_df[res_df["p"] == "kgnet:graph/description"]["o"].values[0]
        if "kgnet:GMLTask/targetEdge" in p_list:
            basic_info["Target Edge"] = res_df[res_df["p"] == "kgnet:GMLTask/targetEdge"]["o"].values[0]
        if "kgnet:GMLTask/targetNode" in p_list:
            basic_info["Target Node Type"] = res_df[res_df["p"] == "kgnet:GMLTask/targetNode"]["o"].values[0]
        if "kgnet:GMLTask/targetLabel" in p_list:
            basic_info["Target Label Type"] = res_df[res_df["p"] == "kgnet:GMLTask/targetLabel"]["o"].values[0]
        if "kgnet:GMLTask/dateCreated" in p_list:
            basic_info["Date Created"] = res_df[res_df["p"] == "kgnet:GMLTask/dateCreated"]["o"].values[0]
        basic_info["# Trained Models"] = len(res_df[res_df["p"] == "kgnet:GMLTask/modelID"]["o"].unique())
        # if len(res_df[res_df["p"] == "kgnet:GMLTask/modelID"]["o"].unique())>0:
        #     basic_info["Trained Models IDs"] = ",".join(list(res_df[res_df["p"] == "kgnet:GMLTask/modelID"]["o"].unique()))
        return pd.DataFrame(list(basic_info.items()),columns=["property","Value"])
    def getGMLModelInfoByID(self,mid):
        mid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                        select ?s ?p ?o
                        from <""" + KGNET_Config.KGMeta_IRI + """> where { ?s <kgnet:GMLModel/id> """+str(mid)+" . \n ?s ?p ?o .}"
        return self.executeSparqlquery(mid_query)
    def getGMLModelBasicInfoByID(self,mid):
        mid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                        select ?s ?p ?o
                        from <""" + KGNET_Config.KGMeta_IRI + """> where { ?s <kgnet:GMLModel/id> """+str(mid)+" . \n ?s ?p ?o .}"
        res_df=self.executeSparqlquery(mid_query)
        res_df["p"]=res_df["p"].apply(lambda x:x.replace("\"",""))
        res_df["o"] = res_df["o"].apply(lambda x: x.replace("\"", ""))
        p_list=res_df["p"].unique().tolist()
        basic_info={}
        basic_info["Model ID"] = res_df[res_df["p"] == "kgnet:GMLModel/id"]["o"].values[0]
        basic_info["GNN Method"] = res_df[res_df["p"] == "kgnet:GMLModel/GNNMethod"]["o"].values[0]
        # basic_info["Iinference Time"] = res_df[res_df["p"] == "kgnet:GMLModel/inferenceTime"]["o"].values[0]
        basic_info["Training Memory GB."] = float(res_df[res_df["p"] == "kgnet:GMLModel/trainingMemory"]["o"].values[0])/(1024*1024)
        basic_info["Training Time Sec."] = res_df[res_df["p"] == "kgnet:GMLModel/trainingTime"]["o"].values[0]
        if "kgnet:GMLModel/testMRR" in p_list:
            basic_info["Test MRR Score"] = res_df[res_df["p"] == "kgnet:GMLModel/testMRR"]["o"].values[0]
        if "kgnet:GMLModel/test_Hits@10" in p_list:
            basic_info["Test Hits@10 Score"] = res_df[res_df["p"] == "kgnet:GMLModel/test_Hits@10"]["o"].values[0]
        if "kgnet:GMLModel/testAccuracy" in p_list:
            basic_info["Test Acc Score"] = res_df[res_df["p"] == "kgnet:GMLModel/testAccuracy"]["o"].values[0]

        # if "kgnet:GMLModel/taskSubgraph/TOSG" in p_list:
        #     basic_info["Task-Oriented Subgraph Pattern"] = res_df[res_df["p"] == "kgnet:GMLModel/taskSubgraph/TOSG"]["o"].values[0]
        if "kgnet:GMLModel/taskSubgraph/edgeCount" in p_list:
            basic_info["# Training Triples "] = res_df[res_df["p"] == "kgnet:GMLModel/taskSubgraph/edgeCount"]["o"].values[0]
        if "kgnet:GMLTask/dateCreated" in p_list:
            basic_info["Date Created"] = res_df[res_df["p"] == "kgnet:GMLModel/dateCreated"]["o"].values[0]
        return pd.DataFrame(list(basic_info.items()), columns=["property", "Value"])

    def getGMLTaskModelsBasicInfoByID(self, tid):
        mid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                           select ?mid as ?s ?p ?o
                           from <""" + KGNET_Config.KGMeta_IRI + """> where { ?s <kgnet:GMLTask/id> """ + str(tid) + " . ?s <kgnet:GMLTask/modelID> ?mid. ?mid ?p ?o .}"
        res_df = self.executeSparqlquery(mid_query)
        res_df["s"] = res_df["s"].apply(lambda x: str(x)[1:-1] if str(x).startswith("\"") else x)
        res_df["p"] = res_df["p"].apply(lambda x: str(x)[1:-1] if str(x).startswith("\"") else x)
        res_df["o"] = res_df["o"].apply(lambda x: str(x)[1:-1] if str(x).startswith("\"") else x)
        models_lst=res_df[res_df["p"]=="kgnet:GMLModel/id"]["s"].unique().tolist()
        p_list = res_df["p"].unique().tolist()
        models_info_list=[]
        for model_uri in models_lst:
            basic_info = {}
            model_df=res_df[res_df["s"]==model_uri]
            basic_info["Model ID"] = int(model_uri.split('-')[1])
            basic_info["GNN Method"] = model_df[model_df["p"] == "kgnet:GMLModel/GNNMethod"]["o"].values[0]
            # basic_info["Iinference Time"] = res_df[res_df["p"] == "kgnet:GMLModel/inferenceTime"]["o"].values[0]
            basic_info["Training Memory GB."] = float(
                model_df[model_df["p"] == "kgnet:GMLModel/trainingMemory"]["o"].values[0]) / (1024 * 1024)
            basic_info["Training Time Sec."] =  round(float(model_df[model_df["p"] == "kgnet:GMLModel/trainingTime"]["o"].values[0]),2)
            if "kgnet:GMLModel/testMRR" in p_list:
                basic_info["Test MRR Score"] =  round(float(model_df[model_df["p"] == "kgnet:GMLModel/testMRR"]["o"].values[0]),2)
            if "kgnet:GMLModel/test_Hits@10" in p_list:
                basic_info["Test Hits@10 Score"] =  round(float(model_df[model_df["p"] == "kgnet:GMLModel/test_Hits@10"]["o"].values[0]),2)
            if "kgnet:GMLModel/testAccuracy" in p_list:
                basic_info["Test Acc Score"] =  round(float(model_df[model_df["p"] == "kgnet:GMLModel/testAccuracy"]["o"].values[0]))
            if "kgnet:GMLModel/inferenceTime" in p_list:
                basic_info["Inference Time"] = round(float(model_df[model_df["p"] == "kgnet:GMLModel/inferenceTime"]["o"].values[0]))

            # if "kgnet:GMLModel/taskSubgraph/TOSG" in p_list:
            #     basic_info["Task-Oriented Subgraph Pattern"] = \
            #     model_df[model_df["p"] == "kgnet:GMLModel/taskSubgraph/TOSG"]["o"].values[0]
            if "kgnet:GMLModel/dateCreated" in p_list:
                basic_info["Date Created"] = model_df[model_df["p"] == "kgnet:GMLModel/dateCreated"]["o"].values[0]
            models_info_list.append(basic_info)
        models_res_lst=[]
        for idx,elem in enumerate(models_info_list):
            lst=list(elem.values())
            lst.insert(0, "Model #" + str(idx + 1))
            models_res_lst.append(lst)
        # df = pd.DataFrame(models_res_lst)
        # df = df.transpose()
        header=list(models_info_list[0].keys())
        header.insert(0,'Property')
        res_df=pd.DataFrame(models_res_lst, columns=header)
        score_colums=""
        if "Test Acc Score" in res_df.columns:
            Acc_InferTime_lst = res_df[["Test Acc Score", "Inference Time"]].values.tolist()
            score_colums="Test Acc Score"
        elif "Test Hits@10 Score" in res_df.columns:
            Acc_InferTime_lst = res_df[["Test Hits@10 Score", "Inference Time"]].values.tolist()
            score_colums = "Test Hits@10 Score"
        model_idx = ModelSelector.getBestModelIdx(Acc_InferTime_lst)
        styler_df=(res_df.style
                           # .apply(lambda x:utils.highlight_value_in_column(x,color='#FF5C5C',agg='max'), subset=['Training Time Sec.', 'Training Memory GB.'], axis=0)
                           .apply(lambda x: Constants.utils.highlight_value_in_column(x, color=Constants.colors.green, agg=Constants.aggregations.min),subset=['Training Time Sec.','Inference Time', 'Training Memory GB.'], axis=0)
                           .apply(lambda x: Constants.utils.highlight_value_in_column(x, color=Constants.colors.green, agg=Constants.aggregations.max),subset=[score_colums], axis=0)
                           .apply( lambda row:  Constants.utils.highlightRowByIdx(row, model_idx , bgcolor=Constants.colors.orange, textcolor='', fontweight='bold'), axis = 1)
                           # .apply(lambda x: utils.highlight_value_in_column(x, color='#FF5C5C', agg='min'),subset=["Test Acc Score"], axis=0
        )
        return res_df,styler_df

    def OptimizeForBestModel(self, tid):
        mid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                           select (?mid as ?s) ?p ?o
                           from <""" + KGNET_Config.KGMeta_IRI + """> where { ?t <kgnet:GMLTask/id> """ + str(tid) + " . ?t <kgnet:GMLTask/modelID> ?mid. ?mid ?p ?o .}"
        res_df = self.executeSparqlquery(mid_query)
        res_df["s"] = res_df["s"].apply(lambda x: str(x)[1:-1] if str(x).startswith("\"") or str(x).startswith("<")  else x)
        res_df["p"] = res_df["p"].apply(lambda x: str(x)[1:-1] if str(x).startswith("\"") or str(x).startswith("<") else x)
        res_df["o"] = res_df["o"].apply(lambda x: str(x)[1:-1] if str(x).startswith("\"") or str(x).startswith("<") else x)
        models_lst=res_df[res_df["p"]=="kgnet:GMLModel/id"]["s"].unique().tolist()
        p_list = res_df["p"].unique().tolist()
        models_info_list=[]
        for model_uri in models_lst:
            basic_info = {}
            model_df=res_df[res_df["s"]==model_uri]
            basic_info["Model ID"] = model_df[model_df["p"] == "kgnet:GMLModel/id"]["o"].values[0]
            if "kgnet:GMLModel/test_Hits@10" in p_list:
                basic_info["Test Hits@10 Score"] =  round(float(model_df[model_df["p"] == "kgnet:GMLModel/test_Hits@10"]["o"].values[0]),2)
            if "kgnet:GMLModel/testAccuracy" in p_list:
                basic_info["Test Acc Score"] =  round(float(model_df[model_df["p"] == "kgnet:GMLModel/testAccuracy"]["o"].values[0]))
            if "kgnet:GMLModel/inferenceTime" in p_list:
                basic_info["Inference Time"] = round(float(model_df[model_df["p"] == "kgnet:GMLModel/inferenceTime"]["o"].values[0]))
            models_info_list.append(basic_info)
        models_res_lst=[]
        for idx,elem in enumerate(models_info_list):
            lst=list(elem.values())
            lst.insert(0, "Model #" + str(idx + 1))
            models_res_lst.append(lst)
        # df = pd.DataFrame(models_res_lst)
        # df = df.transpose()
        header=list(models_info_list[0].keys())
        header.insert(0,'Property')
        res_df=pd.DataFrame(models_res_lst, columns=header)
        if "Test Acc Score" in res_df.columns:
            Acc_InferTime_lst = res_df[["Test Acc Score", "Inference Time"]].values.tolist()
        elif "Test Hits@10 Score" in res_df.columns:
            Acc_InferTime_lst = res_df[["Test Hits@10 Score", "Inference Time"]].values.tolist()
        model_idx = ModelSelector.getBestModelIdx(Acc_InferTime_lst)
        return int(res_df.iloc[model_idx]["Model ID"])
    def getGraphUriByPrefix(self,prefix="MAG"):
        next_mid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                          select ?g 
                          from <""" + KGNET_Config.KGMeta_IRI + """> where { ?g <kgnet:graph/prefix> \""""+prefix+"""\" . } limit 1"""
        df_g=self.executeSparqlquery(next_mid_query)
        if len(df_g)>0:
            return df_g["g"].values[0]
        else:
            return None
    def getGMLTaskID(self,query_dict):
        task_query = ""
        operator_type = query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1]
        for pref in query_dict["prefixes"]:
            task_query += "PREFIX " + pref + ":<" + query_dict["prefixes"][pref] + "> \n"
        if operator_type==GML_Operator_Types.NodeClassification:
            task_query+= """select ?tid  from <""" + KGNET_Config.KGMeta_IRI + """> where {
                              ?task	<kgnet:GMLTask/taskType>  <kgnet:type/nodeClassification> .
                              ?task	<kgnet:GMLTask/labelNode> <"""+ query_dict["insertJSONObject"]["GMLTask"]["labelNode"] + "> . \n"
            task_query += "?task <kgnet:GMLTask/targetNode>	<" + query_dict["insertJSONObject"]["GMLTask"]["targetNode"] + "> . \n"
            task_query += "?task <kgnet:GMLTask/id>	?tid. \n} limit 1"
        elif operator_type==GML_Operator_Types.LinkPrediction:
            task_query += """select ?tid  from <""" + KGNET_Config.KGMeta_IRI + """> where {
                                        ?task	<kgnet:GMLTask/taskType>	<kgnet:type/linkPrediction> . \n"""
            task_query += "?task <kgnet:GMLTask/targetEdge>	\"" + query_dict["insertJSONObject"]["GMLTask"]["targetEdge"] + "\" . \n"
            task_query += "?task <kgnet:GMLTask/id>	?tid. \n} limit 1"

        task_df = self.executeSparqlquery(task_query)
        if len(task_df) > 0:
            return task_df["tid"].values[0]
        else:
            return None

    def insertGMLModel(self,query_dict,task_uri,task_exist,next_model_id,model_model_uri,transform_results_dict,train_results_dict):
        Insert_Triples = ""
        for pref in query_dict["prefixes"]:
            Insert_Triples += "PREFIX " + pref + ":<" + query_dict["prefixes"][pref] + "> \n"
        Insert_Triples += "With <"+KGNET_Config.KGMeta_IRI+"> Insert {"
        if task_exist==False:
            Insert_Triples+="<"+task_uri +"> a <kgnet:type/GMLTask> . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/id> "+str(int(task_uri.split("tid-")[1]))+" . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/createdBy> <kgnet:user/uid-0000001> . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/dateCreated> \""+ datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") +"\" . \n"
            if "namedGraphPrefix" in query_dict["insertJSONObject"]["GMLTask"]:
                Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/KGPrefix> \"" + query_dict["insertJSONObject"]["GMLTask"]["namedGraphPrefix"] + "\" . \n"
                graph_uri=self.getGraphUriByPrefix(query_dict["insertJSONObject"]["GMLTask"]["namedGraphPrefix"])
            if graph_uri:
                Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/appliedOnGraph> <"+ graph_uri.replace("\"","")+"> . \n"

            if query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1].strip().lower()==GML_Operator_Types.NodeClassification.strip().lower():
                Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/taskType> <kgnet:type/nodeClassification> . \n"
            elif query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1].strip().lower()==GML_Operator_Types.LinkPrediction.strip().lower():
                Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/taskType> <kgnet:type/linkPrediction> . \n"

            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/name> \""+query_dict["insertJSONObject"]["name"]+"\" . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/description>	\"\" . \n"
            if "labelNode" in query_dict["insertJSONObject"]["GMLTask"]:
                Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/labelNode>	<"+ query_dict["insertJSONObject"]["GMLTask"]["labelNode"]+"> . \n"
            if "targetNode" in query_dict["insertJSONObject"]["GMLTask"]:
                Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/targetNode> <"+ query_dict["insertJSONObject"]["GMLTask"]["targetNode"]+"> . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/targetEdge> \""+ query_dict["insertJSONObject"]["GMLTask"]["targetEdge"]+"\" . \n"
        ######################
        Insert_Triples += "<" + task_uri + ">  <kgnet:GMLTask/modelID> <" + model_model_uri + ">  . \n"
        Insert_Triples += "<" + model_model_uri + "> a <kgnet:type/GMLModel> . \n"
        Insert_Triples += "<" + model_model_uri + ">  <kgnet:GMLTask/description>	 \""+query_dict["insertJSONObject"]["name"]+"\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/id> \"" + str(next_model_id) + "\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/createBy> <kgnet:user/uid-0000001> . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/dateCreated> \""+datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") +"\". \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/GNNMethod> \""+query_dict["insertJSONObject"]["GMLTask"]["GNNMethod"]+"\" . \n"

        if query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1].strip().lower() == GML_Operator_Types.NodeClassification.strip().lower():
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/classifierType> \"MCSL\" . \n"
        elif query_dict["insertJSONObject"]["GMLTask"]["taskType"].split(":")[1].strip().lower() == GML_Operator_Types.LinkPrediction.strip().lower():
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/linkPredictionType> \"missingObject\" . \n"
        if "Inference_Time" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/inferenceTime> "+ str(train_results_dict["Inference_Time"]) +" . \n"
        else:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/inferenceTime> 0 . \n"
        #################################################### train_results_dict #######################################
        # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/modelParametersSize> " + str(train_results_dict["Model_Trainable_Paramters_Count"]) + " . \n"
        Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/modelParametersSize>', train_results_dict, "Model_Trainable_Paramters_Count")
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/modelFileSize> 0 . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/modelFileCheckSum> \"" +"" + "\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/id-SHA-25> \"\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/inferenceAPI-URL> \"\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainingTime> "+ str(train_results_dict["Train_Time"])+" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainingMemory> "+ str(train_results_dict["model_ru_maxrss"])+" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainingMachineCount> 1 . \n"

        ####################################################### Metrics ############################################
        if "Final_Test_F1" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/testF1> "+ str(train_results_dict["Final_Test_F1"])+" . \n"
        if "Final_Valid_F1" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/validF1> "+ str(train_results_dict["Final_Valid_F1"])+" . \n"
        if "Final_Train_F1" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainF1> "+ str(train_results_dict["Final_Test_F1"])+" . \n"
        if "Final_Test_Acc" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/testAccuracy> "+ str(train_results_dict["Final_Test_Acc"])+" . \n"
        if "Final_Valid_Acc" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/validAccuracy> "+ str(train_results_dict["Final_Valid_Acc"])+" . \n"
        if "Final_Train_Acc" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainAccuracy> "+ str(train_results_dict["Final_Train_Acc"])+" . \n"
        if "Final_Valid_MRR" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/validMRR> " + str(train_results_dict["Final_Valid_MRR"]) + " . \n"
        if "Final_Test_MRR" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/testMRR> " + str(train_results_dict["Final_Test_MRR"]) + " . \n"
        if "Final_Train_MRR" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainMRR> " + str(train_results_dict["Final_Train_MRR"]) + " . \n"
        if "Final_Train_Hits@10" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/train_Hits@10> " + str(train_results_dict["Final_Train_Hits@10"]) + " . \n"
        if "Final_Test_Hits@10" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/test_Hits@10> " + str(train_results_dict["Final_Test_Hits@10"]) + " . \n"
        if "Final_Valid_Hits@10" in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/valid_Hits@10> " + str(train_results_dict["Final_Valid_Hits@10"]) + " . \n"

        #################################################### train_results_dict hyperparameter #######################################
        if 'gnn_hyper_params' in train_results_dict:
            # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/hyperparameter/lr> "+ str(train_results_dict["gnn_hyper_params"]["lr"])+" . \n"
            Insert_Triples = append_triple(Insert_Triples,model_model_uri,'<kgnet:GMLModel/hyperparameter/lr>',train_results_dict["gnn_hyper_params"],"lr")
            # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/hyperparameter/numLayers> " + str(train_results_dict["gnn_hyper_params"]["num_layers"]) + " . \n"
            Insert_Triples = append_triple(Insert_Triples,model_model_uri,'<kgnet:GMLModel/hyperparameter/numLayers>',train_results_dict["gnn_hyper_params"],"num_layers")

            # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/hyperparameter/hiddenChannels> " + str(train_results_dict["gnn_hyper_params"]["hidden_channels"]) + " . \n"
            Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/hyperparameter/hiddenChannels>',
                                           train_results_dict["gnn_hyper_params"], "hidden_channels")

            # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/hyperparameter/numEpochs> "+ str(train_results_dict["gnn_hyper_params"]["epochs"]) + " . \n"
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/hyperparameter/numEpochs> " + str(
                train_results_dict["gnn_hyper_params"]["epochs"]) + " . \n"

            # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/hyperparameter/numRuns> " + str(train_results_dict["gnn_hyper_params"]["runs"]) + " . \n"
            Insert_Triples = append_triple(Insert_Triples,model_model_uri,'<kgnet:GMLModel/hyperparameter/numRuns>',train_results_dict["gnn_hyper_params"],"runs")


            # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/hyperparameter/embSize> " + str(train_results_dict["gnn_hyper_params"]["emb_size"]) + " . \n"
            Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/hyperparameter/embSize>',
                                           train_results_dict["gnn_hyper_params"], "emb_size")

            Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/hyperparameter/batchSize>',
                                           train_results_dict["gnn_hyper_params"], "batch_size")

            Insert_Triples = append_triple(Insert_Triples, model_model_uri,
                                           '<kgnet:GMLModel/hyperparameter/walkLength>',
                                           train_results_dict["gnn_hyper_params"], "walkLength")

        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/hyperparameter/optimizer> \"adam\" . \n"


        #################
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/GNNSampler/method> \"RW\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/GNNSampler/level> \"subgraph\" . \n"
        #################################################### transform_results_dict  #######################################
        # print("transform_results_dict=",transform_results_dict)
        Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/taskSubgraph/edgeCount>',transform_results_dict, "TriplesCount")
        if 'data_obj' in transform_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/nodeCount> "+str(sum(list(transform_results_dict['data_obj']['num_nodes_dict'].values())))+" . \n"
            if 'edge_reltype' in transform_results_dict['data_obj']:
                Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/edgeTypeCount> "+str(len(transform_results_dict['data_obj']['edge_reltype'].keys()))+" . \n"
            if 'num_nodes_dict' in transform_results_dict['data_obj']:
                Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/nodeTypeCount> "+str(len(transform_results_dict['data_obj']['num_nodes_dict'].keys()))+" . \n"
            if 'y_dict' in transform_results_dict['data_obj']:
                Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/targetNodeCount> "+str(transform_results_dict['data_obj']['y_dict'][list(transform_results_dict['data_obj']['y_dict'].keys())[0]].shape[0])+" . \n"
        Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/taskSubgraph/labelCount>',transform_results_dict, "ClassesCount")
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/TOSG> \""+query_dict["insertJSONObject"]["GMLTask"]["TOSG"]+"\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/targetEdge> \""+ query_dict["insertJSONObject"]["GMLTask"]["targetEdge"] + "\". \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/transformationTime>  20 . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/transformationMemory>  2 . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/split>  \"random\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/splitRel>  \"\" . \n"
        if "targetNodeFilters" in query_dict["insertJSONObject"]["GMLTask"]:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/filters>  \""+str(list(query_dict["insertJSONObject"]["GMLTask"]["targetNodeFilters"].values()))+"\" . \n"
        Insert_Triples+="} where {}"
        ###########################
        res=self.executeSparqlInsertQuery(Insert_Triples)
        if len(res)>0:
            return res[res.columns[0]].values[0].split(",")[1]
        else:
            return "#"
    def insertTriples(self,triples_lst):
        """not implemented yet"""
    def deleteTriples(self,triples_lst):
        """not implemented yet"""
class KGMeta_OntologyManger(sparqlEndpoint):
    def __init__(self,endpointUrl,KGMeta_URI="http://kgnet"):
        sparqlEndpoint.__init__(self, endpointUrl)
        self.KGMeta_URI = KGMeta_URI
    def deleteTriples(self,triples_lst):
        """not implemented yet"""
if __name__ == '__main__':
    ""
    # kgmeta_govener = KGMeta_Governer_rdflib(ttl_file_path ='KGNET_MetaGraph.ttl')
    # kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.98.118:8890/sparql',KGMeta_URI="http://kgnet")
    # query = """
    #     SELECT distinct ?LinkPredictor ?gmlModel ?mID ?apiUrl
    #     WHERE
    #     {
    #         ?LinkPredictor <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <kgnet:types/LinkPredictor> .
    #         ?LinkPredictor <kgnet:GML/SourceNode> <dblp:author> .
    #         ?LinkPredictor <kgnet:GML/DestinationNode> <dblp:Affiliation> .
    #         ?LinkPredictor <kgnet:term/uses> ?gmlModel .
    #         ?gmlModel <kgnet:GML_ID> ?mID .
    #         ?mID <kgnet:API_URL> ?apiUrl .
    #     }"""
    # res_df = kgmeta_govener.executeSparqlquery(query)
    # print(res_df)
    kgmeta_govener = KGMeta_Governer(endpointUrl=KGNET_Config.KGMeta_endpoint_url,KGMeta_URI=KGNET_Config.KGMeta_IRI)
    # query="""prefix kgnet:<http://kgnet/>
    #     select * where
    #     {
    #         {
    #             select ?t as ?s  ?p as ?p  ?o as ?o
    #             from <http://kgnet/>
    #             where
    #             {
    #             #?s ?p ?o.
    #             ?t ?p ?o.
    #             ?t ?tp ?s.
    #             ?s <kgnet:GMLModel/id> 47.
    #             }
    #         }
    #         union
    #         {
    #             select ?m as ?s ?p as ?p  ?o as ?o
    #             from <http://kgnet/>
    #             where
    #             {
    #             ?m ?p ?o.
    #             ?m <kgnet:GMLModel/id> 47.
    #             }
    #         }
    #     }
    #     limit 100
    #     """
    # res_df=kgmeta_govener.executeSparqlquery(query)
    # print(res_df)

    # kgmeta_govener.insertKGMetadata(sparqlendpoint='http://206.12.98.118:8890/sparql',
    #                                        prefix='dblp2022',
    #                                        namedGraphURI='https://dblp2022.org',
    #                                        name='dblp2022',
    #                                        description='Subgraph of dblp 2022-03 KG that contains only the papers published in 2022 and thier connected neighbours',
    #                                        domain='academic')
    kgmeta_govener.getGMLTaskModelsBasicInfoByID(26)
