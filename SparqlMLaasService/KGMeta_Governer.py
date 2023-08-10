from rdflib import Graph
import rdflib
import os
import pandas as pd
import re

import Constants
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
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
    if key in insert_dict:
        Insert_Triples += "<" + model_model_uri + "> "+triple+" "+ str(insert_dict[key])+" . \n"
    return Insert_Triples

class KGMeta_Governer(sparqlEndpoint):
    def __init__(self,endpointUrl,KGMeta_URI="http://kgnet"):
        sparqlEndpoint.__init__(self, endpointUrl)
        self.KGMeta_URI = KGMeta_URI
    def getNextGMLModelID(self):
        next_tid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                            select max(?mid)+1 as ?mid
                             from <""" + Constants.KGNET_Config.KGMeta_IRI + """> where { ?model <kgnet:GMLModel/id> ?mid.}"""
        model_df = self.executeSparqlquery(next_tid_query)
        if len(model_df) > 0:
            return int(model_df["mid"].values[0])
        else:
            return 1
    def getNextGMLTaskID(self):
        next_tid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                        select max(?tid)+1 as ?tid
                        from <""" + Constants.KGNET_Config.KGMeta_IRI + """> where { ?task <kgnet:GMLTask/id> ?tid.}"""
        task_df = self.executeSparqlquery(next_tid_query)
        if len(task_df)>0:
            return int(task_df["tid"].values[0])
        else:
            return 1
    def getGMLTaskByID(self,tid):
        tid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                        select ?s ?p ?o 
                        from <""" + Constants.KGNET_Config.KGMeta_IRI + """> where { ?s <kgnet:GMLTask/id> """+str(tid)+" . \n ?s ?p ?o .}"
        return self.executeSparqlquery(tid_query)
    def getGMLModelByID(self,mid):
        mid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                        select ?s ?p ?o
                        from <""" + Constants.KGNET_Config.KGMeta_IRI + """> where { ?s <kgnet:GMLModel/id> """+str(mid)+" . \n ?s ?p ?o .}"
        return self.executeSparqlquery(mid_query)

    def getGraphUriByPrefix(self,prefix="MAG"):
        next_mid_query = """PREFIX kgnet: <https://www.kgnet.com/>  
                          select ?g 
                          from <""" + Constants.KGNET_Config.KGMeta_IRI + """> where { ?g <kgnet:graph/prefix> \""""+prefix+"""\" . } limit 1"""
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
        if operator_type==Constants.GML_Operator_Types.NodeClassification:
            task_query+= """select ?tid  from <""" + Constants.KGNET_Config.KGMeta_IRI + """> where {
                              ?task	<kgnet:GMLTask/taskType>  <kgnet:type/nodeClassification> .
                              ?task	<kgnet:GMLTask/labelNode> <"""+ query_dict["insertJSONObject"]["GMLTask"]["labelNode"] + "> . \n"
            task_query += "?task <kgnet:GMLTask/targetNode>	<" + query_dict["insertJSONObject"]["GMLTask"]["targetNode"] + "> . \n"
            task_query += "?task <kgnet:GMLTask/id>	?tid. \n} limit 1"
        elif operator_type==Constants.GML_Operator_Types.LinkPrediction:
            task_query += """select ?tid  from <""" + Constants.KGNET_Config.KGMeta_IRI + """> where {
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
        Insert_Triples += "Insert into <"+Constants.KGNET_Config.KGMeta_IRI+"> {"
        if task_exist==False:
            Insert_Triples+="<"+task_uri +"> a <kgnet:type/GMLTask> . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/id> "+str(int(task_uri.split("tid-")[1]))+" . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/createdBy> <kgnet:user/uid-0000001> . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/dateCreated> \"\" . \n"
            if "targetNode" in query_dict["insertJSONObject"]["GMLTask"]:
                graph_uri=self.getGraphUriByPrefix(query_dict["insertJSONObject"]["GMLTask"]["targetNode"].split(":")[0])
            elif "targetEdge" in query_dict["insertJSONObject"]["GMLTask"]:
                graph_uri = self.getGraphUriByPrefix(
                    query_dict["insertJSONObject"]["GMLTask"]["targetEdge"].split(":")[0])

            if graph_uri:
                Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/appliedOnGraph> <"+ graph_uri.replace("\"","")+"> . \n"
            Insert_Triples += "<" + task_uri + "> <kgnet:GMLTask/taskType> <kgnet:type/nodeClassification> . \n"
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
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/id> " + str(next_model_id) + " . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/createBy> <kgnet:user/uid-0000001> . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/dateCreated> \"\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/GNNMethod> \""+query_dict["insertJSONObject"]["GMLTask"]["GNNMethod"]+"\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/classifierType> \"MCSL\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/linkPredictionType> \"missingObject\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/inferenceTime> 10 . \n"

        # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/modelParametersSize> " + str(train_results_dict["Model_Trainable_Paramters_Count"]) + " . \n"
        Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/modelParametersSize>', train_results_dict, "Model_Trainable_Paramters_Count")

        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/modelFileSize> 0 . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/modelFileCheckSum> \"" +"" + "\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/id-SHA-25> \"\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/inferenceAPI-URL> \"\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainingTime> "+ str(train_results_dict["Train_Time"])+" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainingMemory> "+ str(train_results_dict["model_ru_maxrss"])+" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainingMachineCount> 1 . \n"
        ################
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/testF1> 0 . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainF1> 0 . \n"
        if "Final_Test_Acc" in query_dict["insertJSONObject"]["GMLTask"]:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/testAccuracy> "+ str(train_results_dict["Final_Test_Acc"])+" . \n"
        elif "Final_Test_MRR" in query_dict["insertJSONObject"]["GMLTask"]:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/testMRR> " + str(
                train_results_dict["Final_Test_MRR"]) + " . \n"

        if "Final_Train_Acc" in query_dict["insertJSONObject"]["GMLTask"]:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/trainAccuracy> "+ str(train_results_dict["Final_Train_Acc"])+" . \n"
        elif "Final_Valid_MRR" in query_dict["insertJSONObject"]["GMLTask"]:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/validMRR> " + str(
                train_results_dict["Final_Valid_MRR"]) + " . \n"

        if "Final_Test_Hits@10" in query_dict["insertJSONObject"]["GMLTask"]:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/train_Hits@10> " + str(
                train_results_dict["Final_Test_Hits@10"]) + " . \n"

        if "Final_Valid_Hits@10" in query_dict["insertJSONObject"]["GMLTask"]:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/train_Hits@10> " + str(
                train_results_dict["Final_Valid_Hits@10"]) + " . \n"

        #################
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
        # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/GNNSampler/batchSize> "+ str(train_results_dict["gnn_hyper_params"]["batch_size"] )+ " . \n"

        # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/GNNSampler/walkLength> "+ str(train_results_dict["gnn_hyper_params"]["walk_length"]) + " . \n"

        #################
        # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/edgeCount> "+ str(transform_results_dict["TriplesCount"])+" . \n"
        Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/taskSubgraph/edgeCount>',transform_results_dict, "TriplesCount")

        if 'data_obj' in train_results_dict:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/nodeCount> "+str(sum(list(train_results_dict['data_obj']['num_nodes_dict'].values())))+" . \n"
            # Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/taskSubgraph/nodeCount>',transform_results_dict['data_obj'], "TriplesCount")

            if 'edge_reltype' in train_results_dict['data_obj']:
                Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/edgeTypeCount> "+str(len(train_results_dict['data_obj']['edge_reltype'].keys()))+" . \n"

            if 'num_nodes_dict' in train_results_dict['data_obj']:
                Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/nodeTypeCount> "+str(len(train_results_dict['data_obj']['num_nodes_dict'].keys()))+" . \n"

            if 'y_dict' in train_results_dict['data_obj']:
                Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/targetNodeCount> "+str(train_results_dict['data_obj']['y_dict'][list(train_results_dict['data_obj']['y_dict'].keys())[0]].shape[0])+" . \n"
        # Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/labelCount> "+str(transform_results_dict["ClassesCount"])+" . \n"
        Insert_Triples = append_triple(Insert_Triples, model_model_uri, '<kgnet:GMLModel/taskSubgraph/labelCount>',transform_results_dict, "ClassesCount")


        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/TOSG> \"d1h1\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/targetEdge> \""+ query_dict["insertJSONObject"]["GMLTask"]["targetEdge"] + "\". \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/transformationTime>  20 . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/transformationMemory>  2 . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/split>  \"random\" . \n"
        Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/splitRel>  \"\" . \n"
        if "targetNodeFilters" in query_dict["insertJSONObject"]["GMLTask"]:
            Insert_Triples += "<" + model_model_uri + "> <kgnet:GMLModel/taskSubgraph/filters>  \""+str(list(query_dict["insertJSONObject"]["GMLTask"]["targetNodeFilters"].values()))+"\" . \n"
        Insert_Triples+="}"
        ###########################
        res=self.executeSparqlquery(Insert_Triples)
        return res[res.columns[0]].values[0].split(",")[1]
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
    # kgmeta_govener = KGMeta_Governer_rdflib(ttl_file_path ='KGNET_MetaGraph.ttl')
    kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.98.118:8890/sparql',KGMeta_URI="http://kgnet")
    query = """ 
        SELECT distinct ?LinkPredictor ?gmlModel ?mID ?apiUrl
        WHERE
        {
            ?LinkPredictor <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <kgnet:types/LinkPredictor> .
            ?LinkPredictor <kgnet:GML/SourceNode> <dblp:author> .
            ?LinkPredictor <kgnet:GML/DestinationNode> <dblp:Affiliation> .
            ?LinkPredictor <kgnet:term/uses> ?gmlModel .
            ?gmlModel <kgnet:GML_ID> ?mID .
            ?mID <kgnet:API_URL> ?apiUrl .   
        }"""
    res_df = kgmeta_govener.executeSparqlquery(query)
    print(res_df)
