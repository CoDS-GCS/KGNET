import Constants
import pandas as pd
from Constants import *
from SparqlMLaasService.GMLOperators import gmlInsertOperator,gmlInferenceOperator
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
from SparqlMLaasService.gmlRewriter import gmlQueryParser,gmlQueryRewriter
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from GMLWebServiceApis import GML_to_CSV as gcsv
from GMLQueryRewriter import queryRewriter_old as gr
import sys
# GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"KGNET/"
# GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"KGNET/GMLaaS/"
# GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"KGNET/GMLaaS/models"
# GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"KGNET/SparqlMLaasService/"
class KGNET():
    def __init__(self,KG_endpointUrl,KGMeta_endpointUrl="http://206.12.98.118:8890/sparql", KGMeta_KG_URI=Constants.KGNET_Config.KGMeta_IRI,KG_NamedGraph_IRI=None,KG_Prefix=None):
        self.KGMeta_Governer = KGMeta_Governer(endpointUrl=KGMeta_endpointUrl, KGMeta_URI=KGMeta_KG_URI)
        self.KG_sparqlEndpoint = sparqlEndpoint(endpointUrl=KG_endpointUrl)
        self.gml_insert_op = gmlInsertOperator(self.KGMeta_Governer,self.KG_sparqlEndpoint)
        self.KG_NamedGraph_URI = KG_NamedGraph_IRI
        if KG_Prefix:
            self.kg_Prefix =KG_NamedGraph_IRI.split(".")[1]
        else:
            self.kg_Prefix=KG_Prefix

    def train_GML(self,sparql_ml_insert_query):
        insert_task_dict = gmlQueryParser(sparql_ml_insert_query).extractQueryStatmentsDict()
        model_info, transform_info, train_info = self.gml_insert_op.executeQuery(insert_task_dict)
        return model_info, transform_info, train_info
    def getTargetEdgeTypeIRI(self,kg_prefix,target_edge_short):
        edges_query=""" select   distinct ?p  
                       from <"""+ self.KG_NamedGraph_URI+""">
                       where { ?s ?p ?o.} limit 1000 """
        edges_df=self.KG_sparqlEndpoint.executeSparqlquery(edges_query)
        edges_df["p"] = edges_df["p"].apply(lambda x: str(x).replace("\"", ""))
        edges_df["p_lower"] = edges_df["p"].apply(lambda x: str(x).lower())
        target_edge_df=edges_df[edges_df["p_lower"].str.endswith(target_edge_short.lower())]
        return target_edge_df["p"].values[0]
    def train_GML(self,operatorType,GNNMethod,targetNodeType=None,labelNodeType=None,targetEdge=None):
        if self.kg_Prefix is None:
            if operatorType==Constants.GML_Operator_Types.NodeClassification:
                self.kg_Prefix=targetNodeType.split(":")[0]
                kg_types_path = Constants.KGNET_Config.datasets_output_path + self.kg_Prefix + "_Types.csv"
                kg_types_ds = pd.read_csv(kg_types_path, header=None)
                target_edge_df = kg_types_ds[(kg_types_ds[0].str.lower() == targetNodeType.split(":")[1].lower()) & (kg_types_ds[2].str.lower() == labelNodeType.split(":")[1].lower())]
                targetEdge = target_edge_df[1].values[0]
                targetEdge = self.getTargetEdgeTypeIRI(self.kg_Prefix, targetEdge)
            elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
                self.kg_Prefix = targetEdge.split(":")[0]

        kg_types_path = Constants.KGNET_Config.datasets_output_path + self.kg_Prefix + "_Types.csv"
        ######################### write sparqlML query #########################
        sparql_ml_insert_query=" prefix "+ self.kg_Prefix+":<"+Constants.KGs_prefixs_dic[self.kg_Prefix]+"> \n"
        sparql_ml_insert_query+= """ prefix kgnet:<https://www.kgnet.com/>
           Insert into <kgnet>
           where{
               select * from kgnet.TrainGML(
               {\n"""
        if operatorType == Constants.GML_Operator_Types.NodeClassification:
            sparql_ml_insert_query+="\"name\":\""+operatorType+">"+self.kg_Prefix+">"+targetNodeType+">"+labelNodeType+">"+GNNMethod+"\",\n"
        elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
            sparql_ml_insert_query += "\"name\":\"" + operatorType +">"+self.kg_Prefix+ ">" +targetEdge.split("/")[-1] + ">" + GNNMethod + "\",\n"
        sparql_ml_insert_query+="\"GMLTask\":{\"taskType\":\"kgnet:"+operatorType+"\",\n"
        if targetNodeType is not None:
            sparql_ml_insert_query+="\"targetNode\":\""+targetNodeType+"\",\n"
        if labelNodeType is not None:
            sparql_ml_insert_query+="\"labelNode\":\""+labelNodeType+"\",\n"

        sparql_ml_insert_query+="\"namedGraphURI\":\""+ self.KG_NamedGraph_URI+"\",\n"
        sparql_ml_insert_query += "\"namedGraphPrefix\":\"" + self.kg_Prefix + "\",\n"
        sparql_ml_insert_query+="\"targetEdge\":\""+targetEdge+"\",\"GNNMethod\":\""+GNNMethod+"\",\n"
        sparql_ml_insert_query+="\"datasetTypesFilePath\":\""+kg_types_path+"\",\n"
        if operatorType == Constants.GML_Operator_Types.NodeClassification:
            sparql_ml_insert_query+="\"TOSG\":\""+TOSG_Patterns.d1h1+"\""
        elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
            sparql_ml_insert_query+="\"TOSG\":\""+TOSG_Patterns.d2h1+"\""
        if operatorType == Constants.GML_Operator_Types.NodeClassification and self.kg_Prefix=="dblp":
            sparql_ml_insert_query += """,\n "targetNodeFilters":{"filter1":["<https://dblp.org/rdf/schema#yearOfPublication>", "?year","filter(xsd:integer(?year)<=1950)"] \n}\n"""
        sparql_ml_insert_query += "}\n})}"
        print("sparql_ml_insert_query=",sparql_ml_insert_query)
        ######################### write sparqlML query #########################
        insert_task_dict = gmlQueryParser(sparql_ml_insert_query).extractQueryStatmentsDict()
        model_info, transform_info, train_info = self.gml_insert_op.executeQuery(insert_task_dict)
        return model_info, transform_info, train_info
    def executeSPARQLMLInferenceQuery(self,query):
        gmlInferenceOp=gmlInferenceOperator(self.KGMeta_Governer,self.KG_sparqlEndpoint)
        dataInferQuery,dataQuery,gmlQuery=gmlInferenceOp.executeQuery(query)
        return dataInferQuery,dataQuery,gmlQuery
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

    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',)
    kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',
                  KG_NamedGraph_IRI='http://www.aifb.uni-karlsruhe.de')

    model_info, transform_info, train_info=kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification,targetNodeType="aifb:Person",labelNodeType="aifb:ResearchGroup",GNNMethod=Constants.GNN_Methods.Graph_SAINT)

    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql')
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="dblp:Publication",
    #     labelNodeType="dblp:venue", GNNMethod=Constants.GNN_Methods.Graph_SAINT)
    # print(model_info)
    #################################### LP ############################
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='http://www.aifb.uni-karlsruhe.de',KG_Prefix="aifb")
    # model_info, transform_info, train_info = kgnet.train_GML(
    #     operatorType=Constants.GML_Operator_Types.LinkPrediction, targetEdge="http://swrc.ontoware.org/ontology#publication", GNNMethod=Constants.GNN_Methods.RGCN)
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql')
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="dblp:Publication",
    #     labelNodeType="dblp:venue", GNNMethod=Constants.GNN_Methods.Graph_SAINT)
    # print(model_info)
    ################################## SPARQL ML Execute ##########################
    inference_query_NC = """
                prefix aifb:<http://swrc.ontoware.org/ontology#>
                prefix kgnet:<http://kgnet/>
                select ?person ?aff
                from <http://www.aifb.uni-karlsruhe.de>
                where
                {
                ?person a aifb:Person.
                ?person ?NodeClassifier ?aff.
                ?NodeClassifier a <kgnet:types/NodeClassifier>.
                ?NodeClassifier <kgnet:targetNode> aifb:Person.
                ?NodeClassifier <kgnet:labelNode> aifb:ResearchGroup.
                }
                limit 100
            """
    inference_query_LP = """
                     prefix aifb:<http://swrc.ontoware.org/ontology#>
                    prefix kgnet:<https://kgnet/>
                    select ?author ?publication
                    where {
                    ?author a aifb:Person.    
                    ?author ?LinkPredictor ?publication.    
                    ?LinkPredictor  a <kgnet:types/LinkPredictor>.
                    ?LinkPredictor  <kgnet:targetEdge> "http://swrc.ontoware.org/ontology#publication".
                    ?LinkPredictor <kgnet:GNNMethod> "MorsE" .
                    }
                    limit 100
                    offset 0
                """
    # kgnet=KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',KG_NamedGraph_IRI='http://www.aifb.uni-karlsruhe.de')
    # dataInferQuery,dataQuery,gmlQuery=kgnet.executeSPARQLMLInferenceQuery(inference_query_NC)
    # print(dataInferQuery,"\n\n\n\n##################\n",dataQuery,"\n\n\n\n##################\n",gmlQuery)