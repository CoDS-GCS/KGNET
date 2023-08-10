import Constants
import pandas as pd
from Constants import *
from SparqlMLaasService.GMLOperators import gmlInsertOperator
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
from SparqlMLaasService.gmlRewriter import gmlQueryParser,gmlQueryRewriter
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
import sys
# GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"KGNET/"
# GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"KGNET/GMLaaS/"
# GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"KGNET/GMLaaS/models"
# GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"KGNET/SparqlMLaasService/"
class KGNET():
    def __init__(self,KG_endpointUrl,KGMeta_endpointUrl="http://206.12.98.118:8890/sparql", KGMeta_KG_URI=Constants.KGNET_Config.KGMeta_IRI):
        self.KGMeta_Governer = KGMeta_Governer(endpointUrl=KGMeta_endpointUrl, KGMeta_URI=KGMeta_KG_URI)
        self.KG_sparqlEndpoint = sparqlEndpoint(endpointUrl=KG_endpointUrl)
        self.gml_insert_op = gmlInsertOperator(self.KGMeta_Governer,  self.KG_sparqlEndpoint)
    def train_GML(self,sparql_ml_insert_query):
        insert_task_dict = gmlQueryParser(sparql_ml_insert_query).extractQueryStatmentsDict()
        model_info, transform_info, train_info = self.gml_insert_op.executeQuery(insert_task_dict)
        return model_info, transform_info, train_info
    def getTargetEdgeTypeURI(self,kg_prefix,target_edge_short):
        edges_query=""" select   distinct ?p  
                       from <"""+Constants.namedGraphURI_dic[kg_prefix]+""">
                       where { ?s ?p ?o.} limit 1000 """
        edges_df=self.KG_sparqlEndpoint.executeSparqlquery(edges_query)
        edges_df["p"] = edges_df["p"].apply(lambda x: str(x).replace("\"", ""))
        edges_df["p_lower"] = edges_df["p"].apply(lambda x: str(x).lower())
        target_edge_df=edges_df[edges_df["p_lower"].str.endswith(target_edge_short.lower())]
        return target_edge_df["p"].values[0]
    def train_GML(self,operatorType,targetNodeType,labelNodeType,GNNMethod):
        kg_prefix=targetNodeType.split(":")[0]
        kg_types_path=Constants.KGNET_Config.datasets_output_path+kg_prefix+"_Types.csv"
        kg_types_ds=pd.read_csv(kg_types_path,header=None)
        target_edge_df=kg_types_ds[(kg_types_ds[0].str.lower()==targetNodeType.split(":")[1].lower()) & (kg_types_ds[2].str.lower()==labelNodeType.split(":")[1].lower())]
        target_edge=target_edge_df[1].values[0]
        target_edge=self.getTargetEdgeTypeURI(kg_prefix,target_edge)
        ######################### write sparqlML query #########################
        sparql_ml_insert_query=" prefix "+kg_prefix+":<"+Constants.KGs_prefixs_dic[kg_prefix]+"> \n"
        sparql_ml_insert_query+= """ prefix kgnet:<https://www.kgnet.com/>
           Insert into <kgnet>
           where{
               select * from kgnet.TrainGML(
               {\n"""
        sparql_ml_insert_query+="\"name\":\""+operatorType+"-"+targetNodeType+"-"+labelNodeType+"-"+GNNMethod+"\",\n"
        sparql_ml_insert_query+="\"GMLTask\":{\"taskType\":\"kgnet:"+operatorType+"\",\"targetNode\":\""+targetNodeType+"\",\n"
        sparql_ml_insert_query+="\"labelNode\":\""+labelNodeType+"\",\"namedGraphURI\":\""+Constants.namedGraphURI_dic[kg_prefix]+"\",\n"
        sparql_ml_insert_query+="\"targetEdge\":\""+target_edge+"\",\"GNNMethod\":\""+GNNMethod+"\",\n"
        sparql_ml_insert_query+="\"datasetTypesFilePath\":\""+kg_types_path+"\",\n"
        sparql_ml_insert_query+="\"TOSG\":\""+TOSG_Patterns.d1h1+"\""
        if kg_prefix=="dblp":
            sparql_ml_insert_query += """,\n "targetNodeFilters":{"filter1":["<https://dblp.org/rdf/schema#yearOfPublication>", "?year","filter(xsd:integer(?year)<=1950)"] \n}\n"""
        sparql_ml_insert_query += "}\n})}"
        print("sparql_ml_insert_query=",sparql_ml_insert_query)
        ######################### write sparqlML query #########################
        insert_task_dict = gmlQueryParser(sparql_ml_insert_query).extractQueryStatmentsDict()
        model_info, transform_info, train_info = self.gml_insert_op.executeQuery(insert_task_dict)
        return model_info, transform_info, train_info
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
    # kgnet=KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',KGMeta_KG_URI='http://www.aifb.uni-karlsruhe.de')
    # model_info, transform_info, train_info=kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification,targetNodeType="aifb:ontology#Publication",labelNodeType="aifb:ontology#ResearchTopic",GNNMethod=Constants.GNN_Methods.Graph_SAINT)
    kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql')
    model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="dblp:Publication",
        labelNodeType="dblp:venue", GNNMethod=Constants.GNN_Methods.Graph_SAINT)
    print(model_info)