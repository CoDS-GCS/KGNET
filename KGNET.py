import Constants
import pandas as pd
from Constants import *
from SparqlMLaasService.GMLOperators import gmlInsertOperator,gmlInferenceOperator
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
from SparqlMLaasService.gmlRewriter import gmlQueryParser,gmlQueryRewriter
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from RDFEngineManager.UDF_Manager_Virtuoso import VirtuosoUDFManager
from pyvis.network import Network
color_palette = ["#ff6347", "#d8bfd8", "#66d8ff", "#ff7f50", "#ffa07a",
                         "#ffebcd", "#22d8d8", "#ffe4e1", "#c71585", "#ff8c00", "#ffb6c1", "#f08080", "#dc646c",
                         "#4686b8","#d2796e", "#9aed52", "#cd6c6c", "#32ed52", "#ff5499", "#f1e69c", "#9dff3f", "#01ff8f",
                         "#7b88de","#4682b4", "#d2691e", "#9acd32", "#20b2aa", "#cd5c5c", "#00008b", "#32cd32", "#8fbc8f",
                         "#800080","#9370db", "#9932cc", "#ff4500", "#ffa500", "#ffd700", "#0000cd", "#deb887", "#33ff00",
                         "#00ff7f","#dc143c", "#20eeff", "#00bfff", "#0000ff", "#a020f0", "#adff2f", "#ff6347", "#ff00ff",
                         "#1e90ff","#f0e68c", "#fd7811", "#dda0dd", "#90ee90", "#87ceeb", "#ff1493", "#7b68ee", "#ffa07a",
                         "#ee82ee","#7fffd4", "#ff69b4", "#ffc0cb", "#dc143c", "#20eeff", "#00bfff", "#0000ff", "#a020f0",
                         "#adff2f","#ff6347", "#ff00ff", "#1e90ff", "#f0e68c", "#f87811", "#dda0dd", "#90ee90", "#87ceeb",
                         "#ff1493","#7b68ee", "#ffa07a", "#ee82ee", "#7fffd4", "#ff69b4", "#ffc0cb"]
class KGNET():
    GML_Operator_Types = Constants.GML_Operator_Types
    GNN_Methods = Constants.GNN_Methods
    KGNET_Config = Constants.KGNET_Config
    KGs_prefixs_dic=Constants.KGs_prefixs_dic
    namedGraphURI_dic=Constants.namedGraphURI_dic
    utils=Constants.utils
    "KGNET system main class that automates GML the training and infernce pipeline"
    def __init__(self,KG_endpointUrl,KGMeta_endpointUrl="http://206.12.98.118:8890/sparql", KGMeta_KG_URI=Constants.KGNET_Config.KGMeta_IRI,KG_NamedGraph_IRI=None,KG_Prefix=None,KG_Prefix_URL=None):
        self.KGMeta_Governer = KGMeta_Governer(endpointUrl=KGMeta_endpointUrl, KGMeta_URI=KGMeta_KG_URI)
        self.VirtuosoUDFManager=VirtuosoUDFManager(host=KG_endpointUrl.split(":")[0].split("//")[-1])
        self.KG_sparqlEndpoint = sparqlEndpoint(endpointUrl=KG_endpointUrl)
        self.gml_insert_op = gmlInsertOperator(self.KGMeta_Governer,self.KG_sparqlEndpoint)
        self.KG_NamedGraph_URI = KG_NamedGraph_IRI
        if KG_Prefix:
            self.kg_Prefix = KG_Prefix
        else:
            self.kg_Prefix = KG_NamedGraph_IRI.split("//")[1].split(".")[0]

        if self.kg_Prefix not in Constants.KGs_prefixs_dic.keys():
            if KG_Prefix_URL:
                Constants.KGs_prefixs_dic[self.kg_Prefix]=KG_Prefix_URL
            else:
                Constants.KGs_prefixs_dic[self.kg_Prefix]=KG_NamedGraph_IRI
        if self.kg_Prefix not in Constants.namedGraphURI_dic.keys():
            Constants.namedGraphURI_dic[self.kg_Prefix]=KG_NamedGraph_IRI


    def uploadKG(self,ttl_file_url,name,description,domain):
        # self.VirtuosoUDFManager.uploadKG_ttl(ttl_file_url,self.KG_NamedGraph_URI)
        gid,_=self.KGMeta_Governer.insertKGMetadata(self.KG_sparqlEndpoint.endpointUrl,self.kg_Prefix,self.KG_NamedGraph_URI,name,description,domain)
        return gid
    def getKGNodeEdgeTypes(self,write_to_file=False,prefix=None):
        "returns a dataframe of KG triples node/edge types"
        if self.KG_NamedGraph_URI is None or self.KG_sparqlEndpoint is None:
            raise Exception("KG endpoint or the named-graph IRI is not defined")
        NamedGraph_URI=""
        if self.KG_NamedGraph_URI=="https://dblp2022.org":
            NamedGraph_URI = "http://dblp.org"
        else :
            NamedGraph_URI = self.KG_NamedGraph_URI

        NamedGraph_URI = self.KG_NamedGraph_URI
        predicate_types_query="select distinct ?p \n"
        predicate_types_query+= "" if NamedGraph_URI is None else  "from <"+NamedGraph_URI+"> \n"
        predicate_types_query+= " where {?s ?p ?o.}  "
        predicate_types_df=self.KG_sparqlEndpoint.executeSparqlquery(predicate_types_query)
        edge_types_lst=predicate_types_df["p"].apply(lambda x:x.replace("\"","")).tolist()
        KG_types_lst=[]
        for edgeType in edge_types_lst:
            if edgeType !="http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                s_type_query="select IF(STRLEN(xsd:string(?s_type))>0,xsd:string(?s_type),\""+edgeType.split("/")[-1]+"_Subj\") as ?s_type count(*) as ?count \n"
                s_type_query += "" if NamedGraph_URI is None else "from <" + NamedGraph_URI + "> \n"
                s_type_query += "where { ?s <"+edgeType+"> ?o. \n"
                s_type_query += " OPTIONAL {?s a ?s_type.} } \n group by  ?s_type \n order by desc(count(*))  limit 1000"

                o_type_query = "select IF(STRLEN(xsd:string(?o_type))>0,xsd:string(?o_type),\"" + edgeType.split("/")[-1] + "_Obj\") as ?o_type count(*) as ?count \n"
                o_type_query += "" if NamedGraph_URI is None else "from <" + NamedGraph_URI + "> \n"
                o_type_query += "where { ?s <" + edgeType + "> ?o. \n"
                o_type_query += " OPTIONAL {?o a ?o_type.} } \n group by  ?o_type \n order by desc(count(*))  limit 1000"

                s_types_df = self.KG_sparqlEndpoint.executeSparqlquery(s_type_query)
                o_types_df = self.KG_sparqlEndpoint.executeSparqlquery(o_type_query)
                KG_types_lst.append([s_types_df["s_type"].values[0].replace("\"","").split("/")[-1].split("#")[-1],edgeType.split("/")[-1].split("#")[-1],o_types_df["o_type"].values[0].replace("\"","").split("/")[-1].split("#")[-1]])
            else:
                KG_types_lst.append(["entity", edgeType.split("/")[-1].split("#")[-1],"type"])
        kg_types_df=pd.DataFrame(KG_types_lst,columns=["subject","predicate","object"])
        kg_types_df=kg_types_df.sort_values(by=["subject"])
        if write_to_file:
            if prefix is None:
                raise Exception("prefix is not provided to save the KG types file")
            kg_types_df.to_csv(Constants.KGNET_Config.datasets_output_path+ (self.namedGraphURI.split(".")[1] if prefix is None else prefix) +"_Types.csv",header=None, index=None)
        return kg_types_df
    def visualizeKG(self,types_df,width="100%",height="500px",Notebook=False,Directed=True):
        "Visualize  dataframe where subject and objects columns represents nodes and predicate column represnts the relations between nodes"
        nodes_lst = list(set(types_df["subject"].unique()).union(types_df["object"].unique()))
        # ################# nodes frequancy ##################
        # lst = types_df["subject"].tolist()
        # lst.extend(types_df["object"].tolist())
        # nodes_freq_dist=pd.DataFrame(lst, columns=["nodes"])["nodes"].value_counts().to_dict()
        # colors=[]
        # color_idx=1
        # for elem in nodes_lst:
        #     if nodes_freq_dist[elem]==1:
        #         colors.append(color_palette[0])
        #     else:
        #         colors.append(color_palette[color_idx])
        #         color_idx+=1
        ###################################################
        g = Network(width=width, height=height, notebook=Notebook, directed=Directed)
        g.add_nodes(nodes_lst, value=[50 for x in range(0, len(nodes_lst))],
                    title=nodes_lst,
                    label=nodes_lst,
                    color=color_palette[0:len(nodes_lst)]
                    # color=colors,
                    # shape=['box'] * len(nodes_lst)
                    )
        # G.add_nodes_from(,node_color="red")
        for ind in types_df.index:
            g.add_edge(types_df['subject'][ind], types_df['object'][ind], title=types_df['predicate'][ind])
        return g
    def train_GML(self,sparql_ml_insert_query):
        "Automates the GML training pipeline steps including: parsing the GML insert query ,identifying GML task type and attributes, sample task orianted subgraph, transform sampled subgraph into PYG dataset, train a GNN model, and save trained model meta-data into KGMeta KG "
        insert_task_dict = gmlQueryParser(sparql_ml_insert_query).extractQueryStatmentsDict()
        model_info, transform_info, train_info = self.gml_insert_op.executeQuery(insert_task_dict)
        return model_info, transform_info, train_info
    def getTargetEdgeTypeIRI(self,kg_prefix,target_edge_short):
        "return full edge URI for a given target edge type"
        edges_query=""" select   distinct ?p  
                       from <"""+ self.KG_NamedGraph_URI+""">
                       where { ?s ?p ?o.} limit 1000 """
        edges_df=self.KG_sparqlEndpoint.executeSparqlquery(edges_query)
        edges_df["p"] = edges_df["p"].apply(lambda x: str(x).replace("\"", ""))
        edges_df["p_lower"] = edges_df["p"].apply(lambda x: str(x).lower())
        target_edge_df=edges_df[edges_df["p_lower"].str.endswith(target_edge_short.lower())]
        return target_edge_df["p"].values[0]
    def train_GML(self,operatorType,GNNMethod,targetNodeType=None,labelNodeType=None,targetEdge=None):
        "Automates the GML training pipeline given the minimal task attributes steps including: write a SPARQL-ML insert query,  parsing the GML insert query ,identifying GML task type and attributes, sample task orianted subgraph, transform sampled subgraph into PYG dataset, train a GNN model, and save trained model meta-data into KGMeta KG "
        if self.kg_Prefix is not None:
            if operatorType==Constants.GML_Operator_Types.NodeClassification:
                # self.kg_Prefix=targetNodeType.split(":")[0]
                kg_types_path = Constants.KGNET_Config.datasets_output_path + self.kg_Prefix + "_Types.csv"
                kg_types_ds = pd.read_csv(kg_types_path, header=None)
                target_edge_df = kg_types_ds[(kg_types_ds[0].str.lower() == targetNodeType.split(":")[1].lower()) & (kg_types_ds[2].str.lower() == labelNodeType.split(":")[1].lower())]
                targetEdge = target_edge_df[1].values[0]
                targetEdge = self.getTargetEdgeTypeIRI(self.kg_Prefix, targetEdge)
            elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
                ""
                # self.kg_Prefix = targetEdge.split(":")[0]
        else:
            raise Exception("KG types dataset is not exist")

        print("targetEdge=",targetEdge)
        kg_types_path = Constants.KGNET_Config.datasets_output_path + self.kg_Prefix + "_Types.csv"
        ######################### write sparqlML query #########################
        if self.kg_Prefix in Constants.KGs_prefixs_dic.keys():
            sparql_ml_insert_query=" prefix "+ self.kg_Prefix+":<"+Constants.KGs_prefixs_dic[self.kg_Prefix]+"> \n"
        else:
            sparql_ml_insert_query = " prefix " + self.kg_Prefix + ":<" + Constants.namedGraphURI_dic[self.kg_Prefix] + "> \n"
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
        return int(model_info["task_uri"].split("-")[1]),int(model_info["model_uri"].split("-")[1]), {"model_info":model_info,"transform_info": transform_info, "train_info":train_info}
    def executeSPARQLMLInferenceQuery(self,query):
        """Automates the GML infernrence query execution steps including:
            * parse GML infernce query
            * rewwrite the infernce query into SAPRQL query
            * choose the GML model for the given task
            * execute the final SAPRQL query
        Args:
            query (str) : SPARQL-ML Infernce query as string ,i.e,
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
        Returns:
            pandas dataframe : query results in a dataframe
        """
        gmlInferenceOp=gmlInferenceOperator(self.KGMeta_Governer,self.KG_sparqlEndpoint)
        candidateSparqlQuery,kgDataQuery,kgTargetNodesQuery,kgmetaModelQuery,model_id=gmlInferenceOp.executeQuery(query)
        df_res=self.KG_sparqlEndpoint.executeSparqlquery(candidateSparqlQuery)
        # df_res=df_res.apply(lambda x: (x.str)[1:-1])
        return df_res,{"model_id":model_id,"candidateSparqlQuery":candidateSparqlQuery,"kgDataQuery":kgDataQuery,"kgTargetNodesQuery":kgTargetNodesQuery,"kgmetaModelQuery":kgmetaModelQuery}
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
    #################################### NC ############################
    # kgnet=KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',KG_NamedGraph_IRI='http://www.aifb.uni-karlsruhe.de')
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="aifb:Person",labelNodeType="aifb:ResearchGroup", GNNMethod=Constants.GNN_Methods.Graph_SAINT)
    kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='https://dblp2022.org',
                  KG_Prefix='dblp2022')
    model_info, transform_info, train_info = kgnet.train_GML(operatorType=KGNET.GML_Operator_Types.NodeClassification,
                                                             targetNodeType="dblp2022:Publication",
                                                             labelNodeType="dblp2022:publishedIn_Obj",
                                                             GNNMethod=KGNET.GNN_Methods.Graph_SAINT)

    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="aifb:Project",labelNodeType="aifb:Organization", GNNMethod=GNN_Methods.Graph_SAINT)
    # model_info, transform_info, train_info=  kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification,targetNodeType="aifb:Person",labelNodeType="aifb:ResearchGroup",GNNMethod=GNN_Methods.Graph_SAINT)

    # # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql')
    # # model_info, transform_info, train_info=kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification,targetNodeType="dblp:Publication",labelNodeType="dblp:venue",GNNMethod=GNN_Methods.Graph_SAINT)
    # # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="dblp:Publication",labelNodeType="dblp:venue", GNNMethod=GNN_Methods.Graph_SAINT)
    # print(model_info)
    #################################### LP ############################
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='http://www.aifb.uni-karlsruhe.de',KG_Prefix="aifb")
    # model_info, transform_info, train_info = kgnet.train_GML(
    #     operatorType=Constants.GML_Operator_Types.LinkPrediction, targetEdge="http://swrc.ontoware.org/ontology#publication", GNNMethod=GNN_Methods.RGCN)
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql')
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="dblp:Publication",
    #     labelNodeType="dblp:venue", GNNMethod=GNN_Methods.Graph_SAINT)
    # print(model_info)
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.LinkPrediction,targetEdge="http://swrc.ontoware.org/ontology#author",GNNMethod=GNN_Methods.MorsE)
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='https://dblp2022.org',
    #               KG_Prefix='dblp2022')
    # TargetEdge = "https://dblp.org/rdf/schema#authoredBy"
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=KGNET.GML_Operator_Types.LinkPrediction,
    #                                                          targetEdge=TargetEdge,
    #                                                          GNNMethod=KGNET.GNN_Methods.RGCN)
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
    inference_query_NC2="""
          prefix dblp2022:<https://dblp.org/rdf/schema#>
            prefix kgnet:<http://kgnet/>
            select ?Publication ?Title ?Org_Venue ?Pred_Venue
            from <https://dblp2022.org>
            where
            {
            ?Publication a dblp2022:Publication .
            ?Publication ?NodeClassifier ?Pred_Venue .            
            ?Publication  dblp2022:publishedIn ?Org_Venue .
            ?Publication  dblp2022:title ?Title .
            ?NodeClassifier a <kgnet:types/NodeClassifier>.
            ?NodeClassifier <kgnet:targetNode> dblp2022:Publication.
            ?NodeClassifier <kgnet:labelNode> dblp2022:publishedIn_Obj.
            }
            limit 10
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
                    ?LinkPredictor <kgnet:topK> 3 .
                    }
                    limit 300
                    offset 0
                """
    # kgnet=KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',KG_NamedGraph_IRI='https://dblp2022.org')
    # types_df = kgnet.getKGNodeEdgeTypes(write_to_file=True, prefix='dblp2022')
    # resDF,MetaQueries=kgnet.executeSPARQLMLInferenceQuery(inference_query_NC2)
    # print(resDF)
    # print(MetaQueries)
    #############################################3
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',KG_NamedGraph_IRI='http://www.RVO.net/', KG_Prefix="RVO")
    # kgnet.uploadKG(ttl_file_url="http://206.12.89.16/CodsData/KGNET/KGs/OBA.nt",name="Landenportaal-Rijksdienst voor Ondernemend Nederland (RVO)",description="information about countries around the globe with the objectives of these country monitors",domain="geometric")