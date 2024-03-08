import Constants
import pandas as pd
from Constants import *
from SparqlMLaasService.GMLOperators import gmlInferenceOperator,gmlInsertOperator
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
from SparqlMLaasService.gmlRewriter import gmlQueryParser
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from RDFEngineManager.UDF_Manager_Virtuoso import VirtuosoUDFManager
# from pyvis.network import Network
from statistics import mean
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
    "KGNET system main class that automates GML the training and infernce pipelines"
    def __init__(self,KG_endpointUrl,KGMeta_endpointUrl="http://206.12.98.118:8890/sparql", KGMeta_KG_URI=Constants.KGNET_Config.KGMeta_IRI,RDFEngine=Constants.RDFEngine.OpenlinkVirtuoso,KG_NamedGraph_IRI=None,KG_Prefix=None,KG_Prefix_URL=None):
        self.KGMeta_Governer = KGMeta_Governer(endpointUrl=KGMeta_endpointUrl, KGMeta_URI=KGMeta_KG_URI,RDFEngine=RDFEngine)
        self.VirtuosoUDFManager=VirtuosoUDFManager(host=KG_endpointUrl.split(":")[0].split("//")[-1])
        self.KG_sparqlEndpoint = sparqlEndpoint(endpointUrl=KG_endpointUrl,RDFEngine=RDFEngine)
        self.gml_insert_op = gmlInsertOperator(self.KGMeta_Governer,self.KG_sparqlEndpoint)
        self.KG_NamedGraph_URI = KG_NamedGraph_IRI
        self.RDFEngine=RDFEngine
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
        "returns a dataframe of KG triples node/edge types considers only single source and destinations node types per edge type"
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
        edge_types_lst=[elem.replace("<","").replace(">","") for elem in edge_types_lst]
        for edgeType in edge_types_lst:
            if edgeType !="http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                s_type_query="select (IF(STRLEN(xsd:string(?s_type_p))>0,xsd:string(?s_type_p),\""+edgeType.split("/")[-1]+"_Subj\") as ?s_type) (count(*) as ?count) \n"
                s_type_query += "" if NamedGraph_URI is None else "from <" + NamedGraph_URI + "> \n"
                s_type_query += "where { ?s <"+edgeType+"> ?o. \n"
                s_type_query += " OPTIONAL {?s a ?s_type_p.} } \n group by  ?s_type_p \n order by desc(count(*))  limit 1000"

                o_type_query = "select (IF(STRLEN(xsd:string(?o_type_p))>0,xsd:string(?o_type_p),\"" + edgeType.split("/")[-1] + "_Obj\") as ?o_type) (count(*) as ?count) \n"
                o_type_query += "" if NamedGraph_URI is None else "from <" + NamedGraph_URI + "> \n"
                o_type_query += "where { ?s <" + edgeType + "> ?o. \n"
                o_type_query += " OPTIONAL {?o a ?o_type_p.} } \n group by  ?o_type_p \n order by desc(count(*))  limit 1000"

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

    def getKGNodeEdgeTypes_V2(self, write_to_file=False, prefix=None):
        "returns a dataframe of KG triples node/edge types considring multi node type per edge"
        if self.KG_NamedGraph_URI is None or self.KG_sparqlEndpoint is None:
            raise Exception("KG endpoint or the named-graph IRI is not defined")
        NamedGraph_URI = ""
        if self.KG_NamedGraph_URI == "https://dblp2022.org":
            NamedGraph_URI = "http://dblp.org"
        else:
            NamedGraph_URI = self.KG_NamedGraph_URI

        NamedGraph_URI = self.KG_NamedGraph_URI
        predicate_types_query = "select distinct ?p \n"
        predicate_types_query += "" if NamedGraph_URI is None else "from <" + NamedGraph_URI + "> \n"
        predicate_types_query += " where {?s ?p ?o.}  "
        predicate_types_df = self.KG_sparqlEndpoint.executeSparqlquery(predicate_types_query)
        edge_types_lst = predicate_types_df["p"].apply(lambda x: x.replace("\"", "")).tolist()
        KG_types_lst = []
        edge_types_lst = [elem.replace("<", "").replace(">", "") for elem in edge_types_lst]
        for edgeType in edge_types_lst:
            if edgeType != "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                query="""select distinct ?stype ?otype
                            from <http://wikikg-v2>
                            where 
                            {
                            ?s <"""+edgeType+"""> ?o.
                            ?s a ?stype.
                            ?o a ?otype.
                            }"""
                types_df=self.KG_sparqlEndpoint.executeSparqlquery(query)
                for idx,row in types_df.iterrows():
                    KG_types_lst.append([row["stype"].replace("\"", "").split("/")[-1].split("#")[-1],
                                     edgeType.split("/")[-1].split("#")[-1],
                                     row["otype"].replace("\"", "").split("/")[-1].split("#")[-1]])
            else:
                KG_types_lst.append(["entity", edgeType.split("/")[-1].split("#")[-1], "type"])
        kg_types_df = pd.DataFrame(KG_types_lst, columns=["subject", "predicate", "object"])
        kg_types_df = kg_types_df.sort_values(by=["subject"])
        if write_to_file:
            if prefix is None:
                raise Exception("prefix is not provided to save the KG types file")
            kg_types_df.to_csv(Constants.KGNET_Config.datasets_output_path + (
                self.namedGraphURI.split(".")[1] if prefix is None else prefix) + "_Types.csv", header=None, index=None)
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
        edges_df["p"] = edges_df["p"].apply(lambda x: str(x).replace("\"", "").replace("<","").replace(">",""))
        edges_df["p_lower"] = edges_df["p"].apply(lambda x: str(x).lower())
        target_edge_df=edges_df[edges_df["p_lower"].str.endswith(target_edge_short.lower())]
        return target_edge_df["p"].values[0]
    def train_GML(self,operatorType,GNNMethod,targetNodeType=None,labelNodeType=None,targetEdge=None,TOSG_Pattern=None, epochs=None,emb_size=None,MinInstancesPerLabel=21):
        "Automates the GML training pipeline given the minimal task attributes steps including: write a SPARQL-ML insert query,  parsing the GML insert query ,identifying GML task type and attributes, sample task orianted subgraph, transform sampled subgraph into PYG dataset, train a GNN model, and save trained model meta-data into KGMeta KG "
        if self.kg_Prefix is not None:
            if operatorType==Constants.GML_Operator_Types.NodeClassification and targetEdge is None:
                # self.kg_Prefix=targetNodeType.split(":")[0]
                kg_types_path = Constants.KGNET_Config.datasets_output_path + self.kg_Prefix + "_Types.csv"
                kg_types_ds = pd.read_csv(kg_types_path, header=None)
                tnType=targetNodeType.split(":")[-1].lower()
                lnType=labelNodeType.split(":")[-1].lower()
                target_edge_df = kg_types_ds[(kg_types_ds[0].str.lower() == tnType ) & (kg_types_ds[2].str.lower() == lnType)]
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
            sparql_ml_insert_query+="\"name\":\""+operatorType+">"+self.kg_Prefix+">"+targetNodeType+">"+("None" if labelNodeType is None else labelNodeType) +">"+GNNMethod+"\",\n"
        elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
            sparql_ml_insert_query += "\"name\":\"" + operatorType +">"+self.kg_Prefix+ ">" +targetEdge.split("/")[-1] + ">" + GNNMethod + "\",\n"
        sparql_ml_insert_query+="\"GMLTask\":{\"taskType\":\"kgnet:"+operatorType+"\",\n"
        if targetNodeType is not None:
            sparql_ml_insert_query+="\"targetNode\":\""+targetNodeType+"\",\n"
        if labelNodeType is not None:
            sparql_ml_insert_query+="\"labelNode\":\""+labelNodeType+"\",\n"
        if MinInstancesPerLabel is not None:
            sparql_ml_insert_query += "\"MinInstancesPerLabel\":\"" + str(MinInstancesPerLabel) + "\",\n"


        sparql_ml_insert_query+="\"namedGraphURI\":\""+ self.KG_NamedGraph_URI+"\",\n"
        sparql_ml_insert_query += "\"namedGraphPrefix\":\"" + self.kg_Prefix + "\",\n"
        sparql_ml_insert_query+="\"targetEdge\":\""+targetEdge+"\",\"GNNMethod\":\""+GNNMethod+"\",\n"
        sparql_ml_insert_query+="\"datasetTypesFilePath\":\""+kg_types_path+"\",\n"
        if epochs is not None:
            sparql_ml_insert_query += "\"epochs\":" + str(epochs)+ ","
        if emb_size is not None:
            sparql_ml_insert_query += "\"embSize\":" + str(emb_size)+ ","
        if operatorType == Constants.GML_Operator_Types.NodeClassification:
            sparql_ml_insert_query+="\"TOSG\":\""+(TOSG_Patterns.d1h1 if TOSG_Pattern is None else TOSG_Pattern)+"\""
        elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
            sparql_ml_insert_query+="\"TOSG\":\""+(TOSG_Patterns.d2h1 if TOSG_Pattern is None else TOSG_Pattern)+"\""
        if operatorType == Constants.GML_Operator_Types.NodeClassification and self.kg_Prefix=="dblp" and targetNodeType=="dblp:Publication":
            sparql_ml_insert_query += """,\n "targetNodeFilters":{"filter1":["<https://dblp.org/rdf/schema#yearOfPublication>", "?year","filter(xsd:integer(?year)>=2019)"] \n}\n"""
        sparql_ml_insert_query += "}\n})}"
        print("sparql_ml_insert_query=",sparql_ml_insert_query)
        ######################### write sparqlML query #########################
        insert_task_dict = gmlQueryParser(sparql_ml_insert_query).extractQueryStatmentsDict()
        model_info, transform_info, train_info = self.gml_insert_op.executeQuery(insert_task_dict)
        return model_info["task_uri"].split("/")[-1],model_info["model_uri"].split("/")[-1], {"model_info":model_info,"transform_info": transform_info, "train_info":train_info}
    def executeSPARQLMLInferenceQuery(self,query,in_pipline=True):
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
        gmlInferenceOp = gmlInferenceOperator(self.KGMeta_Governer, self.KG_sparqlEndpoint)
        if in_pipline==True:
            res_df,exectuted_Queries,time_sec = gmlInferenceOp.executeQuery(query,in_pipline)
            return res_df, exectuted_Queries,time_sec
        else:
            df_res,candidateSparqlQuery,kgDataQuery,kgTargetNodesQuery,kgmetaModelQuery,model_ids_lst,time_sec=gmlInferenceOp.executeQuery(query,in_pipline)
            # df_res=df_res.apply(lambda x: (x.str)[1:-1])
            return df_res,{"model_ids_lst":model_ids_lst,"candidateSparqlQuery":candidateSparqlQuery,"kgDataQuery":kgDataQuery,"InferenceTargetNodeQueries":kgTargetNodesQuery,"kgmetaModelQuery":kgmetaModelQuery},time_sec
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
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='https://dblp2022.org',
    #               KG_Prefix='dblp2022')
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=KGNET.GML_Operator_Types.NodeClassification,
    #                                                          targetNodeType="dblp2022:Publication",
    #                                                          labelNodeType="dblp2022:publishedIn_Obj",
    #                                                          GNNMethod=KGNET.GNN_Methods.Graph_SAINT)

    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="aifb:Project",labelNodeType="aifb:Organization", GNNMethod=GNN_Methods.Graph_SAINT)
    # model_info, transform_info, train_info=  kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification,targetNodeType="aifb:Person",labelNodeType="aifb:ResearchGroup",GNNMethod=GNN_Methods.Graph_SAINT)

    # # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql')
    # # model_info, transform_info, train_info=kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification,targetNodeType="dblp:Publication",labelNodeType="dblp:venue",GNNMethod=GNN_Methods.Graph_SAINT)
    # # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="dblp:Publication",labelNodeType="dblp:venue", GNNMethod=GNN_Methods.Graph_SAINT)
    # print(model_info)

    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='http://wikikg-v2',KG_Prefix='WikiKG2015_v2')
    kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='http://dblp.org', KG_Prefix='dblp')
    # types_df = kgnet.getKGNodeEdgeTypes_V2(write_to_file=True, prefix='WikiKG2015_v2')
    # types_df = types_df[(~types_df["object"].str.endswith("_Obj")) | (types_df["object"].str.endswith("publishedIn_Obj"))]
    # types_df

    # TargetEdge = "http://www.wikidata.org/entity/P166" # WikidataKG award received
    # label_type="science_or_engineering_award"

    # TargetEdge = "http://www.wikidata.org/entity/P101"  # WikidataKG work field# area_of_mathematics
    # label_type = "area_of_mathematics"

    # TargetEdge = "http://www.wikidata.org/entity/P27"  # citizenship
    # label_type="country"

    # TargetEdge = "http://www.wikidata.org/entity/P106"  # profession
    # label_type = "occupation"

    #TargetEdge = "http://www.wikidata.org/entity/P108" # Employeer
    #"http://www.wikidata.org/entity/Q3571662" Yan Lucn

    TargetEdge = "dblp:publishedIn"  # profession
    targetNodeType = "dblp:Publication"
    MinInstancesPerLabel = 2260
    # TargetEdge = "https://dblp.org/Affaliation_Country"
    # targetNodeType = "dblp:Person"
    # MinInstancesPerLabel = 115

    # for epoch in range(5,31,5):
    #     for e_size in range(64, 128, 32):
    #         model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType=targetNodeType,labelNodeType=None,targetEdge=TargetEdge, GNNMethod=GNN_Methods.Graph_SAINT,TOSG_Pattern=TOSG_Patterns.d1h1,epochs=epoch,emb_size=e_size,MinInstancesPerLabel=MinInstancesPerLabel)
    #################################### LP ######
    # ######################
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='http://www.aifb.uni-karlsruhe.de',KG_Prefix="aifb")
    # model_info, transform_info, train_info = kgnet.train_GML(
    #     operatorType=Constants.GML_Operator_Types.LinkPrediction, targetEdge="http://swrc.ontoware.org/ontology#publication", GNNMethod=GNN_Methods.RGCN)
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='https://dblp2022.org',KG_Prefix='dblp2022')
    # kgnet = KGNET(KG_endpointUrl='http://206.12.100.35:5820/kgnet_kgs/query',KGMeta_endpointUrl='http://206.12.100.35:5820/kgnet_kgs/query', KG_NamedGraph_IRI='https://dblp2022.org',KG_Prefix='dblp2022',RDFEngine=Constants.RDFEngine.stardog)
    # types_df = kgnet.getKGNodeEdgeTypes(write_to_file=True, prefix='dblp2022')
    # task_id,mode_id,model_info_dict = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType="dblp2022:Publication",labelNodeType="dblp2022:publishedIn_Obj", GNNMethod=GNN_Methods.Graph_SAINT)
    # task_id='tid-0000025'
    # df = kgnet.KGMeta_Governer.getGMLTaskModelsBasicInfoByID(task_id)
    # print(model_info_dict)
    #model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.LinkPrediction,targetEdge="http://swrc.ontoware.org/ontology#author",GNNMethod=GNN_Methods.MorsE)
    #kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='https://dblp2022.org',KG_Prefix='dblp2022')
    kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='http://wikikg-v2',KG_Prefix='WikiKG2015_v2')
    # KGNET.inference_path = KGNET.KGNET_Config.datasets_output_path + 'Inference/'
    # KGNET.KGNET_Config.trained_model_path = KGNET.KGNET_Config.datasets_output_path + 'trained_models/'
    #
    # KGNET.KGNET_Config.GML_API_URL = "http://206.12.102.12:64647/"
    # KGNET.KGNET_Config.fileStorageType = FileStorageType.remoteFileStore
    # #########remoteFileStore######
    # KGNET.KGNET_Config.GML_ModelManager_URL = "http://206.12.102.12"
    # KGNET.KGNET_Config.GML_ModelManager_PORT = "64648"
    # KGNET.KGNET_Config.KGMeta_IRI = "http://kgnet/"
    # KGNET.KGNET_Config.KGMeta_endpoint_url = "http://206.12.98.118:8890/sparql/"
    #
    # TargetEdge = "https://dblp.org/rdf/schema#authoredBy"
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=KGNET.GML_Operator_Types.LinkPrediction,
    #                                                          targetEdge=TargetEdge,
    #                                                          GNNMethod=KGNET.GNN_Methods.RGCN)

    # TargetEdge = "http://www.wikidata.org/entity/P166" # WikidataKG award received
    # TargetEdge= "http://www.wikidata.org/entity/P101" # WikidataKG field of work
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
                     prefix dblp2022:<https://dblp.org/rdf/schema#>
                    prefix kgnet:<https://kgnet/>
                    select ?publication ?author                    
                    from <https://dblp2022.org>
                    where {
                    ?publication a dblp2022:Publication.
                    ?publication ?LinkPredictor ?author.
                    ?LinkPredictor  a <kgnet:types/LinkPredictor>.
                    ?LinkPredictor  <kgnet:targetEdge> "https://dblp.org/rdf/schema#authoredBy" .
                    ?LinkPredictor <kgnet:GNNMethod> "MorsE" .
                    ?LinkPredictor <kgnet:topK> 3 .
                    }
                    limit 300
                    offset 0
                """
    inference_query_wikidata_workField_NC = """
                    prefix wiki:<http://www.wikidata.org/entity/>
                    prefix kgnet:<http://kgnet/>
                    select ?human ?work
                    from <http://wikikg-v2>
                    where
                    {
                    ?human wiki:P101 ?w.
                    ?human a "human".
                    ?w a "area_of_mathematics".
                    ?human ?NodeClassifier ?work.
                    ?NodeClassifier a <kgnet:types/NodeClassifier>.
                    ?NodeClassifier <kgnet:targetNode> "human".
                    ?NodeClassifier <kgnet:labelNode> "area_of_mathematics".
                    ?NodeClassifier <kgnet:targetEdge> "http://www.wikidata.org/entity/P101".
                    }
                    limit 100
                """
    inference_query_wikidata_award_NC = """
                        prefix wiki:<http://www.wikidata.org/entity/>
                        prefix kgnet:<http://kgnet/>
                        select ?human ?award
                        from <http://wikikg-v2>
                        where
                        {
                        ?human wiki:P166 ?w.
                        ?human a "human".
                        ?w a "science_or_engineering_award".
                        ?human ?NodeClassifier ?award.
                        ?NodeClassifier a <kgnet:types/NodeClassifier>.
                        ?NodeClassifier <kgnet:targetNode> "human".
                        ?NodeClassifier <kgnet:labelNode> "science_or_engineering_award".
                        ?NodeClassifier <kgnet:targetEdge> "http://www.wikidata.org/entity/P166".
                        }
                        limit 100
                    """
    inference_query_wikidata_award_workField_NC = """
                            prefix wiki:<http://www.wikidata.org/entity/>
                            prefix kgnet:<http://kgnet/>
                            select ?human ?univ ?pred_award ?pred_work
                            from <http://wikikg-v2>
                            where
                            {
                            ?human wiki:P166 ?award.
                            ?human a "human".
                            ?human wiki:P69 ?univ .
                            
                            ?award a "science_or_engineering_award".
                            ?human ?NodeClassifier ?pred_award.
                            ?NodeClassifier a <kgnet:types/NodeClassifier>.
                            ?NodeClassifier <kgnet:targetNode> "human".
                            ?NodeClassifier <kgnet:labelNode> "science_or_engineering_award".
                            ?NodeClassifier <kgnet:targetEdge> "http://www.wikidata.org/entity/P166".
                            
                            ?human wiki:P101 ?work.
                            ?work a "area_of_mathematics".
                            ?human ?NodeClassifier2 ?pred_work.
                            ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
                            ?NodeClassifier2 <kgnet:targetNode> "human".
                            ?NodeClassifier2 <kgnet:labelNode> "area_of_mathematics".
                            ?NodeClassifier2 <kgnet:targetEdge> "http://www.wikidata.org/entity/P101".                        
                            }
                            limit 100
                        """
    inference_query_wikidata_award_workField_univ_NC = """
                               prefix wiki:<http://www.wikidata.org/entity/>
                               prefix kgnet:<http://kgnet/>
                               select ?human ?univ_label ?pred_awardLabel ?pred_workFieldLabel 
                               from <http://wikikg-v2>
                               where
                               {
                                    ?pred_award_ent  <http://www.w3.org/2000/01/rdf-schema#label> ?pred_awardLabel.
                                    ?pred_work_ent  <http://www.w3.org/2000/01/rdf-schema#label> ?pred_workFieldLabel.
                                    filter(?pred_awardLabel='Royal Medal').
                                    filter(?pred_workFieldLabel='number theory').
                                    {
                                       select distinct ?human ?univ_label (IRI(?pred_award) as ?pred_award_ent) (IRI(?pred_work) as ?pred_work_ent)
                                       where
                                       {
                                           ?human wiki:P166 ?award.
                                           ?human a "human".
                                           ?human wiki:P69 ?univ .
                                           ?univ <http://www.w3.org/2000/01/rdf-schema#label> ?univ_label.                                           
                                           #optional {?univ <http://www.w3.org/2000/01/rdf-schema#label> ?univ_label. }
                                                       
                                           ?award a "science_or_engineering_award".
                                           ?human ?NodeClassifier ?pred_award.
                                           ?NodeClassifier a <kgnet:types/NodeClassifier>.
                                           ?NodeClassifier <kgnet:targetNode> "human".
                                           ?NodeClassifier <kgnet:labelNode> "science_or_engineering_award".
                                           ?NodeClassifier <kgnet:targetEdge> "http://www.wikidata.org/entity/P166".
            
                                           ?human wiki:P101 ?work.
                                           ?work a "area_of_mathematics".
                                           ?human ?NodeClassifier2 ?pred_work.
                                           ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
                                           ?NodeClassifier2 <kgnet:targetNode> "human".
                                           ?NodeClassifier2 <kgnet:labelNode> "area_of_mathematics".
                                           ?NodeClassifier2 <kgnet:targetEdge> "http://www.wikidata.org/entity/P101".                        
                                       }
                                       #limit 100
                                    }
                                }
                           """
    inference_query_wikidata_award_workField_univ_NC_v2 = """
                                   prefix wiki:<http://www.wikidata.org/entity/>
                                   prefix kgnet:<http://kgnet/>
                                   select distinct ?human  ?pred_awardLabel ?pred_workFieldLabel 
                                   from <http://wikikg-v2>
                                   where
                                   {                 
                                       ?human a "human".                                                                              
                                       ?human wiki:P69 ?univ .
                                       optional {?univ <http://www.w3.org/2000/01/rdf-schema#label> ?univ_label }.
                                       ?human wiki:P166 ?award.
                                       ?award a "science_or_engineering_award".
                                       ?human wiki:P101 ?work.
                                       ?work a "area_of_mathematics".
                                       
                                       ?human ?NodeClassifier ?pred_award.
                                       ?NodeClassifier a <kgnet:types/NodeClassifier>.
                                       ?NodeClassifier <kgnet:targetNode> "human".
                                       ?NodeClassifier <kgnet:labelNode> "science_or_engineering_award".
                                       ?NodeClassifier <kgnet:targetEdge> "http://www.wikidata.org/entity/P166".
                                       
                                       ?human ?NodeClassifier2 ?pred_work.
                                       ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
                                       ?NodeClassifier2 <kgnet:targetNode> "human".
                                       ?NodeClassifier2 <kgnet:labelNode> "area_of_mathematics".
                                       ?NodeClassifier2 <kgnet:targetEdge> "http://www.wikidata.org/entity/P101".
                                       
                                       ?pred_award  <http://www.w3.org/2000/01/rdf-schema#label> ?pred_awardLabel.
                                       ?pred_work  <http://www.w3.org/2000/01/rdf-schema#label> ?pred_workFieldLabel. 
                                       filter(?pred_awardLabel='Royal Medal').
                                       #filter(?pred_awardLabel='Fields Medal').
                                       filter(?pred_workFieldLabel='number theory').                        
                                  }
                                #limit 100                                    
                               """
    inference_query_wikidata_award_univ_NC_v2 = """
                                      prefix wiki:<http://www.wikidata.org/entity/>
                                      prefix kgnet:<http://kgnet/>
                                      select distinct ?human  ?pred_awardLabel  
                                      from <http://wikikg-v2>
                                      where
                                      {                 
                                          ?human a "human".
                                          ?human wiki:P69 ?univ .
                                          ?univ <http://www.w3.org/2000/01/rdf-schema#label> ?univ_label.
                                          
                                          ?human wiki:P166 ?award.
                                          ?award a "science_or_engineering_award".
                                          ?human ?NodeClassifier ?pred_award.
                                          ?NodeClassifier a <kgnet:types/NodeClassifier>.
                                          ?NodeClassifier <kgnet:targetNode> "human".
                                          ?NodeClassifier <kgnet:labelNode> "science_or_engineering_award".
                                          ?NodeClassifier <kgnet:targetEdge> "http://www.wikidata.org/entity/P166".

                                          ?human wiki:P101 ?work.
                                          ?work a "area_of_mathematics".                                          

                                          ?pred_award  <http://www.w3.org/2000/01/rdf-schema#label> ?pred_awardLabel. 
                                          filter(?pred_awardLabel='Royal Medal').                                                                 
                                     }
                                   #limit 100                                    
                                  """
    inference_query_wikidata_workField_univ_NC_v2 = """
                                          prefix wiki:<http://www.wikidata.org/entity/>
                                          prefix kgnet:<http://kgnet/>
                                          select distinct ?human  ?pred_workFieldLabel 
                                          from <http://wikikg-v2>
                                          where
                                          {                 
                                              ?human a "human".
                                              ?human wiki:P69 ?univ .
                                              ?univ <http://www.w3.org/2000/01/rdf-schema#label> ?univ_label.

                                              ?human wiki:P166 ?award.
                                              ?award a "science_or_engineering_award".
                                              
                                              ?human wiki:P101 ?work.
                                              ?work a "area_of_mathematics".
                                              ?human ?NodeClassifier2 ?pred_work.
                                              ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
                                              ?NodeClassifier2 <kgnet:targetNode> "human".
                                              ?NodeClassifier2 <kgnet:labelNode> "area_of_mathematics".
                                              ?NodeClassifier2 <kgnet:targetEdge> "http://www.wikidata.org/entity/P101".

                                              ?pred_work  <http://www.w3.org/2000/01/rdf-schema#label> ?pred_workFieldLabel. 
                                              filter(?pred_workFieldLabel='number theory').                        
                                         }
                                       #limit 100                                    
                                      """
    nested_Query=""" select  (count(*) as ?s)
                from <http://wikikg-v2>
                where
                {
                    ?s ?p ?o.
                    {
                        select distinct ?s
                        where
                        {
                           ?s <http://www.wikidata.org/entity/P69> ?o.
                           ?s a "human".
                           ?o <http://schema.org/description> ?ol.
                        }
                    }
                } 
                """
    inference_MQuery_NC = """
               prefix dblp2022:<https://dblp.org/rdf/schema#>
               prefix kgnet:<http://kgnet/>
               select ?Publication ?Title ?Org_Venue ?Pred_Venue
               from <https://dblp2022.org>
               where
               {
               ?Publication a dblp2022:Publication .
               ?Publication  dblp2022:title ?Title .
               ?Publication ?authored_by ?Author .
               ?Publication  dblp2022:publishedIn ?Org_Venue .
               ?Auhor ?aff ?Org_Aff_Country .

               ?Publication ?NodeClassifier ?Pred_Venue .
               ?NodeClassifier a <kgnet:types/NodeClassifier>.
               ?NodeClassifier <kgnet:targetNode> dblp2022:Publication.
               ?NodeClassifier <kgnet:labelNode> dblp2022:publishedIn_Obj.

               ?Auhor ?NodeClassifier2 ?Pred_Aff_Country .
               ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
               ?NodeClassifier2 <kgnet:targetNode> dblp2022:Author.
               ?NodeClassifier2 <kgnet:labelNode> dblp2022:Country_Obj.
               }
               limit 10
       """
    inference_MQuery_dblp2022_NC_LP = """
               prefix dblp2022:<https://dblp.org/rdf/schema#>
               prefix kgnet:<http://kgnet/>
               select ?Publication ?Title ?Org_Venue ?Pred_Venue  ?Org_author ?Pred_author
               from <https://dblp2022.org>
               where
               {
                   ?Publication a dblp2022:Publication .
                   ?Publication  dblp2022:title ?Title .
                   ?Publication  dblp2022:publishedIn ?Org_Venue .
                   ?Publication dblp2022:authoredBy ?Org_author .

                   ?Publication ?NodeClassifier ?Pred_Venue .
                   ?NodeClassifier a <kgnet:types/NodeClassifier>.
                   ?NodeClassifier <kgnet:targetNode> dblp2022:Publication.
                   ?NodeClassifier <kgnet:labelNode> dblp2022:publishedIn_Obj.

                    ?Publication ?LinkPredictor ?Pred_author.
                    ?LinkPredictor  a <kgnet:types/LinkPredictor>.
                    ?LinkPredictor  <kgnet:targetEdge> "https://dblp.org/rdf/schema#authoredBy" .
                    ?LinkPredictor <kgnet:GNNMethod> "MorsE" .
                    ?LinkPredictor <kgnet:topK> 3 .
               }
               limit 10
           """
    inference_MQuery_dblp_NC_venue_aff=""" 
        prefix dblp:<https://dblp.org/rdf/schema#>
        select distinct  ?Publication ?venue ?country
        from <http://dblp.org>
        where
        {
            ?Publication a dblp:Publication .
            ?Publication dblp:publishedIn ?venue.
            ?Publication dblp:yearOfPublication ?year .
            ?Publication dblp:authoredBy ?Author.
            
            ?Author a dblp:Person.
            ?Author <https://dblp.org/Affaliation_Country> ?country.
             
            ?Publication ?NodeClassifier ?Pred_Venue .
            ?NodeClassifier a <kgnet:types/NodeClassifier>.
            ?NodeClassifier <kgnet:targetNode> dblp:Publication.
            ?NodeClassifier <kgnet:targetEdge> dblp:publishedIn.
            
            ?Author ?NodeClassifier2 ?Pred_aff_country .
            ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
            ?NodeClassifier2 <kgnet:targetNode> dblp:Person.
            ?NodeClassifier2 <kgnet:targetEdge> "https://dblp.org/Affaliation_Country".             
            filter(xsd:integer(?year)=2021).
            filter(?country in ("germany","china")).            
            filter(str(?Pred_aff_country) in ("germany","china")).
            filter(?Pred_Venue in ("AAAI","VLDB","ACM")).
        }       
        """
    inference_MQuery_dblp_NC_venue_aff_v2 = """ 
            prefix dblp:<https://dblp.org/rdf/schema#>
            select distinct  ?venue count(*)
            from <http://dblp.org>
                where
                {
                    ?Publication a dblp:Publication .
                    ?Publication <https://dblp.org/rdf/schema#publishedIn> ?venue.
                    ?Publication <https://dblp.org/rdf/schema#authoredBy> ?Author.
                    ?Author a <https://dblp.org/rdf/schema#Person>.
                    ?Author <https://dblp.org/Affaliation_Country> ?aff_country.                

                    ?Publication ?NodeClassifier ?Pred_Venue .
                    ?NodeClassifier a <kgnet:types/NodeClassifier>.
                    ?NodeClassifier <kgnet:targetNode> dblp:Publication.
                    ?NodeClassifier <kgnet:targetEdge> dblp:publishedIn.

                    ?Author ?NodeClassifier2 ?Pred_aff_country .
                    ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
                    ?NodeClassifier2 <kgnet:targetNode> dblp:Person.
                    ?NodeClassifier2 <kgnet:targetEdge> "https://dblp.org/Affaliation_Country". 
                    filter (?Pred_aff_country in ("china","usa","germany"))
                    filter (?Pred_Venue in ("VLDB","ACM"))
                }
            group by ?venue
            having (count(*)>1)
            order by DESC(count(*))        
            """
    #############################
    inference_query_wikidata_Citizenship_Profession_univ_NC_v2 = """
                                       prefix wiki:<http://www.wikidata.org/entity/>
                                       prefix kgnet:<http://kgnet/>
                                       select distinct ?human  ?pred_nationality_label ?pred_profession_label 
                                       from <http://wikikg-v2>
                                       where
                                       {                 
                                           ?human a "human".                                         
                                           ?human wiki:P27 ?nationality.
                                           ?nationality a "country".
                                           ?human wiki:P106 ?profession.
                                           ?profession a "occupation".

                                           ?human ?NodeClassifier ?pred_nationality.
                                           ?NodeClassifier a <kgnet:types/NodeClassifier>.
                                           ?NodeClassifier <kgnet:targetNode> "human".
                                           ?NodeClassifier <kgnet:labelNode> "country".
                                           ?NodeClassifier <kgnet:labelValue> "Nederland".
                                           ?NodeClassifier <kgnet:targetEdge> "http://www.wikidata.org/entity/P27".

                                           ?human ?NodeClassifier2 ?pred_profession.
                                           ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
                                           ?NodeClassifier2 <kgnet:targetNode> "human".
                                           ?NodeClassifier2 <kgnet:labelNode> "occupation".
                                           ?NodeClassifier2 <kgnet:labelValue> "baseball player".
                                           ?NodeClassifier2 <kgnet:targetEdge> "http://www.wikidata.org/entity/P106".

                                           ?pred_nationality  <http://www.w3.org/2004/02/skos/core#altLabel> ?pred_nationality_label.
                                           #?pred_profession  <http://www.w3.org/2004/02/skos/core#altLabel> ?pred_profession_label.
                                           ?pred_profession  <http://www.w3.org/2000/01/rdf-schema#label> ?pred_profession_label. 
                                           filter(?pred_nationality_label='Nederland').
                                           filter(?pred_profession_label='baseball player').                         
                                      }
                                    #limit 100                                    
                                   """
    # kgnet=KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',KG_NamedGraph_IRI='https://dblp2022.org',KGMeta_endpointUrl="http://206.12.98.118:8890/sparql",RDFEngine=RDFEngine.OpenlinkVirtuoso)
    # kgnet = KGNET(KG_endpointUrl="http://206.12.100.35:5820/kgnet_kgs/query",KGMeta_endpointUrl="http://206.12.100.35:5820/kgnet_kgs/query", KG_NamedGraph_IRI='https://dblp2022.org',RDFEngine=RDFEngine.stardog)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_workField_NC)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_award_NC)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_award_workField_univ_NC)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_award_univ_NC_v2)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_workField_univ_NC_v2)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_Citizenship_Profession_univ_NC_v2)
    exec_time=[]
    true_prad_lst,not_pred_lst,false_pred_lst=[],[],[]
    field_medal_number_theory_Kg_humans=['http://www.wikidata.org/entity/Q1398727','http://www.wikidata.org/entity/Q211041','http://www.wikidata.org/entity/Q212063','http://www.wikidata.org/entity/Q220402','http://www.wikidata.org/entity/Q295981','http://www.wikidata.org/entity/Q310769','http://www.wikidata.org/entity/Q333538','http://www.wikidata.org/entity/Q333968','http://www.wikidata.org/entity/Q334045','http://www.wikidata.org/entity/Q369561','http://www.wikidata.org/entity/Q77137']
    royal_medal_number_theory_Kg_humans=['http://www.wikidata.org/entity/Q184337', 'http://www.wikidata.org/entity/Q184433','http://www.wikidata.org/entity/Q295981', 'http://www.wikidata.org/entity/Q310781','http://www.wikidata.org/entity/Q353426']
    nethierland_athelete_Kg_humans = ["http://www.wikidata.org/entity/Q128912","http://www.wikidata.org/entity/Q713390","http://www.wikidata.org/entity/Q152125","http://www.wikidata.org/entity/Q2485779","http://www.wikidata.org/entity/Q2491603","http://www.wikidata.org/entity/Q2520376","http://www.wikidata.org/entity/Q258475","http://www.wikidata.org/entity/Q441270","http://www.wikidata.org/entity/Q443882","http://www.wikidata.org/entity/Q697176","http://www.wikidata.org/entity/Q4040442","http://www.wikidata.org/entity/Q313877","http://www.wikidata.org/entity/Q839952","http://www.wikidata.org/entity/Q1852089","http://www.wikidata.org/entity/Q1993932","http://www.wikidata.org/entity/Q2676560","http://www.wikidata.org/entity/Q2933814","http://www.wikidata.org/entity/Q5171587","http://www.wikidata.org/entity/Q832258","http://www.wikidata.org/entity/Q462694","http://www.wikidata.org/entity/Q1071546","http://www.wikidata.org/entity/Q3354942","http://www.wikidata.org/entity/Q465865","http://www.wikidata.org/entity/Q1903135","http://www.wikidata.org/entity/Q2623735","http://www.wikidata.org/entity/Q2624601","http://www.wikidata.org/entity/Q2745460","http://www.wikidata.org/entity/Q15894148","http://www.wikidata.org/entity/Q19848886","http://www.wikidata.org/entity/Q264515","http://www.wikidata.org/entity/Q2134424","http://www.wikidata.org/entity/Q2221157","http://www.wikidata.org/entity/Q2413132","http://www.wikidata.org/entity/Q2423097","http://www.wikidata.org/entity/Q936526","http://www.wikidata.org/entity/Q4661093","http://www.wikidata.org/entity/Q15442354","http://www.wikidata.org/entity/Q6183233","http://www.wikidata.org/entity/Q7155101","http://www.wikidata.org/entity/Q17627341","http://www.wikidata.org/entity/Q15605374","http://www.wikidata.org/entity/Q2355008","http://www.wikidata.org/entity/Q2582703","http://www.wikidata.org/entity/Q2788007","http://www.wikidata.org/entity/Q2325683","http://www.wikidata.org/entity/Q2329623","http://www.wikidata.org/entity/Q142733","http://www.wikidata.org/entity/Q2301217","http://www.wikidata.org/entity/Q2363382","http://www.wikidata.org/entity/Q2475656","http://www.wikidata.org/entity/Q1983169","http://www.wikidata.org/entity/Q2000612","http://www.wikidata.org/entity/Q2106831","http://www.wikidata.org/entity/Q2516725","http://www.wikidata.org/entity/Q2661063","http://www.wikidata.org/entity/Q2864788","http://www.wikidata.org/entity/Q2238626","http://www.wikidata.org/entity/Q2238989","http://www.wikidata.org/entity/Q2241930","http://www.wikidata.org/entity/Q2286746","http://www.wikidata.org/entity/Q2289901","http://www.wikidata.org/entity/Q3027581","http://www.wikidata.org/entity/Q3007712","http://www.wikidata.org/entity/Q2749597","http://www.wikidata.org/entity/Q2933246","http://www.wikidata.org/entity/Q5326298","http://www.wikidata.org/entity/Q4784578","http://www.wikidata.org/entity/Q5076023","http://www.wikidata.org/entity/Q153801","http://www.wikidata.org/entity/Q63325","http://www.wikidata.org/entity/Q686752","http://www.wikidata.org/entity/Q3313872","http://www.wikidata.org/entity/Q238857","http://www.wikidata.org/entity/Q239758","http://www.wikidata.org/entity/Q368492","http://www.wikidata.org/entity/Q706125","http://www.wikidata.org/entity/Q710476","http://www.wikidata.org/entity/Q1867846","http://www.wikidata.org/entity/Q1893161","http://www.wikidata.org/entity/Q2018841","http://www.wikidata.org/entity/Q7802917","http://www.wikidata.org/entity/Q1025876","http://www.wikidata.org/entity/Q2279436","http://www.wikidata.org/entity/Q2764289","http://www.wikidata.org/entity/Q48384","http://www.wikidata.org/entity/Q827524","http://www.wikidata.org/entity/Q2070173","http://www.wikidata.org/entity/Q2095594","http://www.wikidata.org/entity/Q452229","http://www.wikidata.org/entity/Q497623","http://www.wikidata.org/entity/Q1389937","http://www.wikidata.org/entity/Q275854","http://www.wikidata.org/entity/Q474961","http://www.wikidata.org/entity/Q1773622","http://www.wikidata.org/entity/Q1922005","http://www.wikidata.org/entity/Q2718445","http://www.wikidata.org/entity/Q2742873","http://www.wikidata.org/entity/Q2748165","http://www.wikidata.org/entity/Q3416864","http://www.wikidata.org/entity/Q261425","http://www.wikidata.org/entity/Q269855","http://www.wikidata.org/entity/Q271929","http://www.wikidata.org/entity/Q510353","http://www.wikidata.org/entity/Q517635","http://www.wikidata.org/entity/Q2206556","http://www.wikidata.org/entity/Q2321233","http://www.wikidata.org/entity/Q2423919","http://www.wikidata.org/entity/Q3127743","http://www.wikidata.org/entity/Q927805","http://www.wikidata.org/entity/Q5181968","http://www.wikidata.org/entity/Q6215328","http://www.wikidata.org/entity/Q1878190","http://www.wikidata.org/entity/Q2369095","http://www.wikidata.org/entity/Q1921925","http://www.wikidata.org/entity/Q2145902","http://www.wikidata.org/entity/Q2542173","http://www.wikidata.org/entity/Q3197136","http://www.wikidata.org/entity/Q2047317","http://www.wikidata.org/entity/Q2175225","http://www.wikidata.org/entity/Q2180461","http://www.wikidata.org/entity/Q1999453","http://www.wikidata.org/entity/Q2575935","http://www.wikidata.org/entity/Q2423103","http://www.wikidata.org/entity/Q2523424","http://www.wikidata.org/entity/Q2958415","http://www.wikidata.org/entity/Q3165909","http://www.wikidata.org/entity/Q72771","http://www.wikidata.org/entity/Q185389","http://www.wikidata.org/entity/Q2364607","http://www.wikidata.org/entity/Q2557564","http://www.wikidata.org/entity/Q311393","http://www.wikidata.org/entity/Q360042","http://www.wikidata.org/entity/Q1409435","http://www.wikidata.org/entity/Q1873082","http://www.wikidata.org/entity/Q1882937","http://www.wikidata.org/entity/Q169098","http://www.wikidata.org/entity/Q287350","http://www.wikidata.org/entity/Q2032532","http://www.wikidata.org/entity/Q2037137","http://www.wikidata.org/entity/Q2056099","http://www.wikidata.org/entity/Q2256750","http://www.wikidata.org/entity/Q2280454","http://www.wikidata.org/entity/Q2803607","http://www.wikidata.org/entity/Q2928048","http://www.wikidata.org/entity/Q2940829","http://www.wikidata.org/entity/Q2446760","http://www.wikidata.org/entity/Q3280880","http://www.wikidata.org/entity/Q203774","http://www.wikidata.org/entity/Q354629","http://www.wikidata.org/entity/Q973394","http://www.wikidata.org/entity/Q2196060","http://www.wikidata.org/entity/Q2423063","http://www.wikidata.org/entity/Q897510","http://www.wikidata.org/entity/Q1940252","http://www.wikidata.org/entity/Q1970517","http://www.wikidata.org/entity/Q5550485","http://www.wikidata.org/entity/Q5714932","http://www.wikidata.org/entity/Q2374586","http://www.wikidata.org/entity/Q2279403","http://www.wikidata.org/entity/Q2632290","http://www.wikidata.org/entity/Q1961115","http://www.wikidata.org/entity/Q2603665","http://www.wikidata.org/entity/Q2050125","http://www.wikidata.org/entity/Q1982986","http://www.wikidata.org/entity/Q2740079","http://www.wikidata.org/entity/Q4919856","http://www.wikidata.org/entity/Q2520608","http://www.wikidata.org/entity/Q2183012","http://www.wikidata.org/entity/Q2533690","http://www.wikidata.org/entity/Q3140231","http://www.wikidata.org/entity/Q2692004","http://www.wikidata.org/entity/Q2350091","http://www.wikidata.org/entity/Q636089","http://www.wikidata.org/entity/Q2528361","http://www.wikidata.org/entity/Q2579624","http://www.wikidata.org/entity/Q5111206","http://www.wikidata.org/entity/Q216917","http://www.wikidata.org/entity/Q1840423","http://www.wikidata.org/entity/Q1875489","http://www.wikidata.org/entity/Q1887623","http://www.wikidata.org/entity/Q1993457","http://www.wikidata.org/entity/Q2690151","http://www.wikidata.org/entity/Q379502","http://www.wikidata.org/entity/Q739832","http://www.wikidata.org/entity/Q1354265","http://www.wikidata.org/entity/Q2272423","http://www.wikidata.org/entity/Q2800260","http://www.wikidata.org/entity/Q3543752","http://www.wikidata.org/entity/Q46424","http://www.wikidata.org/entity/Q2102663","http://www.wikidata.org/entity/Q2435577","http://www.wikidata.org/entity/Q2459223","http://www.wikidata.org/entity/Q459521","http://www.wikidata.org/entity/Q3386154","http://www.wikidata.org/entity/Q5254375","http://www.wikidata.org/entity/Q1930627","http://www.wikidata.org/entity/Q2650866","http://www.wikidata.org/entity/Q2748094","http://www.wikidata.org/entity/Q2749513","http://www.wikidata.org/entity/Q262062","http://www.wikidata.org/entity/Q489047","http://www.wikidata.org/entity/Q509911","http://www.wikidata.org/entity/Q518422","http://www.wikidata.org/entity/Q1682410","http://www.wikidata.org/entity/Q2177948","http://www.wikidata.org/entity/Q2303304","http://www.wikidata.org/entity/Q1961229","http://www.wikidata.org/entity/Q2222191","http://www.wikidata.org/entity/Q2357082","http://www.wikidata.org/entity/Q2781282","http://www.wikidata.org/entity/Q1950707","http://www.wikidata.org/entity/Q1924855","http://www.wikidata.org/entity/Q2152266","http://www.wikidata.org/entity/Q1909453","http://www.wikidata.org/entity/Q2794527","http://www.wikidata.org/entity/Q2795260","http://www.wikidata.org/entity/Q2170338","http://www.wikidata.org/entity/Q2336652","http://www.wikidata.org/entity/Q2939295","http://www.wikidata.org/entity/Q2375415","http://www.wikidata.org/entity/Q2489517","http://www.wikidata.org/entity/Q1883713","http://www.wikidata.org/entity/Q2006598","http://www.wikidata.org/entity/Q2666163","http://www.wikidata.org/entity/Q291677","http://www.wikidata.org/entity/Q2022531","http://www.wikidata.org/entity/Q2804712","http://www.wikidata.org/entity/Q5171216","http://www.wikidata.org/entity/Q2092290","http://www.wikidata.org/entity/Q2455262","http://www.wikidata.org/entity/Q3277437","http://www.wikidata.org/entity/Q4397542","http://www.wikidata.org/entity/Q4470145","http://www.wikidata.org/entity/Q173972","http://www.wikidata.org/entity/Q454610","http://www.wikidata.org/entity/Q5276314","http://www.wikidata.org/entity/Q1773609","http://www.wikidata.org/entity/Q2715719","http://www.wikidata.org/entity/Q3430723","http://www.wikidata.org/entity/Q261361","http://www.wikidata.org/entity/Q328316","http://www.wikidata.org/entity/Q518219","http://www.wikidata.org/entity/Q1721938","http://www.wikidata.org/entity/Q2231597","http://www.wikidata.org/entity/Q2311688","http://www.wikidata.org/entity/Q2326737","http://www.wikidata.org/entity/Q2389753","http://www.wikidata.org/entity/Q2407817","http://www.wikidata.org/entity/Q2413757","http://www.wikidata.org/entity/Q2430404","http://www.wikidata.org/entity/Q939333","http://www.wikidata.org/entity/Q5550337","http://www.wikidata.org/entity/Q1810754","http://www.wikidata.org/entity/Q1825559","http://www.wikidata.org/entity/Q2224653","http://www.wikidata.org/entity/Q2374734","http://www.wikidata.org/entity/Q2122775","http://www.wikidata.org/entity/Q2159426","http://www.wikidata.org/entity/Q2412238","http://www.wikidata.org/entity/Q2220229","http://www.wikidata.org/entity/Q2049226","http://www.wikidata.org/entity/Q2050257","http://www.wikidata.org/entity/Q1909454","http://www.wikidata.org/entity/Q2247947","http://www.wikidata.org/entity/Q2397808","http://www.wikidata.org/entity/Q2794905","http://www.wikidata.org/entity/Q2043261","http://www.wikidata.org/entity/Q2658539","http://www.wikidata.org/entity/Q2059126","http://www.wikidata.org/entity/Q2334067","http://www.wikidata.org/entity/Q2687946","http://www.wikidata.org/entity/Q3083673","http://www.wikidata.org/entity/Q2452102","http://www.wikidata.org/entity/Q3055177","http://www.wikidata.org/entity/Q2288630","http://www.wikidata.org/entity/Q2805811","http://www.wikidata.org/entity/Q2206661","http://www.wikidata.org/entity/Q2208090","http://www.wikidata.org/entity/Q4714651","http://www.wikidata.org/entity/Q208020","http://www.wikidata.org/entity/Q2362355","http://www.wikidata.org/entity/Q2365639","http://www.wikidata.org/entity/Q2494824","http://www.wikidata.org/entity/Q693849","http://www.wikidata.org/entity/Q1845837","http://www.wikidata.org/entity/Q2017578","http://www.wikidata.org/entity/Q2692582","http://www.wikidata.org/entity/Q6264537","http://www.wikidata.org/entity/Q1364306","http://www.wikidata.org/entity/Q2804494","http://www.wikidata.org/entity/Q1839794","http://www.wikidata.org/entity/Q2080513","http://www.wikidata.org/entity/Q2437754","http://www.wikidata.org/entity/Q229046","http://www.wikidata.org/entity/Q457582","http://www.wikidata.org/entity/Q560725","http://www.wikidata.org/entity/Q973674","http://www.wikidata.org/entity/Q984463","http://www.wikidata.org/entity/Q2990978","http://www.wikidata.org/entity/Q275225","http://www.wikidata.org/entity/Q1771268","http://www.wikidata.org/entity/Q272703","http://www.wikidata.org/entity/Q763833","http://www.wikidata.org/entity/Q1258713","http://www.wikidata.org/entity/Q2152763","http://www.wikidata.org/entity/Q2169615","http://www.wikidata.org/entity/Q2416851","http://www.wikidata.org/entity/Q2593931","http://www.wikidata.org/entity/Q2619538","http://www.wikidata.org/entity/Q2845449","http://www.wikidata.org/entity/Q4933904","http://www.wikidata.org/entity/Q5740436","http://www.wikidata.org/entity/Q6204463","http://www.wikidata.org/entity/Q14519491","http://www.wikidata.org/entity/Q2786973","http://www.wikidata.org/entity/Q1811560","http://www.wikidata.org/entity/Q2269653","http://www.wikidata.org/entity/Q13738620","http://www.wikidata.org/entity/Q2142388","http://www.wikidata.org/entity/Q2633202","http://www.wikidata.org/entity/Q2380263","http://www.wikidata.org/entity/Q2697734","http://www.wikidata.org/entity/Q2569232","http://www.wikidata.org/entity/Q1840004","http://www.wikidata.org/entity/Q2303123","http://www.wikidata.org/entity/Q2201719","http://www.wikidata.org/entity/Q2335192","http://www.wikidata.org/entity/Q2833237","http://www.wikidata.org/entity/Q3158011","http://www.wikidata.org/entity/Q2388743","http://www.wikidata.org/entity/Q2802104","http://www.wikidata.org/entity/Q5665636","http://www.wikidata.org/entity/Q2323566","http://www.wikidata.org/entity/Q2746971","http://www.wikidata.org/entity/Q3381527","http://www.wikidata.org/entity/Q3237188","http://www.wikidata.org/entity/Q4027633","http://www.wikidata.org/entity/Q184218","http://www.wikidata.org/entity/Q189686","http://www.wikidata.org/entity/Q2336382","http://www.wikidata.org/entity/Q2514237","http://www.wikidata.org/entity/Q2519215","http://www.wikidata.org/entity/Q2520287","http://www.wikidata.org/entity/Q1878648","http://www.wikidata.org/entity/Q2690318","http://www.wikidata.org/entity/Q1352440","http://www.wikidata.org/entity/Q2039703","http://www.wikidata.org/entity/Q2054132","http://www.wikidata.org/entity/Q4317458","http://www.wikidata.org/entity/Q169508","http://www.wikidata.org/entity/Q432740","http://www.wikidata.org/entity/Q454488","http://www.wikidata.org/entity/Q458594","http://www.wikidata.org/entity/Q462340","http://www.wikidata.org/entity/Q508504","http://www.wikidata.org/entity/Q469293","http://www.wikidata.org/entity/Q2644802","http://www.wikidata.org/entity/Q514839","http://www.wikidata.org/entity/Q2592745","http://www.wikidata.org/entity/Q1904408","http://www.wikidata.org/entity/Q2295975","http://www.wikidata.org/entity/Q16011718","http://www.wikidata.org/entity/Q1484594","http://www.wikidata.org/entity/Q2478624","http://www.wikidata.org/entity/Q2559291","http://www.wikidata.org/entity/Q540980","http://www.wikidata.org/entity/Q2003785","http://www.wikidata.org/entity/Q1984744","http://www.wikidata.org/entity/Q2136383","http://www.wikidata.org/entity/Q2429118","http://www.wikidata.org/entity/Q2515841","http://www.wikidata.org/entity/Q2186749","http://www.wikidata.org/entity/Q2187420","http://www.wikidata.org/entity/Q2600989","http://www.wikidata.org/entity/Q2933188","http://www.wikidata.org/entity/Q185650","http://www.wikidata.org/entity/Q628422","http://www.wikidata.org/entity/Q2355792","http://www.wikidata.org/entity/Q2490044","http://www.wikidata.org/entity/Q245841","http://www.wikidata.org/entity/Q1850509","http://www.wikidata.org/entity/Q1884934","http://www.wikidata.org/entity/Q2016551","http://www.wikidata.org/entity/Q2681663","http://www.wikidata.org/entity/Q291417","http://www.wikidata.org/entity/Q2244660","http://www.wikidata.org/entity/Q2766387","http://www.wikidata.org/entity/Q2772018","http://www.wikidata.org/entity/Q2938789","http://www.wikidata.org/entity/Q2071792","http://www.wikidata.org/entity/Q4379385","http://www.wikidata.org/entity/Q201367","http://www.wikidata.org/entity/Q433052","http://www.wikidata.org/entity/Q434116","http://www.wikidata.org/entity/Q463156","http://www.wikidata.org/entity/Q273481","http://www.wikidata.org/entity/Q1586971","http://www.wikidata.org/entity/Q2624411","http://www.wikidata.org/entity/Q2719245","http://www.wikidata.org/entity/Q2746960","http://www.wikidata.org/entity/Q271339","http://www.wikidata.org/entity/Q2318321","http://www.wikidata.org/entity/Q2333056","http://www.wikidata.org/entity/Q2423812","http://www.wikidata.org/entity/Q935776","http://www.wikidata.org/entity/Q1964152","http://www.wikidata.org/entity/Q2614802","http://www.wikidata.org/entity/Q18202629","http://www.wikidata.org/entity/Q18692840","http://www.wikidata.org/entity/Q1811818","http://www.wikidata.org/entity/Q2280658","http://www.wikidata.org/entity/Q2479003","http://www.wikidata.org/entity/Q1976636","http://www.wikidata.org/entity/Q1813598","http://www.wikidata.org/entity/Q4846455","http://www.wikidata.org/entity/Q1957984","http://www.wikidata.org/entity/Q2396597","http://www.wikidata.org/entity/Q3908201","http://www.wikidata.org/entity/Q2238309","http://www.wikidata.org/entity/Q2915642","http://www.wikidata.org/entity/Q2676649","http://www.wikidata.org/entity/Q2321758","http://www.wikidata.org/entity/Q2324627","http://www.wikidata.org/entity/Q2491243","http://www.wikidata.org/entity/Q2500026","http://www.wikidata.org/entity/Q241952","http://www.wikidata.org/entity/Q245838","http://www.wikidata.org/entity/Q1856764","http://www.wikidata.org/entity/Q1892249","http://www.wikidata.org/entity/Q2020842","http://www.wikidata.org/entity/Q289121","http://www.wikidata.org/entity/Q735549","http://www.wikidata.org/entity/Q2029164","http://www.wikidata.org/entity/Q2042883","http://www.wikidata.org/entity/Q3050048","http://www.wikidata.org/entity/Q4518257","http://www.wikidata.org/entity/Q1310689","http://www.wikidata.org/entity/Q1838289","http://www.wikidata.org/entity/Q7310284","http://www.wikidata.org/entity/Q229491","http://www.wikidata.org/entity/Q355148","http://www.wikidata.org/entity/Q919601","http://www.wikidata.org/entity/Q2738345","http://www.wikidata.org/entity/Q6763372","http://www.wikidata.org/entity/Q519678","http://www.wikidata.org/entity/Q770117","http://www.wikidata.org/entity/Q776137","http://www.wikidata.org/entity/Q1545416","http://www.wikidata.org/entity/Q1549502","http://www.wikidata.org/entity/Q2180392","http://www.wikidata.org/entity/Q2409956","http://www.wikidata.org/entity/Q1981039","http://www.wikidata.org/entity/Q7141699","http://www.wikidata.org/entity/Q7520410","http://www.wikidata.org/entity/Q2360859","http://www.wikidata.org/entity/Q2281045","http://www.wikidata.org/entity/Q1922487","http://www.wikidata.org/entity/Q2158997","http://www.wikidata.org/entity/Q2863193","http://www.wikidata.org/entity/Q2181949","http://www.wikidata.org/entity/Q1984898","http://www.wikidata.org/entity/Q1997256","http://www.wikidata.org/entity/Q2248073","http://www.wikidata.org/entity/Q2636808","http://www.wikidata.org/entity/Q3182810","http://www.wikidata.org/entity/Q2042615","http://www.wikidata.org/entity/Q2168166","http://www.wikidata.org/entity/Q3018553","http://www.wikidata.org/entity/Q2206273","http://www.wikidata.org/entity/Q2319734","http://www.wikidata.org/entity/Q2316915","http://www.wikidata.org/entity/Q2509763","http://www.wikidata.org/entity/Q726284","http://www.wikidata.org/entity/Q2511529","http://www.wikidata.org/entity/Q259606","http://www.wikidata.org/entity/Q440165","http://www.wikidata.org/entity/Q4770442","http://www.wikidata.org/entity/Q391230","http://www.wikidata.org/entity/Q1889031","http://www.wikidata.org/entity/Q730181","http://www.wikidata.org/entity/Q2271212","http://www.wikidata.org/entity/Q2776434","http://www.wikidata.org/entity/Q1153544","http://www.wikidata.org/entity/Q1295256","http://www.wikidata.org/entity/Q2462169","http://www.wikidata.org/entity/Q173714","http://www.wikidata.org/entity/Q345744","http://www.wikidata.org/entity/Q455411","http://www.wikidata.org/entity/Q497736","http://www.wikidata.org/entity/Q2660496","http://www.wikidata.org/entity/Q271343","http://www.wikidata.org/entity/Q521193","http://www.wikidata.org/entity/Q2225531","http://www.wikidata.org/entity/Q2427118","http://www.wikidata.org/entity/Q3127910","http://www.wikidata.org/entity/Q926482","http://www.wikidata.org/entity/Q1956081","http://www.wikidata.org/entity/Q2587768","http://www.wikidata.org/entity/Q2855717","http://www.wikidata.org/entity/Q2868539","http://www.wikidata.org/entity/Q18032936","http://www.wikidata.org/entity/Q3037170","http://www.wikidata.org/entity/Q2123573","http://www.wikidata.org/entity/Q3285218","http://www.wikidata.org/entity/Q2570394","http://www.wikidata.org/entity/Q2978369","http://www.wikidata.org/entity/Q1840753","http://www.wikidata.org/entity/Q1932918","http://www.wikidata.org/entity/Q2013778","http://www.wikidata.org/entity/Q3148033","http://www.wikidata.org/entity/Q2202429","http://www.wikidata.org/entity/Q1876263","http://www.wikidata.org/entity/Q2162553","http://www.wikidata.org/entity/Q2528044","http://www.wikidata.org/entity/Q2210054","http://www.wikidata.org/entity/Q4459526","http://www.wikidata.org/entity/Q2508912","http://www.wikidata.org/entity/Q402537","http://www.wikidata.org/entity/Q211607","http://www.wikidata.org/entity/Q2354343","http://www.wikidata.org/entity/Q2369770","http://www.wikidata.org/entity/Q2499716","http://www.wikidata.org/entity/Q5605198","http://www.wikidata.org/entity/Q256628","http://www.wikidata.org/entity/Q2572902","http://www.wikidata.org/entity/Q313140","http://www.wikidata.org/entity/Q2004488","http://www.wikidata.org/entity/Q2672583","http://www.wikidata.org/entity/Q2691952","http://www.wikidata.org/entity/Q552349","http://www.wikidata.org/entity/Q507151","http://www.wikidata.org/entity/Q1586429","http://www.wikidata.org/entity/Q1922191","http://www.wikidata.org/entity/Q2661186","http://www.wikidata.org/entity/Q264191","http://www.wikidata.org/entity/Q2118970","http://www.wikidata.org/entity/Q2154479","http://www.wikidata.org/entity/Q2418843","http://www.wikidata.org/entity/Q2423748","http://www.wikidata.org/entity/Q4937803","http://www.wikidata.org/entity/Q6215786","http://www.wikidata.org/entity/Q16013740","http://www.wikidata.org/entity/Q2122679","http://www.wikidata.org/entity/Q2701059","http://www.wikidata.org/entity/Q3113426","http://www.wikidata.org/entity/Q2175546","http://www.wikidata.org/entity/Q1974437","http://www.wikidata.org/entity/Q2103363","http://www.wikidata.org/entity/Q2613710","http://www.wikidata.org/entity/Q2775310","http://www.wikidata.org/entity/Q2872813","http://www.wikidata.org/entity/Q4037563","http://www.wikidata.org/entity/Q2182600","http://www.wikidata.org/entity/Q2186871","http://www.wikidata.org/entity/Q2467097","http://www.wikidata.org/entity/Q2424931","http://www.wikidata.org/entity/Q2677515","http://www.wikidata.org/entity/Q3027752","http://www.wikidata.org/entity/Q3998621","http://www.wikidata.org/entity/Q5631049"]
    nethierland_baseball_Kg_humans=["http://www.wikidata.org/entity/Q628053","http://www.wikidata.org/entity/Q2488184","http://www.wikidata.org/entity/Q2692182","http://www.wikidata.org/entity/Q2035162","http://www.wikidata.org/entity/Q2804281","http://www.wikidata.org/entity/Q2914750","http://www.wikidata.org/entity/Q3891045","http://www.wikidata.org/entity/Q16231839","http://www.wikidata.org/entity/Q2227840","http://www.wikidata.org/entity/Q2354914","http://www.wikidata.org/entity/Q2501799","http://www.wikidata.org/entity/Q2147000","http://www.wikidata.org/entity/Q2050983","http://www.wikidata.org/entity/Q2000889","http://www.wikidata.org/entity/Q3282574","http://www.wikidata.org/entity/Q2435013","http://www.wikidata.org/entity/Q2522966","http://www.wikidata.org/entity/Q2688540","http://www.wikidata.org/entity/Q2289598","http://www.wikidata.org/entity/Q2695788","http://www.wikidata.org/entity/Q2324754","http://www.wikidata.org/entity/Q5502236","http://www.wikidata.org/entity/Q2035095","http://www.wikidata.org/entity/Q5061926","http://www.wikidata.org/entity/Q1815636","http://www.wikidata.org/entity/Q5639453","http://www.wikidata.org/entity/Q774255","http://www.wikidata.org/entity/Q2219401","http://www.wikidata.org/entity/Q2391114","http://www.wikidata.org/entity/Q11317140","http://www.wikidata.org/entity/Q1849349","http://www.wikidata.org/entity/Q1838923","http://www.wikidata.org/entity/Q2295072","http://www.wikidata.org/entity/Q2065390","http://www.wikidata.org/entity/Q2230115","http://www.wikidata.org/entity/Q2150723","http://www.wikidata.org/entity/Q1909542","http://www.wikidata.org/entity/Q2185556","http://www.wikidata.org/entity/Q3457095","http://www.wikidata.org/entity/Q4608880","http://www.wikidata.org/entity/Q4504717","http://www.wikidata.org/entity/Q599372","http://www.wikidata.org/entity/Q2553574","http://www.wikidata.org/entity/Q3814639","http://www.wikidata.org/entity/Q2278023","http://www.wikidata.org/entity/Q2755752","http://www.wikidata.org/entity/Q2788525","http://www.wikidata.org/entity/Q1376903","http://www.wikidata.org/entity/Q2627003","http://www.wikidata.org/entity/Q2127569","http://www.wikidata.org/entity/Q2133371","http://www.wikidata.org/entity/Q2318574","http://www.wikidata.org/entity/Q5924135","http://www.wikidata.org/entity/Q13441010","http://www.wikidata.org/entity/Q3080297","http://www.wikidata.org/entity/Q2660333","http://www.wikidata.org/entity/Q2173093","http://www.wikidata.org/entity/Q2424168","http://www.wikidata.org/entity/Q2316883","http://www.wikidata.org/entity/Q2682031","http://www.wikidata.org/entity/Q3204914","http://www.wikidata.org/entity/Q2329801","http://www.wikidata.org/entity/Q7291852","http://www.wikidata.org/entity/Q16208159","http://www.wikidata.org/entity/Q2079724","http://www.wikidata.org/entity/Q2736571","http://www.wikidata.org/entity/Q2198339","http://www.wikidata.org/entity/Q3020159","http://www.wikidata.org/entity/Q2714220","http://www.wikidata.org/entity/Q2254869","http://www.wikidata.org/entity/Q5165826","http://www.wikidata.org/entity/Q2372458","http://www.wikidata.org/entity/Q1886645","http://www.wikidata.org/entity/Q2286558","http://www.wikidata.org/entity/Q2913674","http://www.wikidata.org/entity/Q2172021","http://www.wikidata.org/entity/Q4292375","http://www.wikidata.org/entity/Q2846523","http://www.wikidata.org/entity/Q7272450","http://www.wikidata.org/entity/Q16194186","http://www.wikidata.org/entity/Q16232185","http://www.wikidata.org/entity/Q2785266","http://www.wikidata.org/entity/Q2862665","http://www.wikidata.org/entity/Q2014115","http://www.wikidata.org/entity/Q2452102","http://www.wikidata.org/entity/Q2589267","http://www.wikidata.org/entity/Q2594495","http://www.wikidata.org/entity/Q2754901","http://www.wikidata.org/entity/Q2953195","http://www.wikidata.org/entity/Q2084490","http://www.wikidata.org/entity/Q1186030","http://www.wikidata.org/entity/Q1849880","http://www.wikidata.org/entity/Q2242497","http://www.wikidata.org/entity/Q3502696","http://www.wikidata.org/entity/Q233819","http://www.wikidata.org/entity/Q2110632","http://www.wikidata.org/entity/Q3483336","http://www.wikidata.org/entity/Q2790621","http://www.wikidata.org/entity/Q1850110","http://www.wikidata.org/entity/Q1851845","http://www.wikidata.org/entity/Q1943665","http://www.wikidata.org/entity/Q2341102","http://www.wikidata.org/entity/Q2697365","http://www.wikidata.org/entity/Q2554339","http://www.wikidata.org/entity/Q2397926","http://www.wikidata.org/entity/Q1833898","http://www.wikidata.org/entity/Q2693888","http://www.wikidata.org/entity/Q2346418","http://www.wikidata.org/entity/Q2571030","http://www.wikidata.org/entity/Q3026879","http://www.wikidata.org/entity/Q3522791","http://www.wikidata.org/entity/Q2450729","http://www.wikidata.org/entity/Q4869410","http://www.wikidata.org/entity/Q2652518","http://www.wikidata.org/entity/Q2738681","http://www.wikidata.org/entity/Q958652","http://www.wikidata.org/entity/Q2302961","http://www.wikidata.org/entity/Q16197034","http://www.wikidata.org/entity/Q2266290","http://www.wikidata.org/entity/Q3200165","http://www.wikidata.org/entity/Q1953350","http://www.wikidata.org/entity/Q2041040","http://www.wikidata.org/entity/Q2461277","http://www.wikidata.org/entity/Q2578563","http://www.wikidata.org/entity/Q3454116","http://www.wikidata.org/entity/Q3987004","http://www.wikidata.org/entity/Q2763115","http://www.wikidata.org/entity/Q4673162","http://www.wikidata.org/entity/Q4517987","http://www.wikidata.org/entity/Q2510039","http://www.wikidata.org/entity/Q3345858","http://www.wikidata.org/entity/Q1903426","http://www.wikidata.org/entity/Q3116243","http://www.wikidata.org/entity/Q1958266","http://www.wikidata.org/entity/Q2079591","http://www.wikidata.org/entity/Q2300863","http://www.wikidata.org/entity/Q2343006","http://www.wikidata.org/entity/Q3051571","http://www.wikidata.org/entity/Q2554781","http://www.wikidata.org/entity/Q1996834","http://www.wikidata.org/entity/Q2308460","http://www.wikidata.org/entity/Q2887115","http://www.wikidata.org/entity/Q4019695","http://www.wikidata.org/entity/Q2518725","http://www.wikidata.org/entity/Q1845262","http://www.wikidata.org/entity/Q3944801","http://www.wikidata.org/entity/Q3335063","http://www.wikidata.org/entity/Q3176824","http://www.wikidata.org/entity/Q2172735","http://www.wikidata.org/entity/Q2317851","http://www.wikidata.org/entity/Q11310638","http://www.wikidata.org/entity/Q16232148","http://www.wikidata.org/entity/Q2132738","http://www.wikidata.org/entity/Q2360899","http://www.wikidata.org/entity/Q331561","http://www.wikidata.org/entity/Q2054501","http://www.wikidata.org/entity/Q2068442","http://www.wikidata.org/entity/Q2181169","http://www.wikidata.org/entity/Q2097086","http://www.wikidata.org/entity/Q2658755","http://www.wikidata.org/entity/Q2649364","http://www.wikidata.org/entity/Q2914404","http://www.wikidata.org/entity/Q2074763","http://www.wikidata.org/entity/Q2209752","http://www.wikidata.org/entity/Q4334168","http://www.wikidata.org/entity/Q5810870","http://www.wikidata.org/entity/Q5258782","http://www.wikidata.org/entity/Q1902109","http://www.wikidata.org/entity/Q2743885","http://www.wikidata.org/entity/Q2424442","http://www.wikidata.org/entity/Q7280212","http://www.wikidata.org/entity/Q2782354","http://www.wikidata.org/entity/Q2785747","http://www.wikidata.org/entity/Q2280625","http://www.wikidata.org/entity/Q2215685","http://www.wikidata.org/entity/Q2570874","http://www.wikidata.org/entity/Q1839840","http://www.wikidata.org/entity/Q1868153","http://www.wikidata.org/entity/Q2247982","http://www.wikidata.org/entity/Q2308272","http://www.wikidata.org/entity/Q3283193","http://www.wikidata.org/entity/Q1994943","http://www.wikidata.org/entity/Q2205483","http://www.wikidata.org/entity/Q2333811","http://www.wikidata.org/entity/Q2531246","http://www.wikidata.org/entity/Q4542024","http://www.wikidata.org/entity/Q5001511","http://www.wikidata.org/entity/Q2529887","http://www.wikidata.org/entity/Q2694461","http://www.wikidata.org/entity/Q2695678","http://www.wikidata.org/entity/Q2443894","http://www.wikidata.org/entity/Q3931212","http://www.wikidata.org/entity/Q2702412","http://www.wikidata.org/entity/Q976298","http://www.wikidata.org/entity/Q1916105","http://www.wikidata.org/entity/Q959800","http://www.wikidata.org/entity/Q2132997","http://www.wikidata.org/entity/Q2789414","http://www.wikidata.org/entity/Q4915065","http://www.wikidata.org/entity/Q2199594","http://www.wikidata.org/entity/Q2619389","http://www.wikidata.org/entity/Q2458436","http://www.wikidata.org/entity/Q2730778","http://www.wikidata.org/entity/Q4996054","http://www.wikidata.org/entity/Q2288205","http://www.wikidata.org/entity/Q2352159"]
    dblp_AAAI_germany = ["https://dblp.org/rec/conf/aaai/0002K21", "https://dblp.org/rec/conf/aaai/TomaniB21","https://dblp.org/rec/conf/aaai/KupcsikSKTWSB21", "https://dblp.org/rec/conf/aaai/PhuocEL21","https://dblp.org/rec/conf/aaai/LuoQXCZDYZWCHRL21", "https://dblp.org/rec/conf/aaai/MohrBH21","https://dblp.org/rec/conf/aaai/PanthaplackelAB21","https://dblp.org/rec/conf/aaai/BentertBG0N21", "https://dblp.org/rec/conf/aaai/Baier0M21","https://dblp.org/rec/conf/aaai/Bilo0LLM21", "https://dblp.org/rec/conf/aaai/0002M021","https://dblp.org/rec/conf/aaai/Behnke021", "https://dblp.org/rec/conf/aaai/WilhelmK21","https://dblp.org/rec/conf/aaai/BodirskyK21", "https://dblp.org/rec/conf/aaai/GilT21","https://dblp.org/rec/conf/aaai/FrikhaKKT21", "https://dblp.org/rec/conf/aaai/Dvorak0W21","https://dblp.org/rec/conf/aaai/KotnisLN21", "https://dblp.org/rec/conf/aaai/DalyMAN21","https://dblp.org/rec/conf/aaai/FichteHM21", "https://dblp.org/rec/conf/aaai/Rothe21","https://dblp.org/rec/conf/aaai/0001W21a", "https://dblp.org/rec/conf/aaai/UnalAP21","https://dblp.org/rec/conf/aaai/0001BW21", "https://dblp.org/rec/conf/aaai/BrosowskyKD021","https://dblp.org/rec/conf/aaai/FickertGF0MR21","https://dblp.org/rec/conf/aaai/BesserveSJS21", "https://dblp.org/rec/conf/aaai/HedderichZK21","https://dblp.org/rec/conf/aaai/LuoZCQDZWCHRL21","https://dblp.org/rec/conf/aaai/LawrenceSN21", "https://dblp.org/rec/conf/aaai/LedentMLK21","https://dblp.org/rec/conf/aaai/HollerB21", "https://dblp.org/rec/conf/aaai/MianMV21","https://dblp.org/rec/conf/aaai/SharifzadehBT21", "https://dblp.org/rec/conf/aaai/Potyka21","https://dblp.org/rec/conf/aaai/MathewSYBG021", "https://dblp.org/rec/conf/aaai/PhanBGSRL21","https://dblp.org/rec/conf/aaai/ShaoSSSK21", "https://dblp.org/rec/conf/aaai/WienobstBL21","https://dblp.org/rec/conf/aaai/0001FBBQZDSM021", "https://dblp.org/rec/conf/aaai/0001CDM21","https://dblp.org/rec/conf/aaai/ArtaleJMOW21", "https://dblp.org/rec/conf/aaai/00010Q0LK21","https://dblp.org/rec/conf/aaai/WuLLK21", "https://dblp.org/rec/conf/aaai/HeegerHMMNS21","https://dblp.org/rec/conf/aaai/DennisB0021", "https://dblp.org/rec/conf/aaai/KruseDKS21","https://dblp.org/rec/conf/aaai/LienenH21", "https://dblp.org/rec/conf/aaai/TorralbaSKS021","https://dblp.org/rec/conf/aaai/NeiderGGT0021", "https://dblp.org/rec/conf/aaai/TanNDNB21","https://dblp.org/rec/conf/aaai/NayyeriVA021", "https://dblp.org/rec/conf/aaai/Torralba21","https://dblp.org/rec/conf/aaai/ZhouLHLZK21"]
    dblp_AAAI_ACM_VLDB_germany_china_2021=pd.read_csv("SPARQLML_HardAnswerResults/dblp_AAAI_ACM_VLDB_germany_china_2021.csv",header=None,sep=",")[0].tolist()
    Real_target_node=dblp_AAAI_ACM_VLDB_germany_china_2021
    # target_node="human"
    target_node = "Publication"
    for elem in range(0,3):
        # inference_query_wikidata_Citizenship_Profession_univ_NC_v2
        resDF, MetaQueries,query_time_sec = kgnet.executeSPARQLMLInferenceQuery(inference_MQuery_dblp_NC_venue_aff,in_pipline=True)
        exec_time.append(query_time_sec)
        # resDF,MetaQueries=kgnet.executeSPARQLMLInferenceQuery(inference_MQuery_dblp2022_NC_LP)
        #resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_award_workField_NC)
        # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(nested_Query)
        print("query_time_sec=",query_time_sec)
        print(resDF)
        true_prad_lst.append(len(set(resDF[target_node].tolist()).intersection(set(Real_target_node)))/len(Real_target_node))
        print(f"True Predictions Ratio:{true_prad_lst[-1]}")
        not_pred_lst.append(len(set(Real_target_node)-set(resDF[target_node].tolist())) / len(Real_target_node))
        print(f"Not Predicted Ratio:{not_pred_lst[-1]}")
        false_pred_lst.append(len(set(resDF[target_node].tolist())-set(Real_target_node)) / len(Real_target_node))
        print(f"False Predictions Ratio:{false_pred_lst[-1]}")
        print(MetaQueries)
        # print("candidateSparqlQuery=",MetaQueries['candidateSparqlQuery'])
    print(f"avg_time={mean(exec_time)} Sec.")
    print(f"avg_true_prads={mean(true_prad_lst)*100}% .")
    print(f"avg_not_preds={mean(not_pred_lst)*100}% .")
    print(f"avg_false_preds={mean(false_pred_lst)*100}% .")
    #############################################3
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',KG_NamedGraph_IRI='http://www.RVO.net/', KG_Prefix="RVO")
    # kgnet.uploadKG(ttl_file_url="http://206.12.89.16/CodsData/KGNET/KGs/OBA.nt",name="Landenportaal-Rijksdienst voor Ondernemend Nederland (RVO)",description="information about countries around the globe with the objectives of these country monitors",domain="geometric")

