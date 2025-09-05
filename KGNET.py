import Constants
import pandas as pd
from Constants import *
from SparqlMLaasService.GMLOperators import gmlInferenceOperator, gmlInsertOperator
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
from SparqlMLaasService.gmlRewriter import gmlQueryParser
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from RDFEngineManager.UDF_Manager_Virtuoso import VirtuosoUDFManager
# from pyvis.network import Network
from statistics import mean

color_palette = ["#ff6347", "#d8bfd8", "#66d8ff", "#ff7f50", "#ffa07a",
                 "#ffebcd", "#22d8d8", "#ffe4e1", "#c71585", "#ff8c00", "#ffb6c1", "#f08080", "#dc646c",
                 "#4686b8", "#d2796e", "#9aed52", "#cd6c6c", "#32ed52", "#ff5499", "#f1e69c", "#9dff3f", "#01ff8f",
                 "#7b88de", "#4682b4", "#d2691e", "#9acd32", "#20b2aa", "#cd5c5c", "#00008b", "#32cd32", "#8fbc8f",
                 "#800080", "#9370db", "#9932cc", "#ff4500", "#ffa500", "#ffd700", "#0000cd", "#deb887", "#33ff00",
                 "#00ff7f", "#dc143c", "#20eeff", "#00bfff", "#0000ff", "#a020f0", "#adff2f", "#ff6347", "#ff00ff",
                 "#1e90ff", "#f0e68c", "#fd7811", "#dda0dd", "#90ee90", "#87ceeb", "#ff1493", "#7b68ee", "#ffa07a",
                 "#ee82ee", "#7fffd4", "#ff69b4", "#ffc0cb", "#dc143c", "#20eeff", "#00bfff", "#0000ff", "#a020f0",
                 "#adff2f", "#ff6347", "#ff00ff", "#1e90ff", "#f0e68c", "#f87811", "#dda0dd", "#90ee90", "#87ceeb",
                 "#ff1493", "#7b68ee", "#ffa07a", "#ee82ee", "#7fffd4", "#ff69b4", "#ffc0cb"]


class KGNET():
    GML_Operator_Types = Constants.GML_Operator_Types
    GNN_Methods = Constants.GNN_Methods
    KGNET_Config = Constants.KGNET_Config
    KGs_prefixs_dic = Constants.KGs_prefixs_dic
    namedGraphURI_dic = Constants.namedGraphURI_dic
    utils = Constants.utils
    "KGNET system main class that automates GML the training and infernce pipelines"

    def __init__(self, KG_endpointUrl, KGMeta_endpointUrl="http://206.12.98.118:8890/sparql",
                 KGMeta_KG_URI=Constants.KGNET_Config.KGMeta_IRI, RDFEngine=Constants.RDFEngine.OpenlinkVirtuoso,
                 KG_NamedGraph_IRI=None, KG_Prefix=None, KG_Prefix_URL=None):
        self.KGMeta_Governer = KGMeta_Governer(endpointUrl=KGMeta_endpointUrl, KGMeta_URI=KGMeta_KG_URI,
                                               RDFEngine=RDFEngine)
        self.VirtuosoUDFManager = VirtuosoUDFManager(host=KG_endpointUrl.split(":")[0].split("//")[-1])
        self.KG_sparqlEndpoint = sparqlEndpoint(endpointUrl=KG_endpointUrl, RDFEngine=RDFEngine)
        self.gml_insert_op = gmlInsertOperator(self.KGMeta_Governer, self.KG_sparqlEndpoint)
        self.KG_NamedGraph_URI = KG_NamedGraph_IRI
        self.RDFEngine = RDFEngine
        if KG_Prefix:
            self.kg_Prefix = KG_Prefix
        else:
            self.kg_Prefix = KG_NamedGraph_IRI.split("//")[1].split(".")[0]

        if self.kg_Prefix not in Constants.KGs_prefixs_dic.keys():
            if KG_Prefix_URL:
                Constants.KGs_prefixs_dic[self.kg_Prefix] = KG_Prefix_URL
            else:
                Constants.KGs_prefixs_dic[self.kg_Prefix] = KG_NamedGraph_IRI
        if self.kg_Prefix not in Constants.namedGraphURI_dic.keys():
            Constants.namedGraphURI_dic[self.kg_Prefix] = KG_NamedGraph_IRI

    def uploadKG(self, ttl_file_url, name, description, domain):
        # self.VirtuosoUDFManager.uploadKG_ttl(ttl_file_url,self.KG_NamedGraph_URI)
        gid, _ = self.KGMeta_Governer.insertKGMetadata(self.KG_sparqlEndpoint.endpointUrl, self.kg_Prefix,
                                                       self.KG_NamedGraph_URI, name, description, domain)
        return gid

    def getKGNodeEdgeTypes(self, write_to_file=False, prefix=None):
        "returns a dataframe of KG triples node/edge types considers only single source and destinations node types per edge type"
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
                s_type_query = "select (IF(STRLEN(xsd:string(?s_type_p))>0,xsd:string(?s_type_p),\"" + \
                               edgeType.split("/")[-1] + "_Subj\") as ?s_type) (count(*) as ?count) \n"
                s_type_query += "" if NamedGraph_URI is None else "from <" + NamedGraph_URI + "> \n"
                s_type_query += "where { ?s <" + edgeType + "> ?o. \n"
                s_type_query += " OPTIONAL {?s a ?s_type_p.} } \n group by  ?s_type_p \n order by desc(count(*))  limit 1000"

                o_type_query = "select (IF(STRLEN(xsd:string(?o_type_p))>0,xsd:string(?o_type_p),\"" + \
                               edgeType.split("/")[-1] + "_Obj\") as ?o_type) (count(*) as ?count) \n"
                o_type_query += "" if NamedGraph_URI is None else "from <" + NamedGraph_URI + "> \n"
                o_type_query += "where { ?s <" + edgeType + "> ?o. \n"
                o_type_query += " OPTIONAL {?o a ?o_type_p.} } \n group by  ?o_type_p \n order by desc(count(*))  limit 1000"

                s_types_df = self.KG_sparqlEndpoint.executeSparqlquery(s_type_query)
                o_types_df = self.KG_sparqlEndpoint.executeSparqlquery(o_type_query)
                KG_types_lst.append([s_types_df["s_type"].values[0].replace("\"", "").split("/")[-1].split("#")[-1],
                                     edgeType.split("/")[-1].split("#")[-1],
                                     o_types_df["o_type"].values[0].replace("\"", "").split("/")[-1].split("#")[-1]])
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

    def getKGNodeEdgeTypes_V2(self, write_to_file=False, prefix=None):
        "returns a dataframe of KG triples node/edge types considering multi node type per edge"
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
                query = """select distinct ?stype ?otype
                            from <""" + NamedGraph_URI + """>
                            where 
                            {
                            ?s <""" + edgeType + """> ?o.
                            ?s a ?stype.
                            ?o a ?otype.                            

                            #?stype a <http://schema.org/topLevelClass>. # For YAGO
                            #?otype a <http://schema.org/topLevelClass>. # For YAGO
                            }"""
                types_df = self.KG_sparqlEndpoint.executeSparqlquery(query)
                for idx, row in types_df.iterrows():
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

    def visualizeKG(self, types_df, width="100%", height="500px", Notebook=False, Directed=True):
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

    def train_GML(self, sparql_ml_insert_query):
        "Automates the GML training pipeline steps including: parsing the GML insert query ,identifying GML task type and attributes, sample task orianted subgraph, transform sampled subgraph into PYG dataset, train a GNN model, and save trained model meta-data into KGMeta KG "
        insert_task_dict = gmlQueryParser(sparql_ml_insert_query).extractQueryStatmentsDict()
        model_info, transform_info, train_info = self.gml_insert_op.executeQuery(insert_task_dict)
        return model_info, transform_info, train_info

    def getTargetEdgeTypeIRI(self, kg_prefix, target_edge_short):
        "return full edge URI for a given target edge type"
        edges_query = """ select   distinct ?p  
                       from <""" + self.KG_NamedGraph_URI + """>
                       where { ?s ?p ?o.} limit 1000 """
        edges_df = self.KG_sparqlEndpoint.executeSparqlquery(edges_query)
        edges_df["p"] = edges_df["p"].apply(lambda x: str(x).replace("\"", "").replace("<", "").replace(">", ""))
        edges_df["p_lower"] = edges_df["p"].apply(lambda x: str(x).lower())
        target_edge_df = edges_df[edges_df["p_lower"].str.endswith(target_edge_short.lower())]
        return target_edge_df["p"].values[0]

    def train_GML(self, operatorType, GNNMethod, targetNodeType=None, labelNodeType=None, targetEdge=None,
                  TOSG_Pattern=None, epochs=None, emb_size=None, MinInstancesPerLabel=21):
        "Automates the GML training pipeline given the minimal task attributes steps including: write a SPARQL-ML insert query,  parsing the GML insert query ,identifying GML task type and attributes, sample task orianted subgraph, transform sampled subgraph into PYG dataset, train a GNN model, and save trained model meta-data into KGMeta KG "
        if self.kg_Prefix is not None:
            if operatorType == Constants.GML_Operator_Types.NodeClassification and targetEdge is None:
                # self.kg_Prefix=targetNodeType.split(":")[0]
                kg_types_path = Constants.KGNET_Config.datasets_output_path + self.kg_Prefix + "_Types.csv"
                kg_types_ds = pd.read_csv(kg_types_path, header=None)
                tnType = targetNodeType.split(":")[-1].lower()
                lnType = labelNodeType.split(":")[-1].lower()
                target_edge_df = kg_types_ds[
                    (kg_types_ds[0].str.lower() == tnType) & (kg_types_ds[2].str.lower() == lnType)]
                targetEdge = target_edge_df[1].values[0]
                targetEdge = self.getTargetEdgeTypeIRI(self.kg_Prefix, targetEdge)
            elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
                ""
                # self.kg_Prefix = targetEdge.split(":")[0]
        else:
            raise Exception("KG types dataset is not exist")

        print("targetEdge=", targetEdge)
        kg_types_path = Constants.KGNET_Config.datasets_output_path + self.kg_Prefix + "_Types.csv"
        ######################### write sparqlML query #########################
        if self.kg_Prefix in Constants.KGs_prefixs_dic.keys():
            sparql_ml_insert_query = " prefix " + self.kg_Prefix + ":<" + Constants.KGs_prefixs_dic[
                self.kg_Prefix] + "> \n"
        else:
            sparql_ml_insert_query = " prefix " + self.kg_Prefix + ":<" + Constants.namedGraphURI_dic[
                self.kg_Prefix] + "> \n"
        sparql_ml_insert_query += """ prefix kgnet:<https://www.kgnet.com/>
           Insert into <kgnet>
           where{
               select * from kgnet.TrainGML(
               {\n"""
        if operatorType == Constants.GML_Operator_Types.NodeClassification:
            sparql_ml_insert_query += "\"name\":\"" + operatorType + ">" + self.kg_Prefix + ">" + targetNodeType + ">" + (
                "None" if labelNodeType is None else labelNodeType) + ">" + GNNMethod + "\",\n"
        elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
            sparql_ml_insert_query += "\"name\":\"" + operatorType + ">" + self.kg_Prefix + ">" + targetEdge.split("/")[
                -1] + ">" + GNNMethod + "\",\n"
        sparql_ml_insert_query += "\"GMLTask\":{\"taskType\":\"kgnet:" + operatorType + "\",\n"
        if targetNodeType is not None:
            sparql_ml_insert_query += "\"targetNode\":\"" + targetNodeType + "\",\n"
        if labelNodeType is not None:
            sparql_ml_insert_query += "\"labelNode\":\"" + labelNodeType + "\",\n"
        if MinInstancesPerLabel is not None:
            sparql_ml_insert_query += "\"MinInstancesPerLabel\":\"" + str(MinInstancesPerLabel) + "\",\n"

        sparql_ml_insert_query += "\"namedGraphURI\":\"" + self.KG_NamedGraph_URI + "\",\n"
        sparql_ml_insert_query += "\"namedGraphPrefix\":\"" + self.kg_Prefix + "\",\n"
        sparql_ml_insert_query += "\"targetEdge\":\"" + targetEdge + "\",\"GNNMethod\":\"" + GNNMethod + "\",\n"
        sparql_ml_insert_query += "\"datasetTypesFilePath\":\"" + kg_types_path + "\",\n"
        if epochs is not None:
            sparql_ml_insert_query += "\"epochs\":" + str(epochs) + ","
        if emb_size is not None:
            sparql_ml_insert_query += "\"embSize\":" + str(emb_size) + ","
        if operatorType == Constants.GML_Operator_Types.NodeClassification:
            sparql_ml_insert_query += "\"TOSG\":\"" + (
                TOSG_Patterns.d1h1 if TOSG_Pattern is None else TOSG_Pattern) + "\""
        elif operatorType == Constants.GML_Operator_Types.LinkPrediction:
            sparql_ml_insert_query += "\"TOSG\":\"" + (
                TOSG_Patterns.d2h1 if TOSG_Pattern is None else TOSG_Pattern) + "\""
        if operatorType == Constants.GML_Operator_Types.NodeClassification and self.kg_Prefix == "dblp" and targetNodeType == "dblp:Publication":
            sparql_ml_insert_query += """,\n "targetNodeFilters":{"filter1":["<https://dblp.org/rdf/schema#yearOfPublication>", "?year","filter(xsd:integer(?year)>=2020)"] \n}\n"""
        elif operatorType == Constants.GML_Operator_Types.NodeClassification and self.kg_Prefix == "yago" and targetNodeType == "yago:Person":
            sparql_ml_insert_query += """,\n "targetNodeFilters":{"filter1":["yago:birthDate", "?bdate","filter(xsd:date(?bdate)>=xsd:date('1970-01-01'))"] \n}\n"""
        elif operatorType == Constants.GML_Operator_Types.NodeClassification and self.kg_Prefix == "yago" and targetNodeType == "yago:CreativeWork":
            sparql_ml_insert_query += """,\n "targetNodeFilters":{"filter1":["yago:datePublished", "?pdate","filter(xsd:date(?pdate)>=xsd:date('1970-01-01'))"] \n}\n"""
        elif operatorType == Constants.GML_Operator_Types.NodeClassification and self.kg_Prefix == "crunchbase": 
            sparql_ml_insert_query += """,\n "targetNodeFilters":{"filter1":["crunchbase:born_on", "?born_date","filter(xsd:date(?born_date)>=xsd:date('1900-01-01'))"] \n}\n"""
        elif operatorType == Constants.GML_Operator_Types.NodeClassification and self.kg_Prefix == "lkmdb" and targetNodeType == "lkmdb:film":
            sparql_ml_insert_query += """,\n "targetNodeFilters":{"filter1":["<http://purl.org/dc/terms/date>", "?pdate","filter(strlen(?pdate)>=4).  filter (xsd:int(substr(?pdate,0,4)) <= 2005 )"] \n}\n"""
        sparql_ml_insert_query += "}\n})}"
        print("sparql_ml_insert_query=", sparql_ml_insert_query)
        ######################### write sparqlML query #########################
        insert_task_dict = gmlQueryParser(sparql_ml_insert_query).extractQueryStatmentsDict()
        model_info, transform_info, train_info = self.gml_insert_op.executeQuery(insert_task_dict)
        return model_info["task_uri"].split("/")[-1], model_info["model_uri"].split("/")[-1], {"model_info": model_info,
                                                                                               "transform_info": transform_info,
                                                                                               "train_info": train_info}

    def executeSPARQLMLInferenceQuery(self, query, ExecPlanIdx=None, kgwise_full_batch=True):
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
        if ExecPlanIdx is not None:
            res_df, exectuted_Queries, time_sec = gmlInferenceOp.executeQuery(query, ExecPlanIdx, kgwise_full_batch)
            return res_df, exectuted_Queries, time_sec
        else:
            df_res, candidateSparqlQuery, kgDataQuery, kgTargetNodesQuery, kgmetaModelQuery, model_ids_lst, time_sec = gmlInferenceOp.executeQuery(
                query, pipline=None, kgwise_full_batch=kgwise_full_batch)
            # df_res=df_res.apply(lambda x: (x.str)[1:-1])
            return df_res, {"model_ids_lst": model_ids_lst, "candidateSparqlQuery": candidateSparqlQuery,
                            "kgDataQuery": kgDataQuery, "InferenceTargetNodeQueries": kgTargetNodesQuery,
                            "kgmetaModelQuery": kgmetaModelQuery}, time_sec

    def getSPARQLMLExecQueryPlans(self, query):
        """Return all possible GML task execution plans
            * parse GML infernce query
            * Build DAG
            * Generate Possible Execution Plans using DAG
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
        return gmlInferenceOp.getExecQueryPlans(query)

    def getSPARQLMLQueryPlansCost(self, DAGExecPlans, DAG, decomposedSubqueries, CostModelParams):
        """Return cost for all possible execution plans
            * Calculate Cost Model for each Possible Execution Plans
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
        return gmlInferenceOp.getSPARQLMLQueryPlansCost(DAGExecPlans, DAG, decomposedSubqueries, CostModelParams)


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