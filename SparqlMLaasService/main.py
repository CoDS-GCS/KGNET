import Constants
from Constants import *
from GMLOperators import gmlInsertOperator
from KGMeta_Governer import KGMeta_Governer
from gmlRewriter import gmlQueryParser,gmlQueryRewriter
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
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

    DBLP_PV_Insert_Query = """
           prefix dblp:<https://www.dblp.org/>
           prefix kgnet:<https://www.kgnet.com/>
           Insert into <kgnet>
           where{
               select * from kgnet.TrainGML(
               {
               "name":"DBLP_Year_lte_1950_Paper_Venue_Classifer",
               "GMLTask":{
                   "taskType":"kgnet:NodeClassifier",
                   "targetNode":"dblp:Publication",
                   "labelNode":"dblp:venue",
                   "namedGraphURI":"http://dblp.org",
                   "targetEdge":"https://dblp.org/rdf/schema#publishedInJournal",
                   "namedGraphUri":"http://dblp.org",
                   "GNNMethod":\""""+GNN_Methods.RGCN+"""\",
                   "datasetTypesFilePath":\""""+KGNET_Config.datasets_output_path+"""dblp_Types.csv",
                   "TOSG":\""""+TOSG_Patterns.d1h1+"""\",
                   "targetNodeFilters":{
                "filter1":["<https://dblp.org/rdf/schema#yearOfPublication>", "?year","filter(xsd:integer(?year)<=1950)"]
                    }
                   },
               "taskBudget":{
                   "MaxMemory":"50GB",
                   "MaxTime":"1h",
                   "Priority":"ModelScore"}
               })}"""

    LinkedIMDB_Insert_Query = """
               prefix imdb:<https://www.imdb.com/>
               prefix kgnet:<https://www.kgnet.com/>
               Insert into <kgnet>
               where{
                   select * from kgnet.TrainGML(
                   {
                   "name":"LIMDB_Year_lte_d2h1_2020_Film_Language_Classifer",
                   "GMLTask":{
                       "taskType":"kgnet:NodeClassifier",
                       "targetNode":"imdb:Film",
                       "labelNode":"imdb:Language",
                       "targetEdge":"http://data.linkedmdb.org/resource/movie/language",
                       "namedGraphURI":"https://linkedimdb",
                       "GNNMethod":\""""+GNN_Methods.Graph_SAINT+"""\",
                       "datasetTypesFilePath":\""""+KGNET_Config.datasets_output_path+"""lmdb_Types.csv",
                       "TOSG":\""""+TOSG_Patterns.d2h1+"""\",
                       "targetNodeFilters":{
                "filter1":["<http://purl.org/dc/terms/date>", "?year","filter(xsd:integer(?year)<=2020)"]
                    }
                       },
                   "taskBudget":{
                       "maxMemory":"50GB",
                       "maxTime":"1h",
                       "priority":"ModelScore"}
                   })}"""
    LinkedIMDB_F_Subject_Insert_Query = """
                   prefix imdb:<https://www.imdb.com/>
                   prefix kgnet:<https://www.kgnet.com/>
                   Insert into <kgnet>
                   where{
                       select * from kgnet.TrainGML(
                       {
                       "name":"LIMDB_Year_lte_d2h1_2020_Film_Subject_Classifer",
                       "GMLTask":{
                           "taskType":"kgnet:NodeClassifier",
                           "targetNode":"imdb:Film",
                           "labelNode":"imdb:film_subject",
                           "targetEdge":"http://data.linkedmdb.org/resource/movie/film_subject",
                           "namedGraphURI":"https://linkedimdb",
                           "GNNMethod":\"""" + GNN_Methods.Graph_SAINT + """\",
                           "datasetTypesFilePath":\"""" + KGNET_Config.datasets_output_path + """lmdb_Types.csv",
                           "TOSG":\"""" + TOSG_Patterns.d2h1 + """\",
                           "targetNodeFilters":{
                    "filter1":["<http://purl.org/dc/terms/date>", "?year","filter(xsd:integer(?year)<=2020)"]
                        }
                           },
                       "taskBudget":{
                           "maxMemory":"50GB",
                           "maxTime":"1h",
                           "priority":"ModelScore"}
                       })}"""
    MAG_Insert_Query = """
                   prefix mag:<http://mag.org/>
                   prefix kgnet:<https://www.kgnet.com/>
                   Insert into <kgnet>
                   where{
                       select * from kgnet.TrainGML(
                       {
                       "name":"MAG_Year_lte_2011_Paper_Venue_Classifer",
                       "GMLTask":{
                           "taskType":"kgnet:NodeClassifier",
                           "targetNode":"mag:Paper",
                           "labelNode":"mag:Venue",
                           "LabelPredicate":"https://makg.org/property/appearsInJournal",
                           "named_graph_uri":"http://mag.org",
                           "GNN_Method":\""""+GNN_Methods.Graph_SAINT+"""\",
                           "dataset_types_path":\""""+KGNET_Config.datasets_output_path+"""mag_Types.csv",
                           "TOSG":\""""+TOSG_Patterns.d1h1+"""\",
                           "TargetNodeFilters":{
                    "filter1":["<https://makg.org/publish_year>", "?year","filter(xsd:integer(?year)=2012)"]
                        }
                           },
                       "TaskBudget":{
                           "MaxMemory":"50GB",
                           "MaxTime":"1h",
                           "Priority":"ModelScore"}
                       })}"""

    LinkedIMDB_F_Writer_LP_Insert_Query = """
                       prefix imdb:<https://www.imdb.com/>
                       prefix kgnet:<https://www.kgnet.com/>
                       Insert into <kgnet>
                       where{
                           select * from kgnet.TrainGML(
                           {
                           "name":"LIMDB_Year_lte_d2h1_2010_Film_Subject_Classifer",
                           "GMLTask":{
                               "taskType":"kgnet:LinkPredictor",
                               "targetEdge":"http://data.linkedmdb.org/resource/movie/writer",
                               "namedGraphURI":"https://linkedimdb",
                               "GNNMethod":\"""" + GNN_Methods.RGCN + """\",
                               "datasetTypesFilePath":\"""" + KGNET_Config.datasets_output_path + """lmdb_Types.csv",
                               "TOSG":\"""" + TOSG_Patterns.d2h1 + """\",
                               "targetNodeFilters":{
                        "filter1":["<http://purl.org/dc/terms/date>", "?year","filter(xsd:integer(?year)<=2010)"]
                            }
                               },
                           "taskBudget":{
                               "maxMemory":"50GB",
                               "maxTime":"1h",
                               "priority":"ModelScore"}
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

    kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.98.118:8890/sparql', KGMeta_URI=KGNET_Config.KGMeta_IRI)
    # KG_sparqlEndpoint = sparqlEndpoint(endpointUrl='http://206.12.98.118:8890/sparql')
    KG_sparqlEndpoint = sparqlEndpoint(endpointUrl='http://206.12.99.253:8890/sparql')
    # kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.97.2:8890/sparql/', KGMeta_URI="http://kgnet")
    # gmlqp = gmlQueryParser(dblp_NC)
    # (dataq,kmetaq)=gmlQueryRewriter(gmlqp.extractQueryStatmentsDict(),kgmeta_govener).rewrite_gml_query()
    # insert_task_dict = gmlQueryParser(DBLP_PV_Insert_Query).extractQueryStatmentsDict()
    # insert_task_dict = gmlQueryParser(LinkedIMDB_F_Subject_Insert_Query).extractQueryStatmentsDict()
    # insert_task_dict = gmlQueryParser(MAG_Insert_Query).extractQueryStatmentsDict()
    insert_task_dict = gmlQueryParser(LinkedIMDB_F_Writer_LP_Insert_Query).extractQueryStatmentsDict()
    gml_insert_op=gmlInsertOperator(kgmeta_govener,KG_sparqlEndpoint)
    res,transform_results_dict,train_results_dict=gml_insert_op.executeQuery(insert_task_dict)
    print(res)