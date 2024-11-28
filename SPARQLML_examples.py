import pandas as pd
from Constants import *
import Constants
from Evaluation import Metrics
from KGNET import KGNET
from statistics import mean, median
from SparqlMLaasService.ModelSelector import ModelSelector

if __name__ == '__main__':
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


    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql/', KG_NamedGraph_IRI='http://www.aifb.uni-karlsruhe.de',KG_Prefix='aifb')
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='http://wikikg-v2',KG_Prefix='WikiKG2015_v2')
    # kgnet = KGNET(KG_endpointUrl='http://206.12.97.2:8890/sparql', KG_NamedGraph_IRI='https://yago-knowledge.org', KG_Prefix='yago')
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='https://linkedmdb.org', KG_Prefix='lkmdb')
    # kgnet = KGNET(KG_endpointUrl='http://206.12.97.2:8890/sparql/', KG_NamedGraph_IRI='http://crunchbase-dump-2015-10', KG_Prefix='crunchbase')
    kgnet = KGNET(KG_endpointUrl='http://206.12.97.2:8890/sparql/', KG_NamedGraph_IRI='http://www.biokg.com', KG_Prefix='biokg')

    # kgnet.KGMeta_Governer.insertKGMetadata(sparqlendpoint='http://206.12.98.118:8890/sparql',
    #                                        prefix='lkmdb',
    #                                        namedGraphURI='https://linkedmdb.org',
    #                                        name='lkmdb',
    #                                        description='Linked MDB KG',
    #                                        domain='Movies')

    # types_df = kgnet.getKGNodeEdgeTypes_V2(write_to_file=True, prefix='crunchbase')
    # types_df.to_csv("Datasets/crunchbase_Types.csv")

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

    # TargetEdge = "http://www.wikidata.org/entity/P108" # Employeer
    # "http://www.wikidata.org/entity/Q3571662" Yan Lucn

    # TargetEdge = "dblp:publishedIn"  # profession
    # targetNodeType = "dblp:Publication"
    # MinInstancesPerLabel = 2260

    # TargetEdge = "dblp:bibtexType"  # profession
    # targetNodeType = "dblp:Publication"
    # MinInstancesPerLabel = 1000

    # TargetEdge = "https://dblp.org/Affaliation_Country"
    # targetNodeType = "dblp:Person"
    # MinInstancesPerLabel = 115

    # TargetEdge = "yago:alumniOf"  # profession
    # targetNodeType = "yago:Person"
    # MinInstancesPerLabel = 200

    # TargetEdge = "yago:hasOccupation"  # profession
    # targetNodeType = "yago:Person"
    # MinInstancesPerLabel = 100

    # TargetEdge = "yago:parentOrganization"  # profession
    # targetNodeType = "yago:Organization"
    # MinInstancesPerLabel = 15

    # TargetEdge = "yago:nationality"  # profession
    # targetNodeType = "yago:Person"
    # MinInstancesPerLabel = 500

    # TargetEdge = "yago:award"  # profession
    # targetNodeType = "yago:Person"
    # MinInstancesPerLabel = 50

    # TargetEdge = "yago:memberOf" 
    # targetNodeType = "yago:Person"
    # MinInstancesPerLabel = 500

    # TargetEdge = "yago:deathPlace" 
    # targetNodeType = "yago:Person"
    # MinInstancesPerLabel = 50

    # TargetEdge = "yago:knowsLanguage" 
    # targetNodeType = "yago:Person"
    # MinInstancesPerLabel = 1000
    ##### Creative Work #######
    # TargetEdge = "yago:genre" 
    # targetNodeType = "yago:CreativeWork"
    # MinInstancesPerLabel = 1000

    # TargetEdge = "yago:inLanguage" 
    # targetNodeType = "yago:CreativeWork"
    # MinInstancesPerLabel = 100

    # TargetEdge = "yago:genre" 
    # targetNodeType = "yago:CreativeWork"
    # MinInstancesPerLabel = 1000

    # TargetEdge = "yago:productionCompany" 
    # targetNodeType = "yago:CreativeWork"
    # MinInstancesPerLabel = 100

    # TargetEdge = "yago:publisher" 
    # targetNodeType = "yago:CreativeWork"
    # MinInstancesPerLabel = 400
    ########### Linked IMDB #########
    Linked_IMDB_Dict={
    # "lkmdb:genre":{
    # "targetNodeType": "lkmdb:film",
    # "MinInstancesPerLabel": 500},

    # "lkmdb:country":{
    # "targetNodeType" :"lkmdb:film",
    # "MinInstancesPerLabel" : 100},

    # "lkmdb:film_subject" :{
    # "targetNodeType": "lkmdb:film",
    # "MinInstancesPerLabel": 50},

    # "lkmdb:producer":{
    # "targetNodeType":"lkmdb:film",
    # "MinInstancesPerLabel": 50},

    "lkmdb:language":{
    "targetNodeType":"lkmdb:film",
    "MinInstancesPerLabel":100}
    }
################### crunchbase KG ################
    Linked_crunchbase_Dict={
    # "crunchbase:title":{
    # "targetNodeType": "crunchbase:Person",
    # "MinInstancesPerLabel": 500},
    "crunchbase:organization_name":{
    "targetNodeType": "crunchbase:Person",
    "MinInstancesPerLabel": 20},
    "crunchbase:region_name":{
    "targetNodeType": "crunchbase:Person",
    "MinInstancesPerLabel": 300},
    "crunchbase:country_code":{
    "targetNodeType": "crunchbase:Person",
    "MinInstancesPerLabel": 400}
    }
################### biokg ################
    biokg_Dict={
    # "http://www.biokg.com/drug-property/SUPERCLASS":{
    # "targetNodeType": "biokg:drug",
    # "MinInstancesPerLabel": 50},
    # "http://www.biokg.com/drug-property/CLASS":{
    # "targetNodeType": "biokg:drug",
    # "MinInstancesPerLabel": 50},
    # "http://www.biokg.com/drug-property/KINGDOM":{
    # "targetNodeType": "biokg:drug",
    # # "MinInstancesPerLabel": 50},
    # "http://www.biokg.com/drug-property/PRODUCT":{
    # "targetNodeType": "biokg:drug",
    # "MinInstancesPerLabel": 40},

    # "http://www.biokg.com/protein-property/RELATED_KEYWORD":{
    # "targetNodeType": "biokg:protein",
    # "MinInstancesPerLabel": 15000},
    "http://www.biokg.com/protein-property/ORGANISM_CLASS":{
    "targetNodeType": "biokg:protein",
    "MinInstancesPerLabel": 40000},
    "http://www.biokg.com/FAMILY":{
    "targetNodeType": "biokg:protein",
    "MinInstancesPerLabel": 200},
    "http://www.biokg.com/protein-property/SPECIES":{
    "targetNodeType": "biokg:protein",
    "MinInstancesPerLabel": 15},
    }
########################### YAGO ################    
    # TargetEdge = "yago:memberOf"  # profession
    # targetNodeType = "yago:Country"
    # MinInstancesPerLabel = 6
    
    # TargetEdge = "yago:award"  # profession
    # targetNodeType = "yago:Organization"
    # MinInstancesPerLabel = 6
    for k,v in biokg_Dict.items():
        TargetEdge=k
        print("biokg TargetEdge=",TargetEdge)
        targetNodeType=v['targetNodeType']
        MinInstancesPerLabel=v['MinInstancesPerLabel']        
        for epoch in range(15,31,5):
            for e_size in range(64, 128, 32):
                model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.NodeClassification, targetNodeType=targetNodeType,labelNodeType=None,targetEdge=TargetEdge, GNNMethod=GNN_Methods.Graph_SAINT,TOSG_Pattern=TOSG_Patterns.d1h1,epochs=epoch,emb_size=e_size,MinInstancesPerLabel=MinInstancesPerLabel)
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
    # model_info, transform_info, train_info = kgnet.train_GML(operatorType=Constants.GML_Operator_Types.LinkPrediction,targetEdge="http://swrc.ontoware.org/ontology#author",GNNMethod=GNN_Methods.MorsE)
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='https://dblp2022.org',KG_Prefix='dblp2022')
    # kgnet = KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql', KG_NamedGraph_IRI='http://wikikg-v2',
    #               KG_Prefix='WikiKG2015_v2')
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
    ################################## Single Task SPARQLML##########################
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
    inference_query_NC2 = """
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
    nested_Query = """ select  (count(*) as ?s)
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
    ################################## Multi-Task SPARQLML##########################
    ###################### Wikikdata Multi-Task SPARQLML##################
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
    ###################### DBLP Multi-Task SPARQLML##################
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
    inference_MQuery_dblp_NC_venue_aff = """ 
        prefix dblp:<https://dblp.org/rdf/schema#>
        select distinct  ?Publication ?venue ?country
        from <http://dblp.org>
        where
        {
            ?Publication a dblp:Publication .
            ?Publication dblp:publishedIn ?venue.
            ?Publication dblp:yearOfPublication ?year .
            ?Publication dblp:authoredBy ?Author.
            ?Publication dblp:bibtexType ?ptype.

            ?Author a dblp:Person.
            ?Author <https://dblp.org/Affaliation_Country> ?country.

            ?Publication ?NodeClassifier ?Pred_Venue .
            ?NodeClassifier a <kgnet:types/NodeClassifier>.
            ?NodeClassifier <kgnet:targetNode> dblp:Publication.
            ?NodeClassifier <kgnet:targetEdge> dblp:publishedIn.

            ?Publication ?NodeClassifier2 ?pred_ptype .
            ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
            ?NodeClassifier2 <kgnet:targetNode> dblp:Publication.
            ?NodeClassifier2 <kgnet:targetEdge> dblp:bibtexType.

            ?Author ?NodeClassifier3 ?Pred_aff_country .
            ?NodeClassifier3 a <kgnet:types/NodeClassifier>.
            ?NodeClassifier3 <kgnet:targetNode> dblp:Person.
            ?NodeClassifier3 <kgnet:targetEdge> "https://dblp.org/Affaliation_Country".

            filter(xsd:integer(?year)=2021).            
            # filter(?country not in  ("germany","china",'usa')).
            # filter(?venue not in  ("AAAI","VLDB","ACM",'CoRR')).
            # filter(?venue in  ( "Proc. VLDB Endow.","Symmetry","Entropy","IEEE Trans. Knowl. Data Eng.","Autom.","Bioinform.")).
            # filter(?country not in  ("germany","china")).     
            # filter(?venue not in  ("AAAI","VLDB","ACM")).
            # filter (?ptype in (<http://purl.org/net/nknouf/ns/bibtex#Article>)).

            filter(?country   in  ("germany","china","usa")).     
            filter(?venue  in  ("AAAI","VLDB")).      
            #filter (?ptype in (<http://purl.org/net/nknouf/ns/bibtex#Inproceedings>)).

            filter(str(?Pred_aff_country)    in  ("germany","china","usa")).     
            filter(?Pred_Venue  in  ("AAAI","VLDB")).
            filter (?pred_ptype in (<http://purl.org/net/nknouf/ns/bibtex#Inproceedings>)).

            # filter(str(?country) in ("germany","china")).
            # filter(?venue in ("IEEE Trans. Ind. Informatics","Remote. Sens.")).
            # filter(str(?Pred_aff_country) in ("germany","china")).
            # filter(?Pred_Venue in ("IEEE Trans. Ind. Informatics","Remote. Sens.")).
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
                    #filter (?Pred_aff_country in ("china","usa","germany"))
                    #filter (?Pred_Venue in ("VLDB","ACM"))
                    filter(str(?Pred_aff_country) in ("germany","china")).
                    filter(?Pred_Venue in ("IEEE Trans. Ind. Informatics","Remote. Sens.")).
                }
            group by ?venue
            having (count(*)>1)
            order by DESC(count(*))        
            """
    ############################# YAGO Multitask SPARQLML #######################
    inference_MQuery_yago_NC_nationality_alumniof_occupation = """ 
            prefix yago:<http://schema.org/>
            select distinct  ?Person ?country   ?org 
            # ?org ?parentOrg ?Occupation
            from <https://yago-knowledge.org>
            where
            {
                ?Person a yago:Person .
                ?Person yago:nationality ?country.
                ?Person yago:alumniOf ?org.
                # ?Person yago:hasOccupation ?Occupation .           
                ?Person yago:birthDate ?bdate .
                # ?org yago:parentOrganization ?parentOrg.

                ?Person ?NodeClassifier ?pred_country .
                ?NodeClassifier a <kgnet:types/NodeClassifier>.
                ?NodeClassifier <kgnet:targetNode> yago:Person.
                ?NodeClassifier <kgnet:targetEdge> yago:nationality.

                ?Person ?NodeClassifier2 ?pred_org .
                # ?org ?NodeClassifier2 ?pred_parentOrg .
                ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
                ?NodeClassifier2 <kgnet:targetNode> yago:Person.
                ?NodeClassifier2 <kgnet:targetEdge> yago:alumniOf.
                # ?NodeClassifier2 <kgnet:targetNode> yago:Organization.
                # ?NodeClassifier2 <kgnet:targetEdge> yago:parentOrganization.

                # ?Person ?NodeClassifier3 ?pred_Occupation .
                # ?NodeClassifier3 a <kgnet:types/NodeClassifier>.
                # ?NodeClassifier3 <kgnet:targetNode> yago:Person.
                # ?NodeClassifier3 <kgnet:targetEdge> yago:hasOccupation.

                #filter(xsd:date(?bdate)>=xsd:date('1990-01-01')).
                filter(xsd:date(?bdate)>=xsd:date('1980-01-01')).
                filter (?country in (<http://yago-knowledge.org/resource/United_States>,<http://yago-knowledge.org/resource/United_Kingdom>)).
                filter (?org in (<http://yago-knowledge.org/resource/Harvard_University>,<http://yago-knowledge.org/resource/Ohio_State_University>))
                # filter (?Occupation in (<http://yago-knowledge.org/resource/Writer>,<http://yago-knowledge.org/resource/Actor>,
                #           <http://yago-knowledge.org/resource/Screenwriter>,<http://yago-knowledge.org/resource/Politician>)).
                # filter(?parentOrg in(<http://yago-knowledge.org/resource/University_of_California>,<http://yago-knowledge.org/resource/New_York_University>)).

                filter (?pred_country in (<http://yago-knowledge.org/resource/United_States>,<http://yago-knowledge.org/resource/United_Kingdom>)).
                filter (?pred_org in (<http://yago-knowledge.org/resource/Harvard_University>,<http://yago-knowledge.org/resource/Ohio_State_University>))
                # filter (?pred_Occupation in (<http://yago-knowledge.org/resource/Writer>,<http://yago-knowledge.org/resource/Actor>,
                #           <http://yago-knowledge.org/resource/Screenwriter>,<http://yago-knowledge.org/resource/Politician>)).
                # filter(?pred_parentOrg in(<http://yago-knowledge.org/resource/University_of_California>,<http://yago-knowledge.org/resource/New_York_University>)).
            }       
            """
    inference_MQuery_yago_NC_nationality_organization_occupation = """ 
               prefix yago:<http://schema.org/>
               select distinct  ?Person ?country   ?org 
               # ?org ?parentOrg ?Occupation
               from <https://yago-knowledge.org>
               where
               {
                   ?Person a yago:Person .
                   ?Person yago:nationality ?country.
                   ?Person yago:hasOccupation ?Occupation .  
                   ?Occupation yago:parentOrganization ?parentOrg.         
                   ?Person yago:birthDate ?bdate .

                   ?Person ?NodeClassifier ?pred_country .
                   ?NodeClassifier a <kgnet:types/NodeClassifier>.
                   ?NodeClassifier <kgnet:targetNode> yago:Person.
                   ?NodeClassifier <kgnet:targetEdge> yago:nationality.

                   ?Person ?NodeClassifier2 ?pred_org .
                   # ?org ?NodeClassifier2 ?pred_parentOrg .
                   ?NodeClassifier2 a <kgnet:types/NodeClassifier>.
                   ?NodeClassifier2 <kgnet:targetNode> yago:Person.
                   ?NodeClassifier2 <kgnet:targetEdge> yago:alumniOf.
                   # ?NodeClassifier2 <kgnet:targetNode> yago:Organization.
                   # ?NodeClassifier2 <kgnet:targetEdge> yago:parentOrganization.

                   # ?Person ?NodeClassifier3 ?pred_Occupation .
                   # ?NodeClassifier3 a <kgnet:types/NodeClassifier>.
                   # ?NodeClassifier3 <kgnet:targetNode> yago:Person.
                   # ?NodeClassifier3 <kgnet:targetEdge> yago:hasOccupation.

                   #filter(xsd:date(?bdate)>=xsd:date('1990-01-01')).
                   filter(xsd:date(?bdate)>=xsd:date('1980-01-01')).
                   filter (?country in (<http://yago-knowledge.org/resource/United_States>,<http://yago-knowledge.org/resource/United_Kingdom>)).
                   filter (?org in (<http://yago-knowledge.org/resource/Harvard_University>,<http://yago-knowledge.org/resource/Ohio_State_University>))
                   # filter (?Occupation in (<http://yago-knowledge.org/resource/Writer>,<http://yago-knowledge.org/resource/Actor>,
                   #           <http://yago-knowledge.org/resource/Screenwriter>,<http://yago-knowledge.org/resource/Politician>)).
                   # filter(?parentOrg in(<http://yago-knowledge.org/resource/University_of_California>,<http://yago-knowledge.org/resource/New_York_University>)).

                   filter (?pred_country in (<http://yago-knowledge.org/resource/United_States>,<http://yago-knowledge.org/resource/United_Kingdom>)).
                   filter (?pred_org in (<http://yago-knowledge.org/resource/Harvard_University>,<http://yago-knowledge.org/resource/Ohio_State_University>))
                   # filter (?pred_Occupation in (<http://yago-knowledge.org/resource/Writer>,<http://yago-knowledge.org/resource/Actor>,
                   #           <http://yago-knowledge.org/resource/Screenwriter>,<http://yago-knowledge.org/resource/Politician>)).
                   # filter(?pred_parentOrg in(<http://yago-knowledge.org/resource/University_of_California>,<http://yago-knowledge.org/resource/New_York_University>)).
               }       
               """
    # kgnet=KGNET(KG_endpointUrl='http://206.12.98.118:8890/sparql',KG_NamedGraph_IRI='https://dblp2022.org',KGMeta_endpointUrl="http://206.12.98.118:8890/sparql",RDFEngine=RDFEngine.OpenlinkVirtuoso)
    # kgnet = KGNET(KG_endpointUrl="http://206.12.100.35:5820/kgnet_kgs/query",KGMeta_endpointUrl="http://206.12.100.35:5820/kgnet_kgs/query", KG_NamedGraph_IRI='https://dblp2022.org',RDFEngine=RDFEngine.stardog)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_workField_NC)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_award_NC)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_award_workField_univ_NC)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_award_univ_NC_v2)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_workField_univ_NC_v2)
    # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_Citizenship_Profession_univ_NC_v2)
    # field_medal_number_theory_Kg_humans=['http://www.wikidata.org/entity/Q1398727','http://www.wikidata.org/entity/Q211041','http://www.wikidata.org/entity/Q212063','http://www.wikidata.org/entity/Q220402','http://www.wikidata.org/entity/Q295981','http://www.wikidata.org/entity/Q310769','http://www.wikidata.org/entity/Q333538','http://www.wikidata.org/entity/Q333968','http://www.wikidata.org/entity/Q334045','http://www.wikidata.org/entity/Q369561','http://www.wikidata.org/entity/Q77137']
    # royal_medal_number_theory_Kg_humans=['http://www.wikidata.org/entity/Q184337', 'http://www.wikidata.org/entity/Q184433','http://www.wikidata.org/entity/Q295981', 'http://www.wikidata.org/entity/Q310781','http://www.wikidata.org/entity/Q353426']
    # nethierland_athelete_Kg_humans = ["http://www.wikidata.org/entity/Q128912","http://www.wikidata.org/entity/Q713390","http://www.wikidata.org/entity/Q152125","http://www.wikidata.org/entity/Q2485779","http://www.wikidata.org/entity/Q2491603","http://www.wikidata.org/entity/Q2520376","http://www.wikidata.org/entity/Q258475","http://www.wikidata.org/entity/Q441270","http://www.wikidata.org/entity/Q443882","http://www.wikidata.org/entity/Q697176","http://www.wikidata.org/entity/Q4040442","http://www.wikidata.org/entity/Q313877","http://www.wikidata.org/entity/Q839952","http://www.wikidata.org/entity/Q1852089","http://www.wikidata.org/entity/Q1993932","http://www.wikidata.org/entity/Q2676560","http://www.wikidata.org/entity/Q2933814","http://www.wikidata.org/entity/Q5171587","http://www.wikidata.org/entity/Q832258","http://www.wikidata.org/entity/Q462694","http://www.wikidata.org/entity/Q1071546","http://www.wikidata.org/entity/Q3354942","http://www.wikidata.org/entity/Q465865","http://www.wikidata.org/entity/Q1903135","http://www.wikidata.org/entity/Q2623735","http://www.wikidata.org/entity/Q2624601","http://www.wikidata.org/entity/Q2745460","http://www.wikidata.org/entity/Q15894148","http://www.wikidata.org/entity/Q19848886","http://www.wikidata.org/entity/Q264515","http://www.wikidata.org/entity/Q2134424","http://www.wikidata.org/entity/Q2221157","http://www.wikidata.org/entity/Q2413132","http://www.wikidata.org/entity/Q2423097","http://www.wikidata.org/entity/Q936526","http://www.wikidata.org/entity/Q4661093","http://www.wikidata.org/entity/Q15442354","http://www.wikidata.org/entity/Q6183233","http://www.wikidata.org/entity/Q7155101","http://www.wikidata.org/entity/Q17627341","http://www.wikidata.org/entity/Q15605374","http://www.wikidata.org/entity/Q2355008","http://www.wikidata.org/entity/Q2582703","http://www.wikidata.org/entity/Q2788007","http://www.wikidata.org/entity/Q2325683","http://www.wikidata.org/entity/Q2329623","http://www.wikidata.org/entity/Q142733","http://www.wikidata.org/entity/Q2301217","http://www.wikidata.org/entity/Q2363382","http://www.wikidata.org/entity/Q2475656","http://www.wikidata.org/entity/Q1983169","http://www.wikidata.org/entity/Q2000612","http://www.wikidata.org/entity/Q2106831","http://www.wikidata.org/entity/Q2516725","http://www.wikidata.org/entity/Q2661063","http://www.wikidata.org/entity/Q2864788","http://www.wikidata.org/entity/Q2238626","http://www.wikidata.org/entity/Q2238989","http://www.wikidata.org/entity/Q2241930","http://www.wikidata.org/entity/Q2286746","http://www.wikidata.org/entity/Q2289901","http://www.wikidata.org/entity/Q3027581","http://www.wikidata.org/entity/Q3007712","http://www.wikidata.org/entity/Q2749597","http://www.wikidata.org/entity/Q2933246","http://www.wikidata.org/entity/Q5326298","http://www.wikidata.org/entity/Q4784578","http://www.wikidata.org/entity/Q5076023","http://www.wikidata.org/entity/Q153801","http://www.wikidata.org/entity/Q63325","http://www.wikidata.org/entity/Q686752","http://www.wikidata.org/entity/Q3313872","http://www.wikidata.org/entity/Q238857","http://www.wikidata.org/entity/Q239758","http://www.wikidata.org/entity/Q368492","http://www.wikidata.org/entity/Q706125","http://www.wikidata.org/entity/Q710476","http://www.wikidata.org/entity/Q1867846","http://www.wikidata.org/entity/Q1893161","http://www.wikidata.org/entity/Q2018841","http://www.wikidata.org/entity/Q7802917","http://www.wikidata.org/entity/Q1025876","http://www.wikidata.org/entity/Q2279436","http://www.wikidata.org/entity/Q2764289","http://www.wikidata.org/entity/Q48384","http://www.wikidata.org/entity/Q827524","http://www.wikidata.org/entity/Q2070173","http://www.wikidata.org/entity/Q2095594","http://www.wikidata.org/entity/Q452229","http://www.wikidata.org/entity/Q497623","http://www.wikidata.org/entity/Q1389937","http://www.wikidata.org/entity/Q275854","http://www.wikidata.org/entity/Q474961","http://www.wikidata.org/entity/Q1773622","http://www.wikidata.org/entity/Q1922005","http://www.wikidata.org/entity/Q2718445","http://www.wikidata.org/entity/Q2742873","http://www.wikidata.org/entity/Q2748165","http://www.wikidata.org/entity/Q3416864","http://www.wikidata.org/entity/Q261425","http://www.wikidata.org/entity/Q269855","http://www.wikidata.org/entity/Q271929","http://www.wikidata.org/entity/Q510353","http://www.wikidata.org/entity/Q517635","http://www.wikidata.org/entity/Q2206556","http://www.wikidata.org/entity/Q2321233","http://www.wikidata.org/entity/Q2423919","http://www.wikidata.org/entity/Q3127743","http://www.wikidata.org/entity/Q927805","http://www.wikidata.org/entity/Q5181968","http://www.wikidata.org/entity/Q6215328","http://www.wikidata.org/entity/Q1878190","http://www.wikidata.org/entity/Q2369095","http://www.wikidata.org/entity/Q1921925","http://www.wikidata.org/entity/Q2145902","http://www.wikidata.org/entity/Q2542173","http://www.wikidata.org/entity/Q3197136","http://www.wikidata.org/entity/Q2047317","http://www.wikidata.org/entity/Q2175225","http://www.wikidata.org/entity/Q2180461","http://www.wikidata.org/entity/Q1999453","http://www.wikidata.org/entity/Q2575935","http://www.wikidata.org/entity/Q2423103","http://www.wikidata.org/entity/Q2523424","http://www.wikidata.org/entity/Q2958415","http://www.wikidata.org/entity/Q3165909","http://www.wikidata.org/entity/Q72771","http://www.wikidata.org/entity/Q185389","http://www.wikidata.org/entity/Q2364607","http://www.wikidata.org/entity/Q2557564","http://www.wikidata.org/entity/Q311393","http://www.wikidata.org/entity/Q360042","http://www.wikidata.org/entity/Q1409435","http://www.wikidata.org/entity/Q1873082","http://www.wikidata.org/entity/Q1882937","http://www.wikidata.org/entity/Q169098","http://www.wikidata.org/entity/Q287350","http://www.wikidata.org/entity/Q2032532","http://www.wikidata.org/entity/Q2037137","http://www.wikidata.org/entity/Q2056099","http://www.wikidata.org/entity/Q2256750","http://www.wikidata.org/entity/Q2280454","http://www.wikidata.org/entity/Q2803607","http://www.wikidata.org/entity/Q2928048","http://www.wikidata.org/entity/Q2940829","http://www.wikidata.org/entity/Q2446760","http://www.wikidata.org/entity/Q3280880","http://www.wikidata.org/entity/Q203774","http://www.wikidata.org/entity/Q354629","http://www.wikidata.org/entity/Q973394","http://www.wikidata.org/entity/Q2196060","http://www.wikidata.org/entity/Q2423063","http://www.wikidata.org/entity/Q897510","http://www.wikidata.org/entity/Q1940252","http://www.wikidata.org/entity/Q1970517","http://www.wikidata.org/entity/Q5550485","http://www.wikidata.org/entity/Q5714932","http://www.wikidata.org/entity/Q2374586","http://www.wikidata.org/entity/Q2279403","http://www.wikidata.org/entity/Q2632290","http://www.wikidata.org/entity/Q1961115","http://www.wikidata.org/entity/Q2603665","http://www.wikidata.org/entity/Q2050125","http://www.wikidata.org/entity/Q1982986","http://www.wikidata.org/entity/Q2740079","http://www.wikidata.org/entity/Q4919856","http://www.wikidata.org/entity/Q2520608","http://www.wikidata.org/entity/Q2183012","http://www.wikidata.org/entity/Q2533690","http://www.wikidata.org/entity/Q3140231","http://www.wikidata.org/entity/Q2692004","http://www.wikidata.org/entity/Q2350091","http://www.wikidata.org/entity/Q636089","http://www.wikidata.org/entity/Q2528361","http://www.wikidata.org/entity/Q2579624","http://www.wikidata.org/entity/Q5111206","http://www.wikidata.org/entity/Q216917","http://www.wikidata.org/entity/Q1840423","http://www.wikidata.org/entity/Q1875489","http://www.wikidata.org/entity/Q1887623","http://www.wikidata.org/entity/Q1993457","http://www.wikidata.org/entity/Q2690151","http://www.wikidata.org/entity/Q379502","http://www.wikidata.org/entity/Q739832","http://www.wikidata.org/entity/Q1354265","http://www.wikidata.org/entity/Q2272423","http://www.wikidata.org/entity/Q2800260","http://www.wikidata.org/entity/Q3543752","http://www.wikidata.org/entity/Q46424","http://www.wikidata.org/entity/Q2102663","http://www.wikidata.org/entity/Q2435577","http://www.wikidata.org/entity/Q2459223","http://www.wikidata.org/entity/Q459521","http://www.wikidata.org/entity/Q3386154","http://www.wikidata.org/entity/Q5254375","http://www.wikidata.org/entity/Q1930627","http://www.wikidata.org/entity/Q2650866","http://www.wikidata.org/entity/Q2748094","http://www.wikidata.org/entity/Q2749513","http://www.wikidata.org/entity/Q262062","http://www.wikidata.org/entity/Q489047","http://www.wikidata.org/entity/Q509911","http://www.wikidata.org/entity/Q518422","http://www.wikidata.org/entity/Q1682410","http://www.wikidata.org/entity/Q2177948","http://www.wikidata.org/entity/Q2303304","http://www.wikidata.org/entity/Q1961229","http://www.wikidata.org/entity/Q2222191","http://www.wikidata.org/entity/Q2357082","http://www.wikidata.org/entity/Q2781282","http://www.wikidata.org/entity/Q1950707","http://www.wikidata.org/entity/Q1924855","http://www.wikidata.org/entity/Q2152266","http://www.wikidata.org/entity/Q1909453","http://www.wikidata.org/entity/Q2794527","http://www.wikidata.org/entity/Q2795260","http://www.wikidata.org/entity/Q2170338","http://www.wikidata.org/entity/Q2336652","http://www.wikidata.org/entity/Q2939295","http://www.wikidata.org/entity/Q2375415","http://www.wikidata.org/entity/Q2489517","http://www.wikidata.org/entity/Q1883713","http://www.wikidata.org/entity/Q2006598","http://www.wikidata.org/entity/Q2666163","http://www.wikidata.org/entity/Q291677","http://www.wikidata.org/entity/Q2022531","http://www.wikidata.org/entity/Q2804712","http://www.wikidata.org/entity/Q5171216","http://www.wikidata.org/entity/Q2092290","http://www.wikidata.org/entity/Q2455262","http://www.wikidata.org/entity/Q3277437","http://www.wikidata.org/entity/Q4397542","http://www.wikidata.org/entity/Q4470145","http://www.wikidata.org/entity/Q173972","http://www.wikidata.org/entity/Q454610","http://www.wikidata.org/entity/Q5276314","http://www.wikidata.org/entity/Q1773609","http://www.wikidata.org/entity/Q2715719","http://www.wikidata.org/entity/Q3430723","http://www.wikidata.org/entity/Q261361","http://www.wikidata.org/entity/Q328316","http://www.wikidata.org/entity/Q518219","http://www.wikidata.org/entity/Q1721938","http://www.wikidata.org/entity/Q2231597","http://www.wikidata.org/entity/Q2311688","http://www.wikidata.org/entity/Q2326737","http://www.wikidata.org/entity/Q2389753","http://www.wikidata.org/entity/Q2407817","http://www.wikidata.org/entity/Q2413757","http://www.wikidata.org/entity/Q2430404","http://www.wikidata.org/entity/Q939333","http://www.wikidata.org/entity/Q5550337","http://www.wikidata.org/entity/Q1810754","http://www.wikidata.org/entity/Q1825559","http://www.wikidata.org/entity/Q2224653","http://www.wikidata.org/entity/Q2374734","http://www.wikidata.org/entity/Q2122775","http://www.wikidata.org/entity/Q2159426","http://www.wikidata.org/entity/Q2412238","http://www.wikidata.org/entity/Q2220229","http://www.wikidata.org/entity/Q2049226","http://www.wikidata.org/entity/Q2050257","http://www.wikidata.org/entity/Q1909454","http://www.wikidata.org/entity/Q2247947","http://www.wikidata.org/entity/Q2397808","http://www.wikidata.org/entity/Q2794905","http://www.wikidata.org/entity/Q2043261","http://www.wikidata.org/entity/Q2658539","http://www.wikidata.org/entity/Q2059126","http://www.wikidata.org/entity/Q2334067","http://www.wikidata.org/entity/Q2687946","http://www.wikidata.org/entity/Q3083673","http://www.wikidata.org/entity/Q2452102","http://www.wikidata.org/entity/Q3055177","http://www.wikidata.org/entity/Q2288630","http://www.wikidata.org/entity/Q2805811","http://www.wikidata.org/entity/Q2206661","http://www.wikidata.org/entity/Q2208090","http://www.wikidata.org/entity/Q4714651","http://www.wikidata.org/entity/Q208020","http://www.wikidata.org/entity/Q2362355","http://www.wikidata.org/entity/Q2365639","http://www.wikidata.org/entity/Q2494824","http://www.wikidata.org/entity/Q693849","http://www.wikidata.org/entity/Q1845837","http://www.wikidata.org/entity/Q2017578","http://www.wikidata.org/entity/Q2692582","http://www.wikidata.org/entity/Q6264537","http://www.wikidata.org/entity/Q1364306","http://www.wikidata.org/entity/Q2804494","http://www.wikidata.org/entity/Q1839794","http://www.wikidata.org/entity/Q2080513","http://www.wikidata.org/entity/Q2437754","http://www.wikidata.org/entity/Q229046","http://www.wikidata.org/entity/Q457582","http://www.wikidata.org/entity/Q560725","http://www.wikidata.org/entity/Q973674","http://www.wikidata.org/entity/Q984463","http://www.wikidata.org/entity/Q2990978","http://www.wikidata.org/entity/Q275225","http://www.wikidata.org/entity/Q1771268","http://www.wikidata.org/entity/Q272703","http://www.wikidata.org/entity/Q763833","http://www.wikidata.org/entity/Q1258713","http://www.wikidata.org/entity/Q2152763","http://www.wikidata.org/entity/Q2169615","http://www.wikidata.org/entity/Q2416851","http://www.wikidata.org/entity/Q2593931","http://www.wikidata.org/entity/Q2619538","http://www.wikidata.org/entity/Q2845449","http://www.wikidata.org/entity/Q4933904","http://www.wikidata.org/entity/Q5740436","http://www.wikidata.org/entity/Q6204463","http://www.wikidata.org/entity/Q14519491","http://www.wikidata.org/entity/Q2786973","http://www.wikidata.org/entity/Q1811560","http://www.wikidata.org/entity/Q2269653","http://www.wikidata.org/entity/Q13738620","http://www.wikidata.org/entity/Q2142388","http://www.wikidata.org/entity/Q2633202","http://www.wikidata.org/entity/Q2380263","http://www.wikidata.org/entity/Q2697734","http://www.wikidata.org/entity/Q2569232","http://www.wikidata.org/entity/Q1840004","http://www.wikidata.org/entity/Q2303123","http://www.wikidata.org/entity/Q2201719","http://www.wikidata.org/entity/Q2335192","http://www.wikidata.org/entity/Q2833237","http://www.wikidata.org/entity/Q3158011","http://www.wikidata.org/entity/Q2388743","http://www.wikidata.org/entity/Q2802104","http://www.wikidata.org/entity/Q5665636","http://www.wikidata.org/entity/Q2323566","http://www.wikidata.org/entity/Q2746971","http://www.wikidata.org/entity/Q3381527","http://www.wikidata.org/entity/Q3237188","http://www.wikidata.org/entity/Q4027633","http://www.wikidata.org/entity/Q184218","http://www.wikidata.org/entity/Q189686","http://www.wikidata.org/entity/Q2336382","http://www.wikidata.org/entity/Q2514237","http://www.wikidata.org/entity/Q2519215","http://www.wikidata.org/entity/Q2520287","http://www.wikidata.org/entity/Q1878648","http://www.wikidata.org/entity/Q2690318","http://www.wikidata.org/entity/Q1352440","http://www.wikidata.org/entity/Q2039703","http://www.wikidata.org/entity/Q2054132","http://www.wikidata.org/entity/Q4317458","http://www.wikidata.org/entity/Q169508","http://www.wikidata.org/entity/Q432740","http://www.wikidata.org/entity/Q454488","http://www.wikidata.org/entity/Q458594","http://www.wikidata.org/entity/Q462340","http://www.wikidata.org/entity/Q508504","http://www.wikidata.org/entity/Q469293","http://www.wikidata.org/entity/Q2644802","http://www.wikidata.org/entity/Q514839","http://www.wikidata.org/entity/Q2592745","http://www.wikidata.org/entity/Q1904408","http://www.wikidata.org/entity/Q2295975","http://www.wikidata.org/entity/Q16011718","http://www.wikidata.org/entity/Q1484594","http://www.wikidata.org/entity/Q2478624","http://www.wikidata.org/entity/Q2559291","http://www.wikidata.org/entity/Q540980","http://www.wikidata.org/entity/Q2003785","http://www.wikidata.org/entity/Q1984744","http://www.wikidata.org/entity/Q2136383","http://www.wikidata.org/entity/Q2429118","http://www.wikidata.org/entity/Q2515841","http://www.wikidata.org/entity/Q2186749","http://www.wikidata.org/entity/Q2187420","http://www.wikidata.org/entity/Q2600989","http://www.wikidata.org/entity/Q2933188","http://www.wikidata.org/entity/Q185650","http://www.wikidata.org/entity/Q628422","http://www.wikidata.org/entity/Q2355792","http://www.wikidata.org/entity/Q2490044","http://www.wikidata.org/entity/Q245841","http://www.wikidata.org/entity/Q1850509","http://www.wikidata.org/entity/Q1884934","http://www.wikidata.org/entity/Q2016551","http://www.wikidata.org/entity/Q2681663","http://www.wikidata.org/entity/Q291417","http://www.wikidata.org/entity/Q2244660","http://www.wikidata.org/entity/Q2766387","http://www.wikidata.org/entity/Q2772018","http://www.wikidata.org/entity/Q2938789","http://www.wikidata.org/entity/Q2071792","http://www.wikidata.org/entity/Q4379385","http://www.wikidata.org/entity/Q201367","http://www.wikidata.org/entity/Q433052","http://www.wikidata.org/entity/Q434116","http://www.wikidata.org/entity/Q463156","http://www.wikidata.org/entity/Q273481","http://www.wikidata.org/entity/Q1586971","http://www.wikidata.org/entity/Q2624411","http://www.wikidata.org/entity/Q2719245","http://www.wikidata.org/entity/Q2746960","http://www.wikidata.org/entity/Q271339","http://www.wikidata.org/entity/Q2318321","http://www.wikidata.org/entity/Q2333056","http://www.wikidata.org/entity/Q2423812","http://www.wikidata.org/entity/Q935776","http://www.wikidata.org/entity/Q1964152","http://www.wikidata.org/entity/Q2614802","http://www.wikidata.org/entity/Q18202629","http://www.wikidata.org/entity/Q18692840","http://www.wikidata.org/entity/Q1811818","http://www.wikidata.org/entity/Q2280658","http://www.wikidata.org/entity/Q2479003","http://www.wikidata.org/entity/Q1976636","http://www.wikidata.org/entity/Q1813598","http://www.wikidata.org/entity/Q4846455","http://www.wikidata.org/entity/Q1957984","http://www.wikidata.org/entity/Q2396597","http://www.wikidata.org/entity/Q3908201","http://www.wikidata.org/entity/Q2238309","http://www.wikidata.org/entity/Q2915642","http://www.wikidata.org/entity/Q2676649","http://www.wikidata.org/entity/Q2321758","http://www.wikidata.org/entity/Q2324627","http://www.wikidata.org/entity/Q2491243","http://www.wikidata.org/entity/Q2500026","http://www.wikidata.org/entity/Q241952","http://www.wikidata.org/entity/Q245838","http://www.wikidata.org/entity/Q1856764","http://www.wikidata.org/entity/Q1892249","http://www.wikidata.org/entity/Q2020842","http://www.wikidata.org/entity/Q289121","http://www.wikidata.org/entity/Q735549","http://www.wikidata.org/entity/Q2029164","http://www.wikidata.org/entity/Q2042883","http://www.wikidata.org/entity/Q3050048","http://www.wikidata.org/entity/Q4518257","http://www.wikidata.org/entity/Q1310689","http://www.wikidata.org/entity/Q1838289","http://www.wikidata.org/entity/Q7310284","http://www.wikidata.org/entity/Q229491","http://www.wikidata.org/entity/Q355148","http://www.wikidata.org/entity/Q919601","http://www.wikidata.org/entity/Q2738345","http://www.wikidata.org/entity/Q6763372","http://www.wikidata.org/entity/Q519678","http://www.wikidata.org/entity/Q770117","http://www.wikidata.org/entity/Q776137","http://www.wikidata.org/entity/Q1545416","http://www.wikidata.org/entity/Q1549502","http://www.wikidata.org/entity/Q2180392","http://www.wikidata.org/entity/Q2409956","http://www.wikidata.org/entity/Q1981039","http://www.wikidata.org/entity/Q7141699","http://www.wikidata.org/entity/Q7520410","http://www.wikidata.org/entity/Q2360859","http://www.wikidata.org/entity/Q2281045","http://www.wikidata.org/entity/Q1922487","http://www.wikidata.org/entity/Q2158997","http://www.wikidata.org/entity/Q2863193","http://www.wikidata.org/entity/Q2181949","http://www.wikidata.org/entity/Q1984898","http://www.wikidata.org/entity/Q1997256","http://www.wikidata.org/entity/Q2248073","http://www.wikidata.org/entity/Q2636808","http://www.wikidata.org/entity/Q3182810","http://www.wikidata.org/entity/Q2042615","http://www.wikidata.org/entity/Q2168166","http://www.wikidata.org/entity/Q3018553","http://www.wikidata.org/entity/Q2206273","http://www.wikidata.org/entity/Q2319734","http://www.wikidata.org/entity/Q2316915","http://www.wikidata.org/entity/Q2509763","http://www.wikidata.org/entity/Q726284","http://www.wikidata.org/entity/Q2511529","http://www.wikidata.org/entity/Q259606","http://www.wikidata.org/entity/Q440165","http://www.wikidata.org/entity/Q4770442","http://www.wikidata.org/entity/Q391230","http://www.wikidata.org/entity/Q1889031","http://www.wikidata.org/entity/Q730181","http://www.wikidata.org/entity/Q2271212","http://www.wikidata.org/entity/Q2776434","http://www.wikidata.org/entity/Q1153544","http://www.wikidata.org/entity/Q1295256","http://www.wikidata.org/entity/Q2462169","http://www.wikidata.org/entity/Q173714","http://www.wikidata.org/entity/Q345744","http://www.wikidata.org/entity/Q455411","http://www.wikidata.org/entity/Q497736","http://www.wikidata.org/entity/Q2660496","http://www.wikidata.org/entity/Q271343","http://www.wikidata.org/entity/Q521193","http://www.wikidata.org/entity/Q2225531","http://www.wikidata.org/entity/Q2427118","http://www.wikidata.org/entity/Q3127910","http://www.wikidata.org/entity/Q926482","http://www.wikidata.org/entity/Q1956081","http://www.wikidata.org/entity/Q2587768","http://www.wikidata.org/entity/Q2855717","http://www.wikidata.org/entity/Q2868539","http://www.wikidata.org/entity/Q18032936","http://www.wikidata.org/entity/Q3037170","http://www.wikidata.org/entity/Q2123573","http://www.wikidata.org/entity/Q3285218","http://www.wikidata.org/entity/Q2570394","http://www.wikidata.org/entity/Q2978369","http://www.wikidata.org/entity/Q1840753","http://www.wikidata.org/entity/Q1932918","http://www.wikidata.org/entity/Q2013778","http://www.wikidata.org/entity/Q3148033","http://www.wikidata.org/entity/Q2202429","http://www.wikidata.org/entity/Q1876263","http://www.wikidata.org/entity/Q2162553","http://www.wikidata.org/entity/Q2528044","http://www.wikidata.org/entity/Q2210054","http://www.wikidata.org/entity/Q4459526","http://www.wikidata.org/entity/Q2508912","http://www.wikidata.org/entity/Q402537","http://www.wikidata.org/entity/Q211607","http://www.wikidata.org/entity/Q2354343","http://www.wikidata.org/entity/Q2369770","http://www.wikidata.org/entity/Q2499716","http://www.wikidata.org/entity/Q5605198","http://www.wikidata.org/entity/Q256628","http://www.wikidata.org/entity/Q2572902","http://www.wikidata.org/entity/Q313140","http://www.wikidata.org/entity/Q2004488","http://www.wikidata.org/entity/Q2672583","http://www.wikidata.org/entity/Q2691952","http://www.wikidata.org/entity/Q552349","http://www.wikidata.org/entity/Q507151","http://www.wikidata.org/entity/Q1586429","http://www.wikidata.org/entity/Q1922191","http://www.wikidata.org/entity/Q2661186","http://www.wikidata.org/entity/Q264191","http://www.wikidata.org/entity/Q2118970","http://www.wikidata.org/entity/Q2154479","http://www.wikidata.org/entity/Q2418843","http://www.wikidata.org/entity/Q2423748","http://www.wikidata.org/entity/Q4937803","http://www.wikidata.org/entity/Q6215786","http://www.wikidata.org/entity/Q16013740","http://www.wikidata.org/entity/Q2122679","http://www.wikidata.org/entity/Q2701059","http://www.wikidata.org/entity/Q3113426","http://www.wikidata.org/entity/Q2175546","http://www.wikidata.org/entity/Q1974437","http://www.wikidata.org/entity/Q2103363","http://www.wikidata.org/entity/Q2613710","http://www.wikidata.org/entity/Q2775310","http://www.wikidata.org/entity/Q2872813","http://www.wikidata.org/entity/Q4037563","http://www.wikidata.org/entity/Q2182600","http://www.wikidata.org/entity/Q2186871","http://www.wikidata.org/entity/Q2467097","http://www.wikidata.org/entity/Q2424931","http://www.wikidata.org/entity/Q2677515","http://www.wikidata.org/entity/Q3027752","http://www.wikidata.org/entity/Q3998621","http://www.wikidata.org/entity/Q5631049"]
    # nethierland_baseball_Kg_humans=["http://www.wikidata.org/entity/Q628053","http://www.wikidata.org/entity/Q2488184","http://www.wikidata.org/entity/Q2692182","http://www.wikidata.org/entity/Q2035162","http://www.wikidata.org/entity/Q2804281","http://www.wikidata.org/entity/Q2914750","http://www.wikidata.org/entity/Q3891045","http://www.wikidata.org/entity/Q16231839","http://www.wikidata.org/entity/Q2227840","http://www.wikidata.org/entity/Q2354914","http://www.wikidata.org/entity/Q2501799","http://www.wikidata.org/entity/Q2147000","http://www.wikidata.org/entity/Q2050983","http://www.wikidata.org/entity/Q2000889","http://www.wikidata.org/entity/Q3282574","http://www.wikidata.org/entity/Q2435013","http://www.wikidata.org/entity/Q2522966","http://www.wikidata.org/entity/Q2688540","http://www.wikidata.org/entity/Q2289598","http://www.wikidata.org/entity/Q2695788","http://www.wikidata.org/entity/Q2324754","http://www.wikidata.org/entity/Q5502236","http://www.wikidata.org/entity/Q2035095","http://www.wikidata.org/entity/Q5061926","http://www.wikidata.org/entity/Q1815636","http://www.wikidata.org/entity/Q5639453","http://www.wikidata.org/entity/Q774255","http://www.wikidata.org/entity/Q2219401","http://www.wikidata.org/entity/Q2391114","http://www.wikidata.org/entity/Q11317140","http://www.wikidata.org/entity/Q1849349","http://www.wikidata.org/entity/Q1838923","http://www.wikidata.org/entity/Q2295072","http://www.wikidata.org/entity/Q2065390","http://www.wikidata.org/entity/Q2230115","http://www.wikidata.org/entity/Q2150723","http://www.wikidata.org/entity/Q1909542","http://www.wikidata.org/entity/Q2185556","http://www.wikidata.org/entity/Q3457095","http://www.wikidata.org/entity/Q4608880","http://www.wikidata.org/entity/Q4504717","http://www.wikidata.org/entity/Q599372","http://www.wikidata.org/entity/Q2553574","http://www.wikidata.org/entity/Q3814639","http://www.wikidata.org/entity/Q2278023","http://www.wikidata.org/entity/Q2755752","http://www.wikidata.org/entity/Q2788525","http://www.wikidata.org/entity/Q1376903","http://www.wikidata.org/entity/Q2627003","http://www.wikidata.org/entity/Q2127569","http://www.wikidata.org/entity/Q2133371","http://www.wikidata.org/entity/Q2318574","http://www.wikidata.org/entity/Q5924135","http://www.wikidata.org/entity/Q13441010","http://www.wikidata.org/entity/Q3080297","http://www.wikidata.org/entity/Q2660333","http://www.wikidata.org/entity/Q2173093","http://www.wikidata.org/entity/Q2424168","http://www.wikidata.org/entity/Q2316883","http://www.wikidata.org/entity/Q2682031","http://www.wikidata.org/entity/Q3204914","http://www.wikidata.org/entity/Q2329801","http://www.wikidata.org/entity/Q7291852","http://www.wikidata.org/entity/Q16208159","http://www.wikidata.org/entity/Q2079724","http://www.wikidata.org/entity/Q2736571","http://www.wikidata.org/entity/Q2198339","http://www.wikidata.org/entity/Q3020159","http://www.wikidata.org/entity/Q2714220","http://www.wikidata.org/entity/Q2254869","http://www.wikidata.org/entity/Q5165826","http://www.wikidata.org/entity/Q2372458","http://www.wikidata.org/entity/Q1886645","http://www.wikidata.org/entity/Q2286558","http://www.wikidata.org/entity/Q2913674","http://www.wikidata.org/entity/Q2172021","http://www.wikidata.org/entity/Q4292375","http://www.wikidata.org/entity/Q2846523","http://www.wikidata.org/entity/Q7272450","http://www.wikidata.org/entity/Q16194186","http://www.wikidata.org/entity/Q16232185","http://www.wikidata.org/entity/Q2785266","http://www.wikidata.org/entity/Q2862665","http://www.wikidata.org/entity/Q2014115","http://www.wikidata.org/entity/Q2452102","http://www.wikidata.org/entity/Q2589267","http://www.wikidata.org/entity/Q2594495","http://www.wikidata.org/entity/Q2754901","http://www.wikidata.org/entity/Q2953195","http://www.wikidata.org/entity/Q2084490","http://www.wikidata.org/entity/Q1186030","http://www.wikidata.org/entity/Q1849880","http://www.wikidata.org/entity/Q2242497","http://www.wikidata.org/entity/Q3502696","http://www.wikidata.org/entity/Q233819","http://www.wikidata.org/entity/Q2110632","http://www.wikidata.org/entity/Q3483336","http://www.wikidata.org/entity/Q2790621","http://www.wikidata.org/entity/Q1850110","http://www.wikidata.org/entity/Q1851845","http://www.wikidata.org/entity/Q1943665","http://www.wikidata.org/entity/Q2341102","http://www.wikidata.org/entity/Q2697365","http://www.wikidata.org/entity/Q2554339","http://www.wikidata.org/entity/Q2397926","http://www.wikidata.org/entity/Q1833898","http://www.wikidata.org/entity/Q2693888","http://www.wikidata.org/entity/Q2346418","http://www.wikidata.org/entity/Q2571030","http://www.wikidata.org/entity/Q3026879","http://www.wikidata.org/entity/Q3522791","http://www.wikidata.org/entity/Q2450729","http://www.wikidata.org/entity/Q4869410","http://www.wikidata.org/entity/Q2652518","http://www.wikidata.org/entity/Q2738681","http://www.wikidata.org/entity/Q958652","http://www.wikidata.org/entity/Q2302961","http://www.wikidata.org/entity/Q16197034","http://www.wikidata.org/entity/Q2266290","http://www.wikidata.org/entity/Q3200165","http://www.wikidata.org/entity/Q1953350","http://www.wikidata.org/entity/Q2041040","http://www.wikidata.org/entity/Q2461277","http://www.wikidata.org/entity/Q2578563","http://www.wikidata.org/entity/Q3454116","http://www.wikidata.org/entity/Q3987004","http://www.wikidata.org/entity/Q2763115","http://www.wikidata.org/entity/Q4673162","http://www.wikidata.org/entity/Q4517987","http://www.wikidata.org/entity/Q2510039","http://www.wikidata.org/entity/Q3345858","http://www.wikidata.org/entity/Q1903426","http://www.wikidata.org/entity/Q3116243","http://www.wikidata.org/entity/Q1958266","http://www.wikidata.org/entity/Q2079591","http://www.wikidata.org/entity/Q2300863","http://www.wikidata.org/entity/Q2343006","http://www.wikidata.org/entity/Q3051571","http://www.wikidata.org/entity/Q2554781","http://www.wikidata.org/entity/Q1996834","http://www.wikidata.org/entity/Q2308460","http://www.wikidata.org/entity/Q2887115","http://www.wikidata.org/entity/Q4019695","http://www.wikidata.org/entity/Q2518725","http://www.wikidata.org/entity/Q1845262","http://www.wikidata.org/entity/Q3944801","http://www.wikidata.org/entity/Q3335063","http://www.wikidata.org/entity/Q3176824","http://www.wikidata.org/entity/Q2172735","http://www.wikidata.org/entity/Q2317851","http://www.wikidata.org/entity/Q11310638","http://www.wikidata.org/entity/Q16232148","http://www.wikidata.org/entity/Q2132738","http://www.wikidata.org/entity/Q2360899","http://www.wikidata.org/entity/Q331561","http://www.wikidata.org/entity/Q2054501","http://www.wikidata.org/entity/Q2068442","http://www.wikidata.org/entity/Q2181169","http://www.wikidata.org/entity/Q2097086","http://www.wikidata.org/entity/Q2658755","http://www.wikidata.org/entity/Q2649364","http://www.wikidata.org/entity/Q2914404","http://www.wikidata.org/entity/Q2074763","http://www.wikidata.org/entity/Q2209752","http://www.wikidata.org/entity/Q4334168","http://www.wikidata.org/entity/Q5810870","http://www.wikidata.org/entity/Q5258782","http://www.wikidata.org/entity/Q1902109","http://www.wikidata.org/entity/Q2743885","http://www.wikidata.org/entity/Q2424442","http://www.wikidata.org/entity/Q7280212","http://www.wikidata.org/entity/Q2782354","http://www.wikidata.org/entity/Q2785747","http://www.wikidata.org/entity/Q2280625","http://www.wikidata.org/entity/Q2215685","http://www.wikidata.org/entity/Q2570874","http://www.wikidata.org/entity/Q1839840","http://www.wikidata.org/entity/Q1868153","http://www.wikidata.org/entity/Q2247982","http://www.wikidata.org/entity/Q2308272","http://www.wikidata.org/entity/Q3283193","http://www.wikidata.org/entity/Q1994943","http://www.wikidata.org/entity/Q2205483","http://www.wikidata.org/entity/Q2333811","http://www.wikidata.org/entity/Q2531246","http://www.wikidata.org/entity/Q4542024","http://www.wikidata.org/entity/Q5001511","http://www.wikidata.org/entity/Q2529887","http://www.wikidata.org/entity/Q2694461","http://www.wikidata.org/entity/Q2695678","http://www.wikidata.org/entity/Q2443894","http://www.wikidata.org/entity/Q3931212","http://www.wikidata.org/entity/Q2702412","http://www.wikidata.org/entity/Q976298","http://www.wikidata.org/entity/Q1916105","http://www.wikidata.org/entity/Q959800","http://www.wikidata.org/entity/Q2132997","http://www.wikidata.org/entity/Q2789414","http://www.wikidata.org/entity/Q4915065","http://www.wikidata.org/entity/Q2199594","http://www.wikidata.org/entity/Q2619389","http://www.wikidata.org/entity/Q2458436","http://www.wikidata.org/entity/Q2730778","http://www.wikidata.org/entity/Q4996054","http://www.wikidata.org/entity/Q2288205","http://www.wikidata.org/entity/Q2352159"]
    # dblp_AAAI_germany = ["https://dblp.org/rec/conf/aaai/0002K21", "https://dblp.org/rec/conf/aaai/TomaniB21","https://dblp.org/rec/conf/aaai/KupcsikSKTWSB21", "https://dblp.org/rec/conf/aaai/PhuocEL21","https://dblp.org/rec/conf/aaai/LuoQXCZDYZWCHRL21", "https://dblp.org/rec/conf/aaai/MohrBH21","https://dblp.org/rec/conf/aaai/PanthaplackelAB21","https://dblp.org/rec/conf/aaai/BentertBG0N21", "https://dblp.org/rec/conf/aaai/Baier0M21","https://dblp.org/rec/conf/aaai/Bilo0LLM21", "https://dblp.org/rec/conf/aaai/0002M021","https://dblp.org/rec/conf/aaai/Behnke021", "https://dblp.org/rec/conf/aaai/WilhelmK21","https://dblp.org/rec/conf/aaai/BodirskyK21", "https://dblp.org/rec/conf/aaai/GilT21","https://dblp.org/rec/conf/aaai/FrikhaKKT21", "https://dblp.org/rec/conf/aaai/Dvorak0W21","https://dblp.org/rec/conf/aaai/KotnisLN21", "https://dblp.org/rec/conf/aaai/DalyMAN21","https://dblp.org/rec/conf/aaai/FichteHM21", "https://dblp.org/rec/conf/aaai/Rothe21","https://dblp.org/rec/conf/aaai/0001W21a", "https://dblp.org/rec/conf/aaai/UnalAP21","https://dblp.org/rec/conf/aaai/0001BW21", "https://dblp.org/rec/conf/aaai/BrosowskyKD021","https://dblp.org/rec/conf/aaai/FickertGF0MR21","https://dblp.org/rec/conf/aaai/BesserveSJS21", "https://dblp.org/rec/conf/aaai/HedderichZK21","https://dblp.org/rec/conf/aaai/LuoZCQDZWCHRL21","https://dblp.org/rec/conf/aaai/LawrenceSN21", "https://dblp.org/rec/conf/aaai/LedentMLK21","https://dblp.org/rec/conf/aaai/HollerB21", "https://dblp.org/rec/conf/aaai/MianMV21","https://dblp.org/rec/conf/aaai/SharifzadehBT21", "https://dblp.org/rec/conf/aaai/Potyka21","https://dblp.org/rec/conf/aaai/MathewSYBG021", "https://dblp.org/rec/conf/aaai/PhanBGSRL21","https://dblp.org/rec/conf/aaai/ShaoSSSK21", "https://dblp.org/rec/conf/aaai/WienobstBL21","https://dblp.org/rec/conf/aaai/0001FBBQZDSM021", "https://dblp.org/rec/conf/aaai/0001CDM21","https://dblp.org/rec/conf/aaai/ArtaleJMOW21", "https://dblp.org/rec/conf/aaai/00010Q0LK21","https://dblp.org/rec/conf/aaai/WuLLK21", "https://dblp.org/rec/conf/aaai/HeegerHMMNS21","https://dblp.org/rec/conf/aaai/DennisB0021", "https://dblp.org/rec/conf/aaai/KruseDKS21","https://dblp.org/rec/conf/aaai/LienenH21", "https://dblp.org/rec/conf/aaai/TorralbaSKS021","https://dblp.org/rec/conf/aaai/NeiderGGT0021", "https://dblp.org/rec/conf/aaai/TanNDNB21","https://dblp.org/rec/conf/aaai/NayyeriVA021", "https://dblp.org/rec/conf/aaai/Torralba21","https://dblp.org/rec/conf/aaai/ZhouLHLZK21"]
    # dblp_AAAI_ACM_VLDB_germany_china_2021=pd.read_csv("SPARQLML_HardAnswerResults/dblp_AAAI_ACM_VLDB_germany_china_2021.csv",header=None,sep=",")[0].tolist()
    # dblp_Informatics_RemoteSens_germany_china_2022 = pd.read_csv("SPARQLML_HardAnswerResults/dblp_Informatics_RemoteSens_germany_china_2022.csv", header=None, sep=",")[0].tolist()
    # dblp_notin_AAAI_ACM_VLDB_CoRR_germany_china_usa_2022 =pd.read_csv("SPARQLML_HardAnswerResults/dblp_notin_AAAI_ACM_VLDB_CoRR_germany_china_usa_2022.csv", header=None, sep=",")[0].tolist()
    # dblp_notin_AAAI_ACM_VLDB_CoRR_germany_china_usa_2021 = pd.read_csv("SPARQLML_HardAnswerResults/dblp_notin_AAAI_ACM_VLDB_CoRR_germany_china_usa_2021.csv", header=None,sep=",")[0].tolist()
    # dblp_AAAI_VLDB_germany_china_usa_Inproccesding_2021 =pd.read_csv("SPARQLML_HardAnswerResults/dblp_AAAI_VLDB_germany_china_usa_Inproccesding_2021.csv", header=None,sep=",")[0].tolist()
    # dblp_AAAI_VLDB_germany_china_usa_Inproccesding_gte2019 = pd.read_csv("SPARQLML_HardAnswerResults/dblp_AAAI_VLDB_germany_china_usa_Inproccesding_gte2019.csv", header=None, sep=",")[0].tolist()
    # dblp_CVPR_ICASSP_IGARSS_ICML_NeurIPS_germany_china_usa_Inproccesding_gte2020=pd.read_csv("SPARQLML_HardAnswerResults/dblp_CVPR_ICASSP_IGARSS_ICML_NeurIPS_germany_china_usa_Inproccesding_gte2020.csv", header=None, sep=",")[0].tolist()
    # dblp_in_VLDB_Symmetry_Entropy_IEEEData_Eng_Autom_Bioinform_notin_germany_in_Article_gte2020 = pd.read_csv("SPARQLML_HardAnswerResults/dblp_in_VLDB_Symmetry_Entropy_IEEEData-Eng_Autom_Bioinform_notin_germany_in_Article_gte2020.csv",header=None, sep=",")[0].tolist()
    # True_Targets_Path = "yago_in_Us_UK_in_Harvard_Ohio_db1990.csv"
    # True_Targets_Path="yago_in_Us_UK_in_Harvard_Ohio_in_writer_politician_db1980.csv"
    # True_Targets_Path="yago_in_Us_UK_in_Writer_Actor_Screenwriter_Politician_db1980.csv"
    # True_Targets_Path="yago_in_Us_UK_in_Writer_Actor_Screenwriter_Politician_in_univCalifornia_univNY_db1980.csv"
    # True_Targets_df = pd.read_csv(f"SPARQLML_HardAnswerResults/{True_Targets_Path}", header=None, sep=",")[0].tolist()
    # Real_target_node = list(set([str(x).replace("\"", "").strip() for x in True_Targets_df]))
    # Real_target_node = nethierland_baseball_Kg_humans
    # target_node="human"
    # target_node = "Publication"
    target_node = "Person"
    import itertools

    # # n_tasks = 2
    # n_runs = 3
    # patterns = ["1p", "2p", "3p", "2i", "ip", "pi", "3i", "2u", "up", "2in", "3in", "inp", "pni", "pin"]
    # pattern_inst_idx = 0
    # ############### use SPARQLML Benchmark ###############
    # ben_root_path = "/shared_mnt/github_repos/KGNET/SPARQLML-Ben/"
    # SPARQLML_Ben_df = pd.read_csv(f"{ben_root_path}SPARQLML-Ben-Yago4.tsv", sep="\t")
    # for pattern_idx in range(10, len(patterns)):
    #     pattern = patterns[pattern_idx]
    #     print(f">>>>>>>>>>>>>>>>>>>>>>>>> pattern={pattern} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     SPARQLML_Ben_patten_df = SPARQLML_Ben_df[SPARQLML_Ben_df["pattern"] == pattern]
    #     SPARQLML_Ben_patten_df = SPARQLML_Ben_patten_df.reset_index(drop=True)
    #     patten_query = SPARQLML_Ben_patten_df["SPARQLML_query"][pattern_inst_idx]
    #     pred_col = SPARQLML_Ben_patten_df["pred_col"][pattern_inst_idx]
    #     real_col = SPARQLML_Ben_patten_df["real_col"][pattern_inst_idx]
    #     target_col = SPARQLML_Ben_patten_df["target_col"][pattern_inst_idx]
    #     n_tasks = SPARQLML_Ben_patten_df["n_tasks"][pattern_inst_idx]
    #     query_select_both_cols = SPARQLML_Ben_patten_df["query_select_both_cols"][pattern_inst_idx] == 1
    #     eval_metric = SPARQLML_Ben_patten_df["metric"][pattern_inst_idx]
    #     try:
    #         patten_query_results_df = pd.read_csv(
    #             f"{ben_root_path}" + SPARQLML_Ben_patten_df["results_file"][pattern_inst_idx], sep="\t")
    #     except:
    #         print(f"file not exist:{ben_root_path + SPARQLML_Ben_patten_df['results_file'][pattern_inst_idx]}")
    #     ######################################################
    #     # piplines=list(itertools.permutations(range(0,n_tasks)))
    #     # print(f"True_Targets_Path={True_Targets_Path}")
    #     DAGExecPlans, DAG, decomposedSubqueries = kgnet.getSPARQLMLExecQueryPlans(patten_query)
    #     CostModelParams = {'accuracyW': 0.7, 'inferTimeW': 0.3}
    #     SPARQLMLQueryPlansCostDict, SPARQLMLQueryPlansCostLst = kgnet.getSPARQLMLQueryPlansCost(DAGExecPlans, DAG,
    #                                                                                             decomposedSubqueries,
    #                                                                                             CostModelParams)
    #     bestPlanIdx = ModelSelector.getBestPlanIdx(SPARQLMLQueryPlansCostLst, w1=0.8, w2=0.2)
    #     print(f"Best Execution Plan Idx:{bestPlanIdx}")
    #     print(f"DAGExecPlans={['->'.join(elem) for elem in DAGExecPlans]}")
    #     print(f"Possible DAG Execution Plans Count ={len(DAGExecPlans)}")
    #     for ExecPlanIdx in range(0, len(DAGExecPlans)):
    #         print(f"########## Execution Plan:({'->'.join(DAGExecPlans[ExecPlanIdx])})#################")
    #         for infer_mode in range(0, 2):
    #             exec_time = []
    #             true_pred_lst, not_pred_lst, false_pred_lst, f1_pred_lst = [], [], [], []
    #             kgwise_full_batch = infer_mode == 0
    #             print(f"*********kgwise_full_batch={kgwise_full_batch}********")
    #             for run in range(0, n_runs):
    #                 print(f"###### Run({run})#####")
    #                 # inference_query_wikidata_Citizenship_Profession_univ_NC_v2
    #                 # resDF, MetaQueries,query_time_sec = kgnet.executeSPARQLMLInferenceQuery(inference_MQuery_dblp_NC_venue_aff,pipline=pipline_no)
    #                 # resDF, MetaQueries, query_time_sec = kgnet.executeSPARQLMLInferenceQuery(inference_MQuery_yago_NC_nationality_alumniof_occupation,pipline=pipline_no,kgwise_full_batch=kgwise_full_batch)
    #                 resDF, MetaQueries, query_time_sec = kgnet.executeSPARQLMLInferenceQuery(patten_query,
    #                                                                                          ExecPlanIdx=ExecPlanIdx,
    #                                                                                          kgwise_full_batch=kgwise_full_batch)
    #                 # resDF, MetaQueries, query_time_sec = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_Citizenship_Profession_univ_NC_v2,in_pipline=True)
    #                 exec_time.append(query_time_sec)
    #                 # resDF,MetaQueries=kgnet.executeSPARQLMLInferenceQuery(inference_MQuery_dblp2022_NC_LP)
    #                 # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(inference_query_wikidata_award_workField_NC)
    #                 # resDF, MetaQueries = kgnet.executeSPARQLMLInferenceQuery(nested_Query)

    #                 print("query_time_sec=", query_time_sec)
    #                 # print(resDF)
    #                 if eval_metric == "accuracy":
    #                     resDF['match_pred'] = resDF[real_col] == resDF[pred_col]
    #                     true_predicted_targets_set = set(resDF[resDF['match_pred'] == True][target_col].unique())
    #                     false_predicted_targets_set = set(resDF[target_col].unique()) - true_predicted_targets_set
    #                     # pred_target_set = [str(elem).replace("\"", "").strip().split("/")[-1] for elem in resDF[pred_col].tolist()]
    #                     # if query_select_both_cols:
    #                     #     Real_target_node = [str(x).replace("\"", "").strip().split("/")[-1] for x in resDF[real_col].tolist()]
    #                     # else:
    #                     #     Real_target_node = [str(x).replace("\"", "").strip().split("/")[-1] for x in patten_query_results_df[real_col].tolist()]
    #                     # acc=Metrics.get_accuracy_score(Real_target_node,pred_target_set)
    #                     acc = len(true_predicted_targets_set) / (
    #                                 len(true_predicted_targets_set) + len(false_predicted_targets_set))
    #                     true_pred_lst.append(acc)
    #                     false_pred_lst.append(1 - acc)
    #                     # f1_score = Metrics.get_f1_score(Real_target_node, pred_target_set)
    #                     # f1_pred_lst.append(f1_score)
    #                     print(f"accuracy={acc}")
    #                 elif eval_metric == "precision":
    #                     pred_target_set = set([str(elem).replace("\"", "").strip().split("/")[-1] for elem in
    #                                            set(resDF[pred_col].tolist())])
    #                     Real_target_node = list(set([str(x).replace("\"", "").strip().split("/")[-1] for x in
    #                                                  patten_query_results_df[real_col].tolist()]))
    #                     true_pred_lst.append(
    #                         len(pred_target_set.intersection(set(Real_target_node))) / len(Real_target_node))
    #                     print(f"True Predictions Ratio:{true_pred_lst[-1]}")
    #                     not_pred_lst.append(len(set(Real_target_node) - pred_target_set) / len(Real_target_node))
    #                     print(f"Not Predicted Ratio:{not_pred_lst[-1]}")
    #                     false_pred_lst.append(len(pred_target_set - set(Real_target_node)))
    #                     print(f"Inductive Predictions Count:{false_pred_lst[-1]}")
    #                     print(MetaQueries)
    #                     # print("candidateSparqlQuery=",MetaQueries['candidateSparqlQuery'])
    #             print(f"Pipline=({ExecPlanIdx})")
    #             print(
    #                 f"avg_time={mean(exec_time)} | median_time={median(exec_time)} | min={min(exec_time)} | max={max(exec_time)}.")
    #             if len(f1_pred_lst) > 0:
    #                 print(
    #                     f"avg_f1_preds={mean(f1_pred_lst)} | median_f1_preds={median(f1_pred_lst)} | min={min(f1_pred_lst)} | max={max(f1_pred_lst)}.")
    #             if len(true_pred_lst) > 0:
    #                 print(
    #                     f"avg_true_preds={mean(true_pred_lst) * 100}% | median_true_preds={median(true_pred_lst) * 100}% | min={min(true_pred_lst) * 100}% | max={max(true_pred_lst) * 100}%.")
    #             if len(not_pred_lst) > 0:
    #                 print(
    #                     f"avg_not_preds={mean(not_pred_lst) * 100}% | median_not_preds={median(not_pred_lst) * 100}% | min={min(not_pred_lst) * 100}% | max={max(not_pred_lst) * 100}%.")
    #             if len(false_pred_lst) > 0:
    #                 print(
    #                     f"avg_Inductive_pred_counts={mean(false_pred_lst) * 100}% | median_Inductive_pred_counts={median(false_pred_lst) * 100}% | min={min(false_pred_lst) * 100}% | max={max(false_pred_lst) * 100}%.")
    #     #############################################