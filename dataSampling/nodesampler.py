import sys
import os
from pathlib import Path
sys.path.insert(0,'/home/KG-EaaS')
import csv
import pandas
import  pandas as pd
from classificationModels.classificationModel import RandomForest
from GMLQueryRewriter.queryRewriter_old import queryRewriter
from embeddingServices.KGEmbeddingService import embeddingAsaService
from embeddingServices.KGEmbeddingService import startHttpServerThread
from embeddingServices.embeddingStore import pickleInMemoryStore
from embeddingServices.embeddingStore import FaissInMemoryStore
from embeddingServices.similarityMetrics import euclideanDistance
from RDFEngineManager.UDF_Manager_Virtuoso import openlinkVirtuosoEndpoint
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from embeddingServices.similarityMetrics import cosineSimilarity
from embeddingServices.generateKGEmbeddings import ComplEx
from embeddingServices.generateKGEmbeddings import DistMult
from embeddingServices.generateKGEmbeddings import TransE
from embeddingServices.generateKGEmbeddings import HolE
import datetime
if __name__ == '__main__':
    use_generated_embedding = False
    sparqlEndpoint1 = openlinkVirtuosoEndpoint()
    print(type(sparqlEndpoint1))
    print(sparqlEndpoint1.version)
    sparqlEndpoint1.endpointUrl="http://172.17.0.1:8890/sparql"
    publications_type="conf"
    affaliations_Coverage_df=pd.read_csv("/shared_mnt/DBLP/Sparql_Sampling_conf/BDLP_Papers_Per_Affaliation_conf.csv")
    affaliations_Coverage_df=affaliations_Coverage_df[affaliations_Coverage_df["do_train"]==1].reset_index(drop=True)
    # print(df_queries)
    cs = cosineSimilarity()
    StarQuery=""" select distinct ?pub as ?subject ?p as ?predicate ?o as ?object
                from <http://dblp.org>
                where
                {
                    ?auth <https://dblp.org/rdf/schema#primaryAffiliation> ?aff.
                    ?pub <https://dblp.org/rdf/schema#authoredBy> ?auth.
                    ?pub <https://dblp.org/rdf/schema#type> <https://dblp.org/rec/type/conf>.
                    ?pub <https://dblp.org/rdf/schema#yearOfPublication> ?year.
                    ?pub ?p ?o.
                    filter(xsd:Integer(?year)>=2015)
                    filter(!isBlank(?o)) 
                }
                limit 1000000
                """
    BStarQuery = """select distinct ?pub as ?subject ?p as ?predicate ?o as ?object
                where
                {
                    {
                        select distinct ?pub as ?pub ?p  ?o 
                        from <http://dblp.org>
                        where
                        {
                            ?auth <https://dblp.org/rdf/schema#primaryAffiliation> ?aff.
                            ?pub <https://dblp.org/rdf/schema#authoredBy> ?auth.
                            ?pub <https://dblp.org/rdf/schema#type> <https://dblp.org/rec/type/conf>.
                            ?pub <https://dblp.org/rdf/schema#yearOfPublication> ?year.
                            ?pub ?p ?o.
                            filter(xsd:Integer(?year)>=2015)
                            filter(!isBlank(?o))
                        }
                    }
                    union
                    {
                        select distinct ?s2 as ?pub ?p2 as ?p ?pub2 as ?o
                        from <http://dblp.org>
                        where
                        {
                            ?auth2 <https://dblp.org/rdf/schema#primaryAffiliation> ?aff.
                            ?pub2 <https://dblp.org/rdf/schema#authoredBy> ?auth2.
                            ?pub2 <https://dblp.org/rdf/schema#type> <https://dblp.org/rec/type/conf>.
                            ?pub2 <https://dblp.org/rdf/schema#yearOfPublication> ?year2.
                            ?s2 ?p2 ?pub2.
                            filter(xsd:Integer(?year2)>=2015)
                            filter(!isBlank(?pub2))
                        }
                    }
                }
                limit 1000000 """
    PathQuery = """select distinct ?pub as ?subject ?p as ?predicate ?o as ?object
                where
                {
                    {
                        select distinct ?pub as ?pub ?p  ?o 
                        from <http://dblp.org>
                        where
                        {
                            ?auth <https://dblp.org/rdf/schema#primaryAffiliation> ?aff.
                            ?pub <https://dblp.org/rdf/schema#authoredBy> ?auth.
                            ?pub <https://dblp.org/rdf/schema#type> <https://dblp.org/rec/type/conf>.
                            ?pub <https://dblp.org/rdf/schema#yearOfPublication> ?year.
                            ?pub ?p ?o.
                            filter(xsd:Integer(?year)>=2015)
                            filter(!isBlank(?o))
                        }
                    }
                    union
                    {
                        select distinct ?o as ?pub ?p2 as ?p ?o2 as ?o 
                        from <http://dblp.org>
                        where
                        {
                            ?auth <https://dblp.org/rdf/schema#primaryAffiliation> ?aff.
                            ?pub <https://dblp.org/rdf/schema#authoredBy> ?auth.
                            ?pub <https://dblp.org/rdf/schema#type> <https://dblp.org/rec/type/conf>.
                            ?pub <https://dblp.org/rdf/schema#yearOfPublication> ?year.
                            ?pub ?p ?o.
                            ?o ?p2 ?o2.
                            filter(xsd:Integer(?year)>=2015)
                            filter(!isBlank(?o))
                            filter(!isBlank(?o2))
                        }
                     }
                }                    
                limit 1000000 """
    BPathQuery="""select distinct ?pub as ?subject ?p as ?predicate ?o as ?object
                where
                {
                    {
                        select distinct ?pub as ?pub ?p  ?o 
                        from <http://dblp.org>
                        where
                        {
                            ?auth <https://dblp.org/rdf/schema#primaryAffiliation> ?aff.
                            ?pub <https://dblp.org/rdf/schema#authoredBy> ?auth.
                            ?pub <https://dblp.org/rdf/schema#type> <https://dblp.org/rec/type/conf>.
                            ?pub <https://dblp.org/rdf/schema#yearOfPublication> ?year.
                            ?pub ?p ?o.
                            filter(xsd:Integer(?year)>=2015)
                            filter(!isBlank(?o))
                        }
                    }
                union
                {
                    select distinct ?o as ?pub ?p2 as ?p ?o2 as ?o 
                    from <http://dblp.org>
                    where
                    {
                        ?auth <https://dblp.org/rdf/schema#primaryAffiliation> ?aff.
                        ?pub <https://dblp.org/rdf/schema#authoredBy> ?auth.
                        ?pub <https://dblp.org/rdf/schema#type> <https://dblp.org/rec/type/conf>.
                        ?pub <https://dblp.org/rdf/schema#yearOfPublication> ?year.
                        ?pub ?p ?o.
                        ?o ?p2 ?o2.
                        filter(xsd:Integer(?year)>=2015)
                        filter(!isBlank(?o))
                        filter(!isBlank(?o2))
                    }
                }
                union
                {
                    select distinct ?s2 as ?pub ?p3 as ?p ?pub as ?o 
                    from <http://dblp.org>
                    where
                    {
                        ?auth <https://dblp.org/rdf/schema#primaryAffiliation> ?aff.
                        ?pub <https://dblp.org/rdf/schema#authoredBy> ?auth.
                        ?pub <https://dblp.org/rdf/schema#type> <https://dblp.org/rec/type/conf>.
                        ?pub <https://dblp.org/rdf/schema#yearOfPublication> ?year.
                        ?pub ?p ?o.
                        ?s2 ?p3 ?pub.
                        filter(xsd:Integer(?year)>=2015)
                        filter(!isBlank(?o))
                        filter(!isBlank(?pub))
                    }
                }
            }
            limit 1000000"""
    dic_results={}
    sampledQueries={
                    "StarQuery":StarQuery,
                    "BStarQuery":BStarQuery,
                    "PathQuery":PathQuery,
                    "BPathQuery":BPathQuery
                    }
    for i,row in affaliations_Coverage_df.iterrows():
        if i>365 and i%2==0:
            for sample_key in sampledQueries.keys():
                dataset="OBGN_QM_DBLP_"+publications_type+"_"+sample_key+"Usecase_"+str(int(row["Q_idx"]))+"_"+str(str(row["affiliation"]).strip().replace(" ","_").replace("/","_").replace(",","_"))
                dic_results[dataset]={}
                dic_results[dataset]["q_idx"]=int(row["Q_idx"])
                dic_results[dataset]["usecase"] = dataset
                dic_results[dataset]["sample_key"] = sample_key
                # print("dataset=",dataset)
                aff_str = str(row["affiliation"])
                aff_str=aff_str.replace('\n','').replace("\"","")
                query=sampledQueries[sample_key].replace("?aff","\""+aff_str+"\"")
                # print("query=",query)
                print("usecase=",dataset)
                start_t = datetime.datetime.now()
                query_rows_count=query.replace("distinct ?pub as ?subject ?p as ?predicate ?o as ?object","count (DISTINCT * ) as ?rows_count")
                kg_df = sparqlEndpoint1.executeSparqlQuery(query_rows_count)
                q_triples_count=int(str(kg_df["rows_count"][0]))
                dic_results[dataset]["triples_count"]=q_triples_count
                print("q_triples_count=",q_triples_count)
                q_limit=1000000
                q_iter_count=int(q_triples_count/q_limit)+1
                print("q_iter_count=", q_iter_count)
                kg_df=None
                for idx in range(0,q_iter_count):
                    batch_query = query.replace("limit 1000000","offset "+str((idx*q_limit))+" limit "+str(q_limit))
                    batch_q_df = sparqlEndpoint1.executeSparqlQuery(batch_query)
                    if kg_df is None:
                        kg_df=batch_q_df
                    else:
                        kg_df =kg_df.append(batch_q_df)
                    print("query idx=",idx," len(triples)=",len(kg_df))
                end_t = datetime.datetime.now()
                print("Query_time=",end_t - start_t," sec.")
                dic_results[dataset]["query_time"] = (end_t - start_t).total_seconds()
                kg_df.to_csv("/shared_mnt/DBLP/Sparql_Sampling_"+publications_type +"/Conf_Y2015/"+dataset+".csv",index=False)         
                pd.DataFrame(dic_results).transpose().to_csv( "/shared_mnt/DBLP/Sparql_Sampling_"+publications_type+"/OGBN_DBLP_Sampled_"+publications_type+"_2015_Uscases_query_times"+".csv",index=False)