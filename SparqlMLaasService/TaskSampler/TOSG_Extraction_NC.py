import pandas as pd
import datetime
import requests
import traceback
import sys
import argparse
from threading import Thread
import threading
from  Constants import RDFEngine
import validators
lock = threading.Lock()
def ExecQueryAndWriteOutput(endpoint_url,query, offset, batch_size,batches_count, f,RDFEngine=RDFEngine.OpenlinkVirtuoso):
    start_t = datetime.datetime.now()
    r_query = query.replace("?offset", str(offset))
    r_query = r_query.replace("?limit", str(batch_size))
    body = {'query': r_query}
    headers = {'Content-Type': 'application/x-www-form-urlencoded',
               'Accept-Encoding': 'gzip',
               'Accept':  ('text/tab-separated-values; charset=UTF-8' if  RDFEngine==RDFEngine.OpenlinkVirtuoso else 'text/tsv')
               }
    r = requests.post(endpoint_url, data=body, headers=headers)
    lock.acquire()
    f.write(r.text.replace(""""subject"	"predicate"	"object"\n""", ""))
    end_t = datetime.datetime.now()
    print("Query idx: ", (offset / batch_size), "/", batches_count, " records count=", len(r.text.split('\n')),
          " time=", end_t - start_t, " sec.")
    lock.release()
def ExecQuery(endpoint_url,query):
    start_t = datetime.datetime.now()
    body = {'query': query}
    headers = {
               #'Content-Type': 'application/sparql-update',
               'Content-Type':"application/x-www-form-urlencoded",
               'Accept-Encoding': 'gzip',
               'Accept': 'text/tab-separated-values; charset=UTF-8'}
    r = requests.post(endpoint_url, data=body, headers=headers)
    return r.text.split('\n')[1]
def get_KG_entity_types(graph_uri):
    query = """select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
              from <""" + graph_uri + """>
              where
              {
                  select  ?s as ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> as ?p ?o as ?o 
                  from <""" + graph_uri + """>
                  where
                  {
                       ?s a ?o.
                  }
              }
              offset ?offset
              limit ?limit 
           """
    return query
# def get_d1h1_TargetListquery(graph_uri,target_lst):
#     query="""select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
#            from <"""+graph_uri+""">
#            where
#            {
#              ?s ?p ?o
#              values ?s {$VT_Values$}
#              }"""
#     query=query.replace("$VT_Values$"," ".join(target_lst))
#     return query
def get_d1h1_TargetListquery(graph_uri,target_lst):
    query="""select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
           from <"""+graph_uri+""">
           where
           {
             ?s ?p ?o.
             values ?s {$VT_Values$}
           }"""
    query_o_type = """
    select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
    from <""" + graph_uri + """>
    where
    {
       select  (?o as ?s) ('http://www.w3.org/1999/02/22-rdf-syntax-ns#type' as ?p) (?otype as ?o)
       where
       {
         ?s ?p ?o.
         ?o a ?otype.
         values ?s {$VT_Values$}
       }
    }"""
    query=query.replace("$VT_Values$"," ".join(target_lst))
    query_o_type = query_o_type.replace("$VT_Values$", " ".join(target_lst))
    return [query,query_o_type]
# def get_d1h1_query(graph_uri,target_rel_uri,stype=None,otype=None,tragetNode_filter_statments=None):
#     query="""select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
#     query_o_type = query_o_type.replace("$VT_Values$", " ".join(target_lst))
#     return [query,query_o_type]
def get_d1h1_query(graph_uri,target_rel_uri,prefixs=None,stype=None,otype=None,tragetNode_filter_statments=None):
    query=""
    if prefixs and len(prefixs.keys())>0:
        for prefix in prefixs.keys():
            query+="prefix "+prefix+":<"+prefixs[prefix]+">\n"
    query+="""select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)\n
           where
           {
                select ?s ?p ?o
                from <"""+graph_uri+""">
                where
                {"""
    query += ("" if target_rel_uri is None else " ?s " + target_rel_uri + " ?label.\n" if (prefixs is not None and target_rel_uri.split(":")[0] in prefixs.keys()) else " ?s  <" + target_rel_uri + "> ?label .\n" if validators.url(target_rel_uri) else "")
    query+=("" if stype is None else " ?s a "+stype+" .\n" if (prefixs is not None and stype.split(":")[0] in prefixs.keys()) else " ?s a <"+stype+"> .\n" if  validators.url(stype) else "?s a ?stype. \n filter(?stype="+"\""+stype+"\") . \n")
    query += ("" if otype is None else " ?label a " + otype + " .\n" if (prefixs is not None and otype.split(":")[0] in prefixs.keys()) else " ?label a <" + otype + "> .\n" if validators.url(otype) else "?label a ?ltype. \n filter(?ltype="+"\""+otype+"\") . \n")
    query+=""" ?s ?p ?o.\n
              filter(!isBlank(?o)).\n"""
    if tragetNode_filter_statments:
        # query += tragetNode_filter_statments
        for statement in tragetNode_filter_statments:
            query+=statement+"\n"
    query += """} }
          offset ?offset
          limit ?limit """

    query_o_t=""
    if prefixs and len(prefixs.keys()) > 0:
        for prefix in prefixs.keys():
            query_o_t += "prefix " + prefix + ":<" + prefixs[prefix] + ">\n"
    query_o_t += """select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
               where
               {
                    select ?o as ?s 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'  as ?p ?ot as ?o
                    from <"""+graph_uri+""">
                    where
                    {\n"""
    query_o_t += ("" if target_rel_uri is None else " ?s " + target_rel_uri + " ?label.\n" if (prefixs is not None and target_rel_uri.split(":")[0] in prefixs.keys()) else " ?s  <" + target_rel_uri + "> ?label .\n" if validators.url(target_rel_uri) else "")
    query_o_t += ("" if stype is None else " ?s a " + stype + " .\n" if (prefixs is not None and stype.split(":")[0] in prefixs.keys()) else " ?s a <" + stype + "> .\n" if validators.url(stype) else "?s a ?stype. \n filter(?stype=" + "\"" + stype + "\") . \n")
    query_o_t += ("" if otype is None else " ?label a " + otype + " .\n" if (prefixs is not None and otype.split(":")[0] in prefixs.keys()) else " ?label a <" + otype + "> .\n" if validators.url(otype) else "?label a ?ltype. \n filter(?ltype=" + "\"" + otype + "\") . \n")
    query_o_t += """ ?s ?p ?o.
                     ?o a ?ot.
                     filter(!isLiteral(?o)).\n"""
    if tragetNode_filter_statments:
        # query_o_t += tragetNode_filter_statments
        for statement in tragetNode_filter_statments:
            query_o_t += statement+"\n"
    query_o_t += """} }  
               offset ?offset
              limit ?limit """

    return query,query_o_t
def get_d2h1_query(graph_uri,target_rel_uri,stype=None,otype=None,tragetNode_filter_statments=None):
    query="""select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
            where
            {
                 select ?o2 as ?s  ?p2 as ?p  ?s as ?o
                 from <"""+graph_uri+""">
                 where
                 {
                    ?o2 ?p2 ?s.
                    filter(!isBlank(?o2)).
                    {
                        select distinct ?s 
                        from <"""+graph_uri+""">
                        where
                        {
                            ?s <"""+target_rel_uri+"""> ?label. \n"""
    query += ("" if stype is None else (" ?s a <" + stype + "> ." if validators.url(stype) else "?s a ?stype. \n filter(?stype=" + "\"" + stype + "\") . \n"))
    query += ("" if otype is None else ("?label a  <" + otype + "> ." if validators.url(otype) else "?label a ?ltype. \n filter(?ltype=" + "\"" + otype + "\") . \n"))
    query += ("" if tragetNode_filter_statments is None else tragetNode_filter_statments )
    query+="""} } }  \n"""

    query+="""  limit ?limit 
                offset ?offset
            } 
            """
    query_spo,query_o_types=get_d1h1_query(graph_uri, target_rel_uri, stype=stype, otype=otype,tragetNode_filter_statments=tragetNode_filter_statments)
    return [query_spo,query_o_types,query]
def get_d1h2_query(graph_uri,target_rel_uri):
    query="""select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
            from <"""+graph_uri+""">
            where
            {
                 select ?o2 as ?s  ?p3 as ?p  ?o3 as ?o
                 where
                 {
                  ?s <"""+target_rel_uri+"""> ?label.
                  ?s ?p2 ?o2.
                  ?o2 ?p3 ?o3.
                  filter(!isBlank(?o2)).
                  filter(!isBlank(?o3)).
                 }
                limit ?limit 
                offset ?offset
            }  
            """
    return [get_d1h1_query(graph_uri,target_rel_uri),query]
def get_d2h2_query(graph_uri,target_rel_uri,tragetNode_filter_statments=None):
    query1="""select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
            from <"""+graph_uri+""">
            where
            {     
                 select ?o2 as ?s  ?p3 as ?p  ?o3 as ?o
                 where
                 {
                  ?s <"""+target_rel_uri+"""> ?label.
                  ?s ?p2 ?o2.
                  ?o2 ?p3 ?o3.
                  filter(!isBlank(?o2)).
                  filter(!isBlank(?o3)).\n"""
    if tragetNode_filter_statments:
        query1+=tragetNode_filter_statments
    query1+="""\n}
                limit ?limit 
                offset ?offset
            }
            """
    query2 = """select distinct (?s as ?subject) (?p as ?predicate) (?o as ?object)
                from <"""+graph_uri+""">
                where
                {     
                    select ?o3 as ?s  ?p3 as ?p  ?o2 as ?o
                     where
                     {
                      ?s <""" + target_rel_uri + """> ?label.
                      ?o2 ?p ?s.
                      ?o3 ?p3 ?o2.
                      filter(!isBlank(?o2)).
                      filter(!isBlank(?o3)).
                     }
                    limit ?limit 
                    offset ?offset
                }
                """
    return  [get_d1h1_query(graph_uri,target_rel_uri),get_d2h1_query(graph_uri,target_rel_uri),query1,query2]
def execute_sparql_multithreads(start_offset,sparql_endpoint_url,batch_size,queries,out_file,threads_count,RDFEngine=RDFEngine.OpenlinkVirtuoso):
    q_start_t = datetime.datetime.now()
    rows_count_lst=[]
    for query  in queries:
        rows_count_query = query.replace("select distinct ?s as ?subject ?p as ?predicate ?o as ?object", "select count(*) as ?c")
        rows_count_query=rows_count_query.replace("limit ?limit" ,"")
        rows_count_query = rows_count_query.replace("offset ?offset", "")
        rows_count=ExecQuery(sparql_endpoint_url,rows_count_query)
        rows_count_lst.append(int(rows_count))
    q_end_t = datetime.datetime.now()
    print("triples_count=", sum(rows_count_lst), "Query Time=",q_end_t - q_start_t, " sec.")
    #######################
    q_start_t = datetime.datetime.now()
    with open(out_file, 'w') as f:
        for q_idx,query  in enumerate(queries):
            batches_count = int(rows_count_lst[q_idx] / batch_size) + 1
            print("query idx=",q_idx,"batches_count=", batches_count)
            th_idx = 1
            th_lst = []
            for idx, offset in enumerate(range(start_offset, rows_count_lst[q_idx], batch_size)):
                try:
                    t = Thread(target=ExecQueryAndWriteOutput, args=(sparql_endpoint,query, offset, batch_size,batches_count, f,RDFEngine))
                    th_lst.append(t)
                    t.start()
                    th_idx = th_idx + 1
                    if th_idx == threads_count:
                        th_idx = 0
                        for th in th_lst:
                            th.join()
                        th_lst = []
                        print(threads_count, " threads finish at ", datetime.datetime.now() - q_start_t, " sec.")
                except  Exception as e:
                    print("Exception", e)
            for th in th_lst:
                th.join()
            print(threads_count, " threads finish at ", datetime.datetime.now() - q_start_t, " sec.")
    q_end_t = datetime.datetime.now()
    print("total time ", q_end_t - q_start_t, " sec.")
    return  q_end_t - q_start_t,sum(rows_count_lst)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_offset', dest='start_offset', type=int, help='Add start_offset', default=0)
    parser.add_argument('--sparql_endpoint', type=str, help='SPARQL endpoint URL', default='http://206.12.98.118:8890/sparql')
    parser.add_argument('--graph_uri', type=str, help=' KG URI', default='http://dblp.org')
    parser.add_argument('--target_rel_uri', type=str, help='target_rel_uri URI',default='https://dblp.org/rdf/schema#publishedIn')
    parser.add_argument('--TOSG', type=str, help='TOSG Pattern',default='d2h1')
    parser.add_argument('--batch_size', type=int, help='batch_size', default='1000000')
    parser.add_argument('--out_file', dest='out_file', type=str, help='output file to write trplies to', default='dblp_pv.tsv')
    parser.add_argument('--threads_count', dest='threads_count', type=int, help='output file to write trplies to', default=64)
    args = parser.parse_args()
    print('args=',args)
    start_offset = args.start_offset
    graph_uri=args.graph_uri
    sparql_endpoint =args.sparql_endpoint
    target_rel_uri=args.target_rel_uri
    TOSG=args.TOSG
    batch_size=args.batch_size
    out_file=args.out_file.split('.')[0]+"_"+TOSG+".tsv"
    threads_count = args.threads_count
    queries=[]
    if TOSG=='d1h1':
        queries=[get_d1h1_query(graph_uri,target_rel_uri)]
    elif TOSG=='d1h2':
        queries=get_d1h2_query(graph_uri,target_rel_uri)
    elif TOSG=='d2h1':
        queries=get_d2h1_query(graph_uri,target_rel_uri)
    elif TOSG=='d2h2':
        queries=get_d2h2_query(graph_uri,target_rel_uri)
    execute_sparql_multithreads(start_offset, sparql_endpoint, queries, batch_size,out_file, threads_count)
