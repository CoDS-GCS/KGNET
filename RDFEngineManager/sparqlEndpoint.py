import pandas as pd
from threading import Thread
import threading
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
import datetime
import requests
lock = threading.Lock()
class sparqlEndpoint:
    """sparql Endpoint class to excute sparql query on a virtuoso RDF endpoint"""
    def __init__(self,endpointUrl="http://206.12.98.118:8890/sparql"):
        self.endpointUrl = endpointUrl
#    Returns SparqlQuery As Dataframe
    def executeSparqlWrapperQuery(self,Sparql_query):
        sparql = SPARQLWrapper(self.endpointUrl)
        sparql.setQuery(Sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # print(results)
        res_val=[]
        lst_columns=[]
        if len(results["results"]["bindings"])>0:
            lst_columns=results["results"]["bindings"][0].keys()
            # print(lst_columns)
            # df=pd.DataFrame(columns=lst_columns)
            # print("df=",df)
            for result in results["results"]["bindings"]:
                lst_values=[]
                for col in lst_columns:
                    lst_values.append(result[col]["value"])  
                res_val.append(lst_values)
        return pd.DataFrame(res_val, columns = lst_columns)

    #    Returns SparqlQuery As Dataframe
    def executeSparqlQuery_dopost(self,query,firstRowHeader=True):
        """
        Execute sparql query through dopost and return results in form of datafarme.
        :param query:the sparql query string.
        :type query: str
        :param firstRowHeader: wherther to assume frist line as the dataframe columns header.
        :type firstRowHeader: boolean

        """
        body = {'query': query}
        headers = {
            # 'Content-Type': 'application/sparql-update',
            'Content-Type': "application/x-www-form-urlencoded",
            'Accept-Encoding': 'gzip',
            'Accept': 'text/tab-separated-values; charset=UTF-8'}
        r = requests.post(self.endpointUrl, data=body, headers=headers)
        if firstRowHeader:
            return pd.DataFrame([x.split('\t') for x in r.text.split('\n')[1:] if x],columns=r.text.split('\n')[0].replace("\"","").split('\t'))
        else:
            return pd.DataFrame([x.split('\t') for x in r.text.split('\n')])
    def executeSparqlquery(self,query,firstRowHeader=True):
        return self.executeSparqlQuery_dopost(query,firstRowHeader)

    def ExecScalarQuery(self, query):
        body = {'query': query}
        headers = {
            # 'Content-Type': 'application/sparql-update',
            'Content-Type': "application/x-www-form-urlencoded",
            'Accept-Encoding': 'gzip',
            'Accept': 'text/tab-separated-values; charset=UTF-8'}
        r = requests.post(self.endpointUrl, data=body, headers=headers)
        return r.text.split('\n')[1]

    def ExecQueryAndWriteOutput(self, query, offset, batch_size,batches_count, f):
        start_t = datetime.datetime.now()
        r_query = query.replace("?offset", str(offset))
        r_query = r_query.replace("?limit", str(batch_size))
        body = {'query': r_query}
        headers = {'Content-Type': 'application/x-www-form-urlencoded',
                   'Accept-Encoding': 'gzip',
                   'Accept': 'text/tab-separated-values; charset=UTF-8'}
        r = requests.post(self.endpointUrl, data=body, headers=headers)
        lock.acquire()
        f.write(r.text.replace(""""subject"	"predicate"	"object"\n""", ""))
        end_t = datetime.datetime.now()
        print("Query idx: ", (offset / batch_size)+1, "/", batches_count, " records count=", len(r.text.split('\n'))-2,
              " time=", end_t - start_t, " sec.")
        lock.release()
    def execute_sparql_multithreads(self,queries, out_file,start_offset=0, batch_size=10**6, threads_count=16,rows_count=None):
        q_start_t = datetime.datetime.now()
        rows_count_lst = []
        for query in queries:
            if rows_count==None:
                rows_count_query = query.replace("select distinct ?s as ?subject ?p as ?predicate ?o as ?object",
                                                "select count(*) as ?c")
                rows_count_query = rows_count_query.replace("limit ?limit", "")
                rows_count_query = rows_count_query.replace("offset ?offset", "")
                rows_count = self.ExecScalarQuery( rows_count_query)
            rows_count_lst.append(int(rows_count))
        q_end_t = datetime.datetime.now()
        print("rows_count=", sum(rows_count_lst), "Query Time=", q_end_t - q_start_t, " sec.")
        #######################
        q_start_t = datetime.datetime.now()
        with open(out_file, 'w') as f:
            for q_idx, query in enumerate(queries):
                batches_count = int(rows_count_lst[q_idx] / batch_size) + 1
                batch_size = rows_count_lst[q_idx] if rows_count_lst[q_idx] < batch_size else batch_size
                print("query idx=", q_idx, "batches_count=", batches_count)
                th_idx = 1
                th_lst = []
                for idx, offset in enumerate(range(start_offset, rows_count_lst[q_idx], batch_size)):
                    try:
                        t = Thread(target=self.ExecQueryAndWriteOutput, args=(query,offset,batch_size,batches_count,f))
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
        return q_end_t - q_start_t, sum(rows_count_lst)
if __name__ == '__main__':
    ""
    # e=sparqlEndpoint(endpointUrl="http://206.12.98.118:8890/sparql")
    # res_df=e.executeSparqlQuery_dopost("select ?s ?p ?o from <https://linkedmdb.org> where {?s ?p ?o} limit 10")
    # print(res_df)