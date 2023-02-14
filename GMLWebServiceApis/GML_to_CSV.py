# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:54:04 2023

@author: walee
"""
import sys
sys.path.append("..")
import sparqlEndpoints.sparqlEndpoint as se
import os

def sparqlToCSV(query,filename):
    s_endpoint = se.sparqlEndpoint()
    df = s_endpoint.executeSparqlQuery(query)
    df.to_csv(filename,index=False)

query = """ prefix dblp_schema:<https://dblp.org/rdf/schema#>
SELECT ?paper,?title,(sql:getKeyValue_v2(?paper,?venue_dic)) as ?venue
#,?venue_dic
from <http://dblp.org>
WHERE
{
 ?paper <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp_schema:Publication.
 ?paper dblp_schema:title ?title.
 ?paper <https://dblp.org/rdf/schema#publishedIn> ?o.
 {      select ?o from <http://dblp.org>
        where {?s <https://dblp.org/rdf/schema#publishedIn> ?o.}
        group by ?o        order by desc(count(*))        limit 50
 }
 {
        select (sql:getNodeClass_v2('http://127.0.0.1:64646/all','')) As ?venue_dic
        from <http://dblp.org> where {}
 }
}
limit 10000"""

filename = os.path.join('.','test_results.csv')
sparqlToCSV(query, filename)