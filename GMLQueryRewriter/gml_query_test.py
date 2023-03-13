from rdflib import Graph
import rdflib
import gzip
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql import parser

# g = Graph()
# fpath = r'D:/CoDs Lab/kgnet/KGNET_MetaGraph.ttl'
# print(rdflib.util.guess_format(fpath, fmap=None))
# g.parse(fpath,format=rdflib.util.guess_format(fpath, fmap=None))

query_str = """ 
PREFIX kgnet: <http://www.kgnet.ai/>
PREFIX dblp:  <https://dblp.org/rdf/schema/>

SELECT ?apiUrl
WHERE {
  ?nodeClassifier a <kgnet:types/NodeClassifier> ;
                  <kgnet:GML/NodeLabel> <dblp:venue> ;
                  <kgnet:GML/TargetNode> <dblp:publication> ;
  <kgnet:term/uses> ?gmlModel .
  ?gmlModel <kgnet:API_URL> ?apiUrl .

}

"""

query_v2 = """
PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX kgnet: <https://www.kgnet.ai/>

    #SELECT ?NodeClassifier
    SELECT ?apiUrl
    WHERE
    {
        ?NodeClassifier <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <kgnet:types/NodeClassifier> .
        ?NodeClassifier <kgnet:GML/TargetNode> <dblp:Publication> .
        ?NodeClassifier <kgnet:GML/NodeLabel> <dblp:venue> .
        ?NodeClassifier
        <kgnet:term/uses> ?gmlModel .
        ?gmlModel <kgnet:API_URL> ?apiUrl .
        }
"""

def kgnet_getModelURI(query,PATH_TO_KG = 'D:/CoDs Lab/kgnet/KGNET_MetaGraph.ttl'):
    g = Graph()
    g.parse(PATH_TO_KG,format=rdflib.util.guess_format(PATH_TO_KG, fmap=None))
    results = g.query(query)
    for row in results:
        return (row[0])
# Parse the query string into a Query object

# Run the query and iterate over the results
# results = g.query(query_v2)
# for row in results:
#     # api_url = row['api_url']
#     print(row[0])
