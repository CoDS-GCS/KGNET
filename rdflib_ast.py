from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

# Example SPARQL query
sparql_query = """
    SELECT ?subject ?predicate ?object
    WHERE {
        ?subject a ?o.
         ?o ?predicate ?object .
    }
"""

# Parse the SPARQL query
graph = Graph()
query = prepareQuery(sparql_query)

# Obtain the AST
ast = query.algebra
print(ast)