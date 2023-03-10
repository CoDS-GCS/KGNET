import sys
import os
from pathlib import Path
import csv
import pandas
import pandas as pd
from sparqlEndpoints.openlinkVirtuosoEndpoint import openlinkVirtuosoEndpoint
from GMLQueryManager import gmlManager
from sparqlEndpoints.sparqlEndpoint import sparqlEndpoint
from embeddingServices.similarityMetrics import cosineSimilarity
import datetime

if __name__ == '__main__':
    parser.add_argument("--query", type=str, default="{}")
    args = parser.parse_args()
    parsed_query = gmlManager.parseNodeClassifierQuery(args.query)
    gmlOperator = parsed_query.getGMLOperator()
    operatorTriples = gmlManager.executeOprator(gmlOperator)
    sparqlQuery = parsed_query.getSparqlQuery()
    sparqlTriples = openlinkVirtuosoEndpoint.execute(sparqlQuery)
    triples = gmlManager.JoinResults(operatorTriples, sparqlTriples)
    return triples


