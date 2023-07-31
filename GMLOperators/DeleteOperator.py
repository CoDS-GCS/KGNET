import sys
import os
from pathlib import Path
import csv
import pandas
import pandas as pd
from RDFEngineManager.UDF_Manager_Virtuoso import openlinkVirtuosoEndpoint
from GMLQueryManager import gmlManager
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from KGEs.similarityMetrics import cosineSimilarity
import datetime

if __name__ == '__main__':
    parser.add_argument("--query", type=str, default="{}")
    args = parser.parse_args()
    parsed_query = gmlManager.parseInsertQuery(args.query)
    deleteModel = parsed_query.getDeleteModel()
    triples = gmlManager.delete(deleteModel)
    gmlManager.DeleteTriples(triples)
    return True


