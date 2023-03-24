# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:57:00 2023

@author: walee
"""
from flask import Flask, request,render_template,redirect,url_for
import GML_to_CSV as gcsv
import GMLQueryRewriter.gmlRewriter as qr
import time
import os
# import sparqlEndpoints.sparqlEndpoint as se

test_query = """ 
prefix dblp:<https://dblp.org/rdf/schema#>
prefix kgnet: <https://www.kgnet.ai/>
select ?title ?venue 
where {
?paper a dblp:Publication.
?paper dblp:title ?title.
?paper <https://dblp.org/rdf/schema#publishedIn> ?o.


?paper ?NodeClassifier ?venue.

?NodeClassifier a <kgnet:types/NodeClassifier>.
?NodeClassifier <kgnet:GML/TargetNode> <dblp:paper>.
?NodeClassifier <kgnet:GML/NodeLabel> <dblp:venue>.}
limit 10
  """
# print(qr.execute(test_query))

# data_query = qr.execute(test_query)[0]
# gcsv.sparqlToCSV(data_query, 'temp/temp.csv')
# TEMP_FILE = 'flask_temp.csv'

def get_query_results(query,filename):
    
    data_query = qr.execute(query)[0]
    gcsv.sparqlToCSV(data_query, filename)
    try:
        gcsv.mapVenues(filename)
    except Exception as e:
        print('*'*20,'ERROR','*'*20,'\n',e)
        return


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',)

@app.route('/post_query',methods=['POST'])
def post_query():
    query = request.form['query']
    filename = os.path.join('temp',time.strftime("%Y%m%d-%H%M%S"+'.csv'))
    get_query_results(query,filename)
    while not os.path.exists(filename):
        time.sleep(0.5)
    return redirect(url_for('display_results',filename=filename))
    # return 'Query Executed Successfully !'+query

@app.route('/display_results',methods=['GET'])
def display_results():
    filename = request.args.get('filename','')
    html_table = gcsv.csvToHTML(filename)
    return render_template('table.html', html_table = html_table)


if __name__=="__main__":
    app.run()
