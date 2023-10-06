import io
import os
from urllib.parse import urlparse
import pandas as pd
import stardog
from pathlib import Path
import wget as wget
from  UDF_Manager_Virtuoso import  UDFManager
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from io import StringIO
class StardogManager(UDFManager):
    def __init__(self,host='localhost',port=5820,username='admin', password='admin',database='dblp2022'):
        super(StardogManager, self).__init__(host,port,username, password)
        self.conn_details = {
            'endpoint': "http://"+host+":"+str(port),
            'username': username,
            'password': password
        }
        self.database=database
        self.StardogConn=stardog.Connection('pythondb', **self.conn_details)

    def createNewDB(self, db_name):
        db = self.StardogConn.new_database('pythondb')
        print('Created db')
        return db

    def dropDB(self, db):
        db.drop()
        print('Dropped db')
    def uploadTTLFile(self,dbname,TTLFilePath):
        self.StardogConn = stardog.Connection(dbname, **self.conn_details)
        # begin transaction
        self.StardogConn.begin()
        # # add data to transaction from file
        # path = str(Path(__file__).parent.resolve() / 'resources/GettingStarted_Music_Data.ttl')
        self.StardogConn.add(stardog.content.File(TTLFilePath))
        # commit the changes
        self.StardogConn.commit()
    def executeQuery(self,dbname,query,format='text/tsv'):
        self.StardogConn = stardog.Connection(dbname, **self.conn_details)
        self.StardogConn.begin()
        res=self.StardogConn.select(query,content_type=format, timeout=1200000)
        # res = pd.DataFrame([x.split('\\t') for x in str(res).split('\\n')])
        res = pd.read_csv(io.BytesIO(res),sep="\t")
        return res
    def executeDeleteQuery(self,dbname,query,):
        self.StardogConn = stardog.Connection(dbname, **self.conn_details)
        res=self.StardogCon.update(query)
        self.StardogCon.commit()
        return res



if __name__ == '__main__':
    stardog_manger=StardogManager(host="206.12.100.35")
    res=stardog_manger.executeQuery("dblp2022","select * where {?s ?p ?o} limit 10")
    print(type(res))
    print(res)