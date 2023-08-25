import os
from urllib.parse import urlparse
import pandas as pd
import pyodbc
import wget as wget

from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
class UDFManager(sparqlEndpoint):
    def __init__(self,host='localhost',port=1111,username='dba', password='dba'):
        sparqlEndpoint.__init__(self,endpointUrl="http://"+host+":8890/sparql")
class VirtuosoUDFManager(UDFManager):
    def __init__(self,host='localhost',port=1111,username='dba', password='dba'):
        super(VirtuosoUDFManager, self).__init__(host,port,username, password)
        self.version = "vos 7.5.2"
        self.VirtuosoConn = "DRIVER=/mnt/virtuoso_endpoints/virtuoso-opensource_qald9/lib/virtodbc.so;HOST=" + host + ":" + str(
            port) + ";UID=" + username + ";PWD=" + password

    def executeInteractiveSQL(self,SQL):
        conn = pyodbc.connect(self.VirtuosoConn)
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
#         SQL=""" SELECT P_NAME,P_TEXT  FROM SYS_PROCEDURES WHERE P_NAME like '"""+P_Name+"""%' """
        df=[]
        try:
            cursor = conn.cursor()
            res=cursor.execute(SQL)
            col_count=len([column[0] for column in cursor.description])    
            lst_columns = [column[0] for column in cursor.description]
            df = pd.DataFrame(columns=lst_columns)
        #     print(columns)
            for row in cursor:
        #         print(type(row))
                row_to_list = [elem for elem in row]
        #         df=df.append(row_to_list)
                df_length = len(df)
                df.loc[df_length] = row_to_list
        #                 for i in range(0,col_count):
        #                     print(columns[i]+"="+row[i])
        #             print(results)    
            cursor.close()
            conn.commit()
        except Exception as e: 
            print(e)
        finally:
            conn.close()   
        return df
    def getPyodbcConnection(self):
        conn = pyodbc.connect(self.VirtuosoConn)
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setencoding(encoding='utf-8')
        return conn
    def getProcedureScript(self,P_Name):
        conn = self.getPyodbcConnection()
        SQL=""" SELECT P_NAME,P_TEXT  FROM SYS_PROCEDURES WHERE P_NAME = 'DB.DBA."""+P_Name+"""' """
        df=[]
        try:
            cursor = conn.cursor()
            res=cursor.execute(SQL)
            col_count=len([column[0] for column in cursor.description])    
            lst_columns = [column[0] for column in cursor.description]
            df = pd.DataFrame(columns=lst_columns)
        #     print(columns)
            for row in cursor:
        #         print(type(row))
                row_to_list = [elem for elem in row]
        #         df=df.append(row_to_list)
                df_length = len(df)
                df.loc[df_length] = row_to_list
        #                 for i in range(0,col_count):
        #                     print(columns[i]+"="+row[i])
        #             print(results)    
            cursor.close()
            conn.commit()
        except Exception as e: 
            print(e)
        finally:
            conn.close()   
        return df
    def searchProcedure(self,P_Name):
        conn = self.getPyodbcConnection()
        SQL=""" SELECT P_NAME  FROM SYS_PROCEDURES WHERE P_NAME like '%"""+P_Name+"""%' """
        df=[]
        try:
            cursor = conn.cursor()
            res=cursor.execute(SQL)
            col_count=len([column[0] for column in cursor.description])    
            lst_columns = [column[0] for column in cursor.description]
            df = pd.DataFrame(columns=lst_columns)
        #     print(columns)
            for row in cursor:
        #         print(type(row))
                row_to_list = [elem for elem in row]
        #         df=df.append(row_to_list)
                df_length = len(df)
#                 print( row_to_list)
                df.loc[df_length] = row_to_list
        #                 for i in range(0,col_count):
        #                     print(columns[i]+"="+row[i])
        #             print(results)    
            cursor.close()
            conn.commit()
        except Exception as e: 
            print(e)
        finally:
            conn.close()   
        return df
    def createProcedure(self,SQL,Parameters,Description):
        print('SQL=',SQL)
        # http://docs.openlinksw.com/virtuoso/execpythonscript/
        conn = self.getPyodbcConnection()
        result="OK"
        try:
            cursor = conn.cursor()
            cursor.execute(SQL)
            cursor.close()
            conn.commit()
        except Exception as e: 
            result=str(e)
        finally:
            conn.close()
        if result =="OK":
            udf_name = SQL.split("(")[0].split("DB.DBA.")[1]
            self.addCatalogueUDF(udf_name,Parameters,Description)
            self.setVirtuosoProcedureExecuteGrant(udf_name,"dba")
            self.setVirtuosoProcedureExecuteGrant(udf_name,"SPARQL")
        return result
    def setProcedureExecuteGrant(self, procedureName,UserName):
        conn = pyodbc.connect(self.VirtuosoConn)
        SQL="""grant execute on DB.DBA."""+procedureName+""" to \""""+UserName+"""\" """
        print("SQL=",SQL)
        result="OK"
        try:
            cursor = conn.cursor()
            cursor.execute(SQL)
            cursor.close()
            conn.commit()
        except Exception as e: 
            result=str(e)
        finally:
            conn.close()
        return result
    def uploadKG_ttl(self,ttlFileUrl,namedGraphIRI):
#         url = 'https://raw.githubusercontent.com/frmichel/taxref-ld/13.0/dataset/Taxrefld_static_dcat.ttl'
        a = urlparse(ttlFileUrl)
        # print(a.path)                   
        file_name=os.path.basename(a.path).split('.ttl')[0]+'.ttl'
        file_full_path=os.path.abspath(""+file_name)
        d_res=wget.download(ttlFileUrl,out=file_full_path)        
        print('\nfile_full_path=',file_full_path,' d_res=',d_res)
        conn = pyodbc.connect(self.VirtuosoConn)
#         SQL="""select SPARQL_DAWG_LOAD_REMOTE_DATFILE('"""+ttlFileUrl+"""')"""
        SQL="""DB.DBA.TTLP_MT (file_to_string_output ('"""+file_full_path+"""'), '', '"""+namedGraphIRI+"')"""
        print("SQL="+SQL)
        result="OK"
        try:
            cursor = conn.cursor()
            cursor.execute(SQL)
            cursor.close()
            conn.commit()
        except Exception as e: 
            result=str(e)
        finally:
            conn.close()
#         os.remove(file_full_path)

        df,q= self.getVirtuosoGraphsList()
    #         df.head(10)
        return len(df[df["g"].str.contains(file_name)]),result
    def getKGList(self):
        Query="""
            SELECT  DISTINCT ?g 
            WHERE  { GRAPH ?g {?s ?p ?o} } 
            ORDER BY  ?g
            limit 100
        """
        return self.executeSparqlQuery_dopost(Query),Query
if __name__ == '__main__':
    udfm=VirtuosoUDFManager(host="206.12.98.118",port=1111,username='***', password='***')
    print(udfm.getKGList())