import pandas as pd
import numpy as np
import re
class sparqlParser:
    def __init__(self):
        self.query=""  
        self.parseTree=None
        
class queryRewriter:
    def __init__(self):
        self.parseTree=None
        self.df_cognitive_queries= pd.read_csv("/GMLQueryRewriter/cognitive_queries.csv")
    def getOptimizedQuery(self,usecase,replace_dict):
        query = (self.df_cognitive_queries[self.df_cognitive_queries["usecase"]==usecase]["query"].values[0])
        print("Source Query=\n",query)
        # print("query=", query)
        res = re.split('select\s|\(sql:getEntitySimilarityScore.*?Score|where|order by |limit\s', query)
        res = list(filter(None, [x.strip() for x in res if x]))
        # print(res)
        graph_prefixs = res[0]
        graph_prefixs = re.split('prefix\s|.*\s*:<|>', graph_prefixs)
        graph_prefixs = list(filter(None, [x.strip() for x in graph_prefixs if x]))
        # print("graph_prefixs=", graph_prefixs)
        graph_url = re.split('.*:\s<', graph_prefixs[0])
        graph_url = list(filter(None, [x.strip() for x in graph_url if x]))
        # print("graph_url=",graph_url)
        select_att = res[1]
        UDF = re.findall("\(sql:getSimilarEntityScore.*?Score", query)[0].strip()
        UDF_params = re.split('.*\(|,|\).*', UDF)
        UDF_params = list(filter(None, [x.strip() for x in UDF_params if x]))
        # print("UDF_params=", UDF_params)
        select_BGP = res[2].strip()
        select_order_by = res[3]
        select_limit = res[4]

        # print("select_att=",select_att)
        # print("UDF=", UDF)
        # print("select_BGP=", select_BGP)
        # print("select_order_by=", select_order_by)
        # print("select_limit=", select_limit)

        NewQuery = "prefix ";
        # print(int(len(graph_prefixs)))
        for elem in graph_prefixs:
            NewQuery += elem + ">\n"

        NewQuery += "select " + select_att + " xsd:Float(bif:get_keyword(bif:lower(" + UDF_params[
            1] + "),?selected_entities,0)) as ?Score \n"
        NewQuery += " where " + select_BGP.replace("}", "")
        NewQuery += """filter(sql:ContainsKey(?selected_entities, bif:lower(""" + UDF_params[1] + """)))
                        {select   sql:getTopSimilarEntities('""" + replace_dict[UDF_params[1]] + """',""" + replace_dict[select_limit] + """,'""" + \
                    replace_dict[UDF_params[2]] + """','""" + graph_url[0] + """') as ?selected_entities     where   {  }
                    }\n}\n"""
        NewQuery += " order by " + select_order_by + "\n"
        NewQuery += " limit " +  replace_dict[select_limit]
        # print("NewQuery=", NewQuery)

        # oq=str(self.df_cognitive_queries[self.df_cognitive_queries["usecase"]==usecase]["optimizedQuery"].values[0])
        # for key in replace_dict:
        #     oq=oq.replace(key,replace_dict[key])
        # return oq
        return NewQuery
    def getOptimizedQueryFaiss(self,usecase,replace_dict):
        # oq=str(self.df_cognitive_queries[self.df_cognitive_queries["usecase"]==usecase]["optimizedQueryFaiss"].values[0])
        # for key in replace_dict:
        #     oq=oq.replace(key,replace_dict[key])
        # return oq
        query = (self.df_cognitive_queries[self.df_cognitive_queries["usecase"] == usecase]["query"].values[0])
        # print("query=", query)
        res = re.split('select\s|\(sql:getSimilarEntityScore.*?Score|where|order by |limit\s', query)
        res = list(filter(None, [x.strip() for x in res if x]))
        # print(res)
        graph_prefixs = res[0]
        graph_prefixs = re.split('prefix\s|.*\s*:<|>', graph_prefixs)
        graph_prefixs = list(filter(None, [x.strip() for x in graph_prefixs if x]))
        # print("graph_prefixs=", graph_prefixs)
        graph_url = re.split('.*:\s<', graph_prefixs[0])
        graph_url = list(filter(None, [x.strip() for x in graph_url if x]))
        # print("graph_url=",graph_url)
        select_att = res[1]
        UDF = re.findall("\(sql:getSimilarEntityScore.*?Score", query)[0].strip()
        UDF_params = re.split('.*\(|,|\).*', UDF)
        UDF_params = list(filter(None, [x.strip() for x in UDF_params if x]))
        # print("UDF_params=", UDF_params)
        select_BGP = res[2].strip()
        select_order_by = res[3]
        select_limit = res[4]

        # print("select_att=",select_att)
        # print("UDF=", UDF)
        # print("select_BGP=", select_BGP)
        # print("select_order_by=", select_order_by)
        # print("select_limit=", select_limit)

        NewQuery = "prefix ";
        # print(int(len(graph_prefixs)))
        for elem in graph_prefixs:
            NewQuery += elem + ">\n"

        NewQuery += "select " + select_att + " xsd:Float(bif:get_keyword(bif:lower(" + UDF_params[
            1] + "),?selected_entities,0)) as ?Score \n"
        NewQuery += " where " + select_BGP.replace("}", "")
        NewQuery += """filter(sql: ContainsKey(?selected_entities, bif:lower(""" + UDF_params[1] + """)))
                               {select   sql:getTopSimilarEntitiesFaiss('""" + replace_dict[UDF_params[1]] + """',""" + replace_dict[select_limit] + """,'""" + \
                    replace_dict[UDF_params[2]] + """','""" + graph_url[0] + """') as ?selected_entities     where   {  }
                           }\n}\n"""
        NewQuery += " order by " + select_order_by + "\n"
        NewQuery += " limit " +  replace_dict[select_limit]
        # print("NewQuery=", NewQuery)

        # oq=str(self.df_cognitive_queries[self.df_cognitive_queries["usecase"]==usecase]["optimizedQuery"].values[0])
        # for key in replace_dict:
        #     oq=oq.replace(key,replace_dict[key])
        # return oq
        return NewQuery