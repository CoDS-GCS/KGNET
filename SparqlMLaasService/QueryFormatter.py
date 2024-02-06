import json
import rdflib
import re
from rdflib.plugins.sparql.parser import parseQuery as rdflibParseQuery
from rdflib.plugins.sparql.parserutils import CompValue
from pyparsing.results import ParseResults
import Constants
class gmlQueryFormatter:
    pre_keywords_lst = ['ParseResults', 'PrefixDecl_', 'SelectQuery_', '\'projection\'', 'vars_',
                    '\'datasetClause\'', '\'where\'', 'Filter_', 'OptionalGraphPattern_','GroupOrUnionGraphPattern_''GroupGraphPatternSub_', 'SubSelect_']
    post_keywords_lst = [ "{}),"]
    # '[', ']'
    @staticmethod
    def format_gml_query_tree(gml_statments_dict,str_result="",indent=""):
        subselect_lst=[]
        for key in gml_statments_dict.keys():
            if str(key) in ['query_type','from','limit','offset'] and gml_statments_dict[key] is not None:
                str_result+=indent+str(key)+": "+gml_statments_dict[key]+"\n"
            elif str(key) == 'prefixes:':
                str_result += str(key) + " \n"
                for prefix in gml_statments_dict[key]:
                    str_result+=indent+"\t|"+prefix+":"+gml_statments_dict[key][prefix]+"\n"
            elif str(key) == 'select':
                if 'modifier' in gml_statments_dict.keys() and gml_statments_dict['modifier'] is not None:
                    str_result += indent + "modifier:" + gml_statments_dict['modifier']+"\n"
                str_result += indent+"projection_variables:" + "\n"
                for var in gml_statments_dict[key]:
                    str_result+=indent+"\t|Type:"+var['type']+"\tName:"+var['name']
                    if 'args' in var.keys():
                        str_result +="\t|args:"+str(var['args'])
                    str_result +="\n"
            elif str(key) == 'triples':
                str_result += indent+"triples:" + "\n"
                for t in gml_statments_dict[key]:
                    str_result += indent+"\t|"+str(t)+"\n"
            # elif str(key) == 'optional_triples' and len(gml_statments_dict[key])>0:
            #     for t in gml_statments_dict[key]:
            #         str_result += indent+"\t|" + str(t) + "\n"
            # elif str(key) == 'Filter' and len(gml_statments_dict[key])>0:
            #     str_result += indent+"filters:" + "\n"
            #     for t in gml_statments_dict[key]:
            #         str_result += indent+"\t|" + str(t) + "\n"
            elif str(key) == 'SubSelect':
                subselect_lst.append(gml_statments_dict[key])
        for subselect in subselect_lst:
            str_result += indent + "SubSelect query:" + "\n"
            str_result=gmlQueryFormatter.format_gml_query_tree(subselect,str_result,indent+"\t|")
        return str_result

    @staticmethod
    def format_gml_query_str(gml_query):
        for keyword in gmlQueryFormatter.pre_keywords_lst:
            gml_query=gml_query.replace(keyword,"\n"+keyword)
        for keyword in gmlQueryFormatter.post_keywords_lst:
            gml_query = gml_query.replace(keyword, keyword+"\n")
        gml_query = gml_query.replace("\n \n", "\n")
        return gml_query

    @staticmethod
    def highlight_gml_query_str(self, gml_query):
        return gmlQueryFormatter.format_gml_query(gml_query)
