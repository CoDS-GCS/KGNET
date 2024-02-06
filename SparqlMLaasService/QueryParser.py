import json
import copy
import rdflib
import re
from rdflib.plugins.sparql.parser import parseQuery as rdflibParseQuery
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
import numpy as np
# from QueryFormatter import  gmlQueryFormatter
from rdflib.plugins.sparql.parserutils import CompValue
from pyparsing.results import ParseResults
import Constants
import requests
from datetime import datetime
import json
def set_syntax (var):
    """ The set_syntax () function takes redflib variable from the query and adjusts its syntax according
            to its position in the SPARQL query. This function is used in rebuilding the query according to the
            SPARQL syntax rules."""
    if isinstance (var,str) and ('http' in var or ':' in var): #IRI
        return f'<{var}>'
    elif isinstance(var,rdflib.plugins.sparql.parserutils.CompValue): ## Literal Value
        if list(var.keys())[0]=='string':
            return "\""+str(var.string)+"\""
        else:
            return str(var.string)
    elif isinstance(var,rdflib.term.Literal) : # Number
        return str(var)
    elif isinstance(var, rdflib.term.Variable): # Variable
        return '?' + var  #'?' append to it if its a single term else return with prefix:postfix notation
    elif isinstance(var,tuple): # Tuple
        return  f'{str(var[0])}:{str(var[1])}'
    else:
        return str(var) # String
    return var
class Node:
    def __init__(self,var):
       self.var=var
       self.parent=None
       self.child = None
       self.is_gml=False
       self.gml_parent = None
    def add_parent(self,par):
        if self.parent is None:
            self.parent = []
        self.parent.append(par)
    def add_children(self, ch):
        if self.child is None:
            self.child = []
        self.child.append(ch)
    def setIsGML(self,is_gml):
        self.is_gml=is_gml
    def  addGmlParent(self,node):
        if self.gml_parent is None:
            self.gml_parent = []
        self.gml_parent.append(node)

def dopost(url, json_body):
    try:
        response = requests.post(url, json=json_body)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        print('dopost request Error', e)
def getGMLOperatorTaskId(gml_triples):
    """ The function takes dictionary of parsed GML operators and returns a dictionay of KGMeta task-id query per operator """
    GML_taskId_Query_dict= {}
    target_variable=None
    predictetd_variable=None
    string_Q = f"""
               SELECT ?tid
               from <""" + Constants.KGNET_Config.KGMeta_IRI + """>
               WHERE {
               """
    for triple in gml_triples:
       s,p,o,bgpType=triple.values()
       if str(p) in ['kgnet:targetNode']:
            obj=set_syntax(o)
            obj= ("<"+str(obj)+">") if (str(obj).startswith("<") or str(obj).startswith("http") or ":" in str(obj) ) else str(obj)
            string_temp = f"{set_syntax(s)}  <kgnet:GMLTask/targetNode> {obj} .\n"
            string_Q += string_temp
       if str(p) in [ 'kgnet:labelNode']:
           obj = set_syntax(o)
           obj = ("<" + str(obj) + ">") if (str(obj).startswith("<") or str(obj).startswith("http") or ":" in str(obj)) else str(obj)
           string_temp = f"{set_syntax(s)}  <kgnet:GMLTask/labelNode> {obj} .\n"
           string_Q += string_temp
       elif str(p) == 'kgnet:targetEdge':
            string_temp = f"{set_syntax(s)} <kgnet:GMLTask/targetEdge> {set_syntax(o)} .\n"
            string_Q += string_temp
       elif isinstance(p, list):
            for list_triple in p:
                if str(p).lower() not in ['kgnet:topk','kgnet:gml/gnnmethod']:
                    string_temp = f"{set_syntax(list_triple['subject'])} {set_syntax(list_triple['predicate'])} {set_syntax(list_triple['object'])} .\n"
                    string_Q += string_temp
       elif str(p)=='http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
            string_temp = f"{set_syntax(s)} {set_syntax(p)} <kgnet:type/GMLTask> .\n"
            string_Q += string_temp
            string_temp = f"{set_syntax(s)} <kgnet:GMLTask/taskType> <kgnet:type/nodeClassification> .\n"
            string_Q += string_temp
            target_variable=s
    string_Q += f"{set_syntax(target_variable)} <kgnet:GMLTask/id> ?tid .\n"
    # string_Q += """?gmlModel <kgnet:GMLModel/id> ?mID .  """
    string_Q += "}"
    ################## get predicted Variable #################
    for triple in gml_triples:
           s, p, o, bgpType = triple.values()
           if str(p) == str(target_variable):
               predictetd_variable=o
               break

    return string_Q,target_variable,predictetd_variable
class gmlQueryParser:
    """
    This module will take a basic SPARQL query and return two queries:
        1. SPARQL GML Query
        2. SPARQL Data Query

    The flow of of this module is as following:
        -> extract ()
        -> gen_queries ()
        -> split_data_gml_triples ()
        -> prep_gml_vars ()
        -> construct_gml_q ()
        -> construct_data_q ()
    """

    def __init__(self,gmlquery):
       self.gmlquery=gmlquery
       self.query_statments ={}
    def build_DAG(self):
        KGNET_LOOKUP = ['kgnet', '<kgnet']
        variables_DAG={}
        if 'triples' in self.query_statments.keys():
            for triple in self.query_statments['triples']:
                if triple['bgpType'] in ['Optional','NonOptional']:
                    s,p,o,bgpType=triple.values()
                    if isinstance(s,rdflib.term.Variable):
                        if str(s) not in variables_DAG.keys():
                            variables_DAG[str(s)]=Node(s)
                    if isinstance(p,rdflib.term.Variable):
                        if str(p) not in variables_DAG.keys():
                            pnode=Node(p)
                            pnode.add_parent(variables_DAG[str(s)])
                            variables_DAG[str(s)].add_children(pnode)
                            variables_DAG[str(p)]=pnode
                    if p=="http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and isinstance(o,rdflib.term.URIRef) and str(o).split(":")[0] in KGNET_LOOKUP:
                        variables_DAG[str(s)].setIsGML(True)
                    if isinstance(o,rdflib.term.Variable):
                        if str(o) not in variables_DAG.keys():
                            onode=Node(o)
                            if isinstance(p, rdflib.term.Variable):
                                onode.add_parent(variables_DAG[str(p)])
                                variables_DAG[str(p)].add_children(onode)
                            else:
                                onode.add_parent(variables_DAG[str(s)])
                                variables_DAG[str(s)].add_children(onode)
                            variables_DAG[str(o)]=onode
        ############ Projections ###################
        if 'select' in self.query_statments.keys():
            for var in self.query_statments['select']:
                if 'alias' in var.keys() and str(var['alias']) not in  list(variables_DAG.keys()):
                    if var['name'] in list(variables_DAG.keys()):
                        proj_node = Node(var['variable'])
                        proj_node.add_parent(variables_DAG[str(s)])
                        variables_DAG[variables_DAG[str(s)]].add_children(proj_node)
                        variables_DAG[var['name']] = proj_node
                elif var['name'] in list(variables_DAG.keys()):
                    continue

        variables_DAG['root']=[]
        for node in variables_DAG.keys():
            if node!='root' and variables_DAG[node].parent is None:
                variables_DAG['root'].append(variables_DAG[node])
        for var in variables_DAG.keys():
            if var!="root":
                var_parent=variables_DAG[var].parent
                while var_parent is not None and len(var_parent)>0:
                    if var_parent[0].is_gml == True:
                        variables_DAG[var].addGmlParent(var_parent[0])
                        break
                    var_parent=var_parent[0].parent

        return variables_DAG
    def draw_DAG(self,DAG):
        # install graphviz: https://stackoverflow.com/questions/15661384/python-does-not-see-pygraphviz
        # sudo apt-get install graphviz libgraphviz-dev pkg-config
        # Create and activate virtualenv if needed. The commands looks something like sudo apt-get install python-pip python-virtualenv
        # Run pip install pygraphviz

        node_color_map=[]
        edge_color_map = []
        g = nx.MultiDiGraph()
        for node in DAG.keys():
            if node!='root':
                if DAG[node].is_gml==True:
                    node_color_map.append('#A8D890')
                else:
                    node_color_map.append('#ffff8f')
                if DAG[node].parent is not None:
                    for elem in DAG[node].parent:
                        g.add_edge("?"+str(elem.var),str("?"+DAG[node].var),label='', color='#000000', weight=2)
                if DAG[node].gml_parent is not None:
                    for elem in DAG[node].gml_parent:
                        g.add_edge(str("?"+DAG[node].var),"?"+str(elem.var),label='gml', color='r', weight=0.5,)
        edges = g.edges()
        colors = []
        weights = []
        labels= {}
        for u, v in edges:
            for k in g[u][v].keys():
                colors.append(g[u][v][k]['color'])
                weights.append(g[u][v][k]['weight'])
                labels[(u,v)]=(g[u][v][k]['label'])
        write_dot(g, 'test.dot')
        plt.title('GML-Query-DAG')
        pos = graphviz_layout(g, prog='dot')
        nx.draw(g, pos, with_labels=True, arrows=True, node_color=node_color_map, edge_color=colors, width=weights,node_size=3000, font_size=10,connectionstyle='arc3, rad = 0.2')
        nx.draw_networkx_edge_labels(g, pos,edge_labels=labels,font_color='red')
        plt.tight_layout()
        plt.savefig('Query_DAG.png',dpi=250)
        # plt.show()
    def DecomposeSubqueries(self,DAG):
        gml_vars=[var for var in DAG.keys() if var!='root' and DAG[var].is_gml==True ]
        dependent_gml_vars = [var for var in DAG.keys() if var != 'root' and (DAG[var].is_gml == False and DAG[var].gml_parent is not None)]
        SQ1_Stmt={}
        SQ2_Stmt = {}
        subqueries={"SQ1":SQ1_Stmt,"SQ2":SQ2_Stmt}
        SQ1_Stmt["is_gml"]=False
        SQ2_Stmt["is_gml"] = True
        gml_vars_parents = set([item for var in DAG.keys() if var != 'root' and (DAG[var].is_gml == True and DAG[var].parent is not None)for item in DAG[var].parent])
        for key in subqueries:
            subqueries[key]['query_type']=self.query_statments['query_type']
            subqueries[key]['prefixes']={}
            for prefix in self.query_statments['prefixes']:
                if  prefix not in ['kgnet']:
                    subqueries[key]['prefixes'][prefix]=self.query_statments['prefixes'][prefix]
            #############################

            subqueries[key]['select']=[]
            if key in["SQ2"]:
                for var in self.query_statments['select']:
                    if var in dependent_gml_vars or var not in gml_vars:
                        subqueries[key]['select'].append(var['variable'])
            elif key in ["SQ1"]:
                for var in self.query_statments['select']:
                    if var not in dependent_gml_vars and var not in gml_vars :
                        subqueries[key]['select'].append(var['variable'])
            ##############################
            subqueries[key]['triples']=[]
        if 'triples' in self.query_statments.keys():
            for triple in self.query_statments['triples']:
                if triple['bgpType'] in ['Optional', 'NonOptional']:
                    s, p, o, bgpType = triple.values()
                    if str(s) in gml_vars or str(p) in gml_vars or str(o) in gml_vars or str(s) in dependent_gml_vars or str(p) in dependent_gml_vars or str(o) in dependent_gml_vars:
                        SQ2_Stmt['triples'].append(triple.copy())
                        # triple['subquery'] = "SQ2"
                    else:
                        SQ1_Stmt['triples'].append(triple.copy())
                        # triple['subquery']="SQ1"
                elif triple['bgpType'] in ['filter']:
                    var, op, val, bgpType = triple.values()
                    if str(var) in gml_vars or str(val) in gml_vars or str(var) in dependent_gml_vars or str(val) in dependent_gml_vars:
                        SQ2_Stmt['triples'].append(triple.copy())
                        # triple['subquery'] = "SQ2"
                    else:
                        SQ1_Stmt['triples'].append(triple.copy())
                        # triple['subquery'] = "SQ1"
        # SQ3_Stmt['limit']=self.query_statments['limit']
        # SQ3_Stmt['offset'] = self.query_statments['offset']
        ######################## Decompose SQ2 ##############################
        sq2_lst={}
        if 'triples' in SQ2_Stmt.keys():
            for gml_var in gml_vars:
                gml_var_dependents=[]
                for var in DAG:
                    if var!='root' and DAG[var].gml_parent is not None and gml_var in [str(elem.var) for elem in DAG[var].gml_parent]:
                        gml_var_dependents.append(var)
                sq2_lst[gml_var]={}
                sq2_lst[gml_var]['triples']=[]
                for triple in SQ2_Stmt['triples']:
                    if triple['bgpType'] in ['Optional', 'NonOptional']:
                        s, p, o, bgpType = triple.values()
                        if str(s) ==gml_var or str(p) ==gml_var or str(o) ==gml_var or str(s) in gml_var_dependents or str(p) in gml_var_dependents or str(o) in gml_var_dependents:
                            sq2_lst[gml_var]['triples'].append(triple.copy())
                    elif triple['bgpType'] in ['filter']:
                        var, op, val, bgpType = triple.values()
                        if str(var) ==gml_var or str(val) ==gml_var or str(var) in gml_var_dependents or str(val) in gml_var_dependents:
                            sq2_lst[gml_var]['triples'].append(triple.copy())
                sq2_lst[gml_var]['select']=[]
                for var in SQ2_Stmt['select']:
                    if str(var) in gml_var_dependents or str(var) ==gml_var:
                        sq2_lst[gml_var]['select'].append(var)
        for key in sq2_lst:
            sq2_lst[key]['query_type']=SQ2_Stmt['query_type']
            sq2_lst[key]['prefixes'] = SQ2_Stmt['prefixes']
            sq2_lst[key]['is_gml'] = SQ2_Stmt['is_gml']
        subqueries['SQ2']=list(sq2_lst.values())
        return subqueries
    def ReWriteSubqueryToSPARQL(self,subquery,DAG):
         SPARQL_Query=""
         if "prefixes" in subquery.keys():
             for prefix in subquery['prefixes']:
                 SPARQL_Query+=f"prefix {prefix}:<{subquery['prefixes'][prefix]}>\n"
         if "query_type" in subquery.keys() and subquery['query_type']=="SelectQuery":
             SPARQL_Query += f"Select distinct "
         if "select" in subquery.keys():
            for var in subquery['select']:
                SPARQL_Query += f" ?{str(var)} "
            SPARQL_Query+="\nwhere{\n"
         if 'triples' in subquery.keys():
            for triple in subquery['triples']:
                if triple['bgpType'] in ['Optional', 'NonOptional']:
                    s, p, o, bgpType = triple.values()
                    triple_str=f"{set_syntax(s)} {set_syntax(p)} {set_syntax(o)} ."
                    if triple['bgpType']=='Optional':
                        SPARQL_Query+="optional{"+triple_str+"}\n"
                    else:
                        SPARQL_Query += f"{triple_str}\n"
                elif triple['bgpType'] in ['filter']:
                    var, op, val, bgpType = triple.values()
                    SPARQL_Query+=f"filter({set_syntax(var)} {set_syntax(op)} {set_syntax(val)}).\n"
         SPARQL_Query+="}"
         if "limit" in subquery.keys() and subquery['limit'] is not None:
            SPARQL_Query += "limit "+subquery['limit'] +"\n"
         if "offset" in subquery.keys() and subquery['offset'] is not None:
            SPARQL_Query += "offset "+subquery['offset'] +"\n"
         return SPARQL_Query
    def breakdown_Sparql_Select_statment(self,where_part):
        projection_variable_lst = []
        modifier=None
        gml_variable=[]
        if 'modifier' in where_part.keys():
            modifier=where_part['modifier']
        for s in where_part['projection']:
            var_dict = {}
            if 'var' in s.keys():
                var_dict['type']='variable'
                var_dict['name']=str(s['var'])
                var_dict['variable'] = s['var']
            elif 'expr' in s.keys():
                var_dict['type'] = 'function'
                expr_variable=s['expr']
                while ((type(expr_variable) is rdflib.plugins.sparql.parserutils.Expr)
                        and (expr_variable.name.split('_')[-1] not in ['Builtin_','Function'])
                        and 'expr' in expr_variable.keys()):
                    expr_variable=expr_variable['expr']
                if expr_variable.name.startswith('Builtin_'):
                    exp_func=expr_variable.name.split("_")[1]
                    var_dict["name"]= exp_func
                    var_dict['args'] = {}
                    for idx,key in enumerate(expr_variable.keys()):
                        expr_arg = expr_variable[key]
                        while (type(expr_arg) is rdflib.plugins.sparql.parserutils.Expr) and 'expr' in expr_arg:
                            expr_arg = expr_arg['expr']
                        if type(expr_arg) is rdflib.term.Variable:
                            var_dict['args']["arg"+str(idx)] = expr_arg
                elif expr_variable.name=='Function':
                    exp_func=expr_variable['iri']['prefix']+":"+expr_variable['iri']['localname']
                    expr_variable = expr_variable['expr']
                    var_dict["name"]= exp_func
                    var_dict['args']={}
                    for idx,expr_arg in enumerate(expr_variable):
                        while (type(expr_arg) is rdflib.plugins.sparql.parserutils.Expr) and 'expr' in expr_arg:
                            expr_arg = expr_arg['expr']
                        if type(expr_arg) is rdflib.term.Variable:
                            var_dict['args']["arg"+str(idx)] = expr_arg
            if 'evar' in s.keys():
                var_dict['alias']=s['evar']
            projection_variable_lst.append(var_dict)

        from_graph=None
        if 'datasetClause' in where_part:
             from_graph= str(where_part['datasetClause'][0]['default'])
        triples_list = []
        SubSelect=None
        for where_stmt in where_part['where']['part']:
            if where_stmt.name in "TriplesBlock":
                for t in where_stmt['triples']:  # iterating through the triples
                    triples = {}
                    # subject = str(t[0])
                    subject = t[0]
                    if isinstance(t[1], rdflib.term.Variable):  # if predicate is a variable
                        # predicate = str(t[1])
                        predicate = t[1]
                    elif isinstance(t[1]['part'][0]['part'][0]['part'],
                                    rdflib.term.URIRef):  # else if predicate is a URI
                        predicate = str(t[1]['part'][0]['part'][0]['part'])
                    else:  # else it is a prefix:postfix pair
                        p_prefix = str(t[1]['part'][0]['part'][0]['part']['prefix'])
                        p_lName = str(t[1]['part'][0]['part'][0]['part']['localname'])
                        predicate = (p_prefix, p_lName)

                    object_ = t[2]
                    if not isinstance(object_, (
                            rdflib.term.Variable, rdflib.term.URIRef,
                            rdflib.term.Literal)):  # if object is not a URI or Variabel
                        # object_prefix = str(object_['prefix'])
                        if "prefix" in object_:
                            object_prefix = object_['prefix']
                            object_lName = object_['localname']
                            object_ = (object_prefix, object_lName)

                    triples['subject'] = subject
                    triples['predicate'] = predicate
                    # triples['object'] = str(object_)
                    triples['object'] = object_
                    triples['bgpType']='NonOptional'
                    triples_list.append(triples)
            elif where_stmt.name == "OptionalGraphPattern":
                for t in where_stmt['graph']['part'][0]['triples']:  # iterating through the triples
                    triples = {}
                    # subject = str(t[0])
                    subject = t[0]
                    if isinstance(t[1], rdflib.term.Variable):  # if predicate is a variable
                        # predicate = str(t[1])
                        predicate = t[1]
                    elif isinstance(t[1]['part'][0]['part'][0]['part'],
                                    rdflib.term.URIRef):  # else if predicate is a URI
                        predicate = str(t[1]['part'][0]['part'][0]['part'])
                    else:  # else it is a prefix:postfix pair
                        p_prefix = str(t[1]['part'][0]['part'][0]['part']['prefix'])
                        p_lName = str(t[1]['part'][0]['part'][0]['part']['localname'])
                        predicate = (p_prefix, p_lName)

                    object_ = t[2]
                    if not isinstance(object_, (
                            rdflib.term.Variable, rdflib.term.URIRef,
                            rdflib.term.Literal)):  # if object is not a URI or Variabel
                        # object_prefix = str(object_['prefix'])
                        if "prefix" in object_:
                            object_prefix = object_['prefix']
                            object_lName = object_['localname']
                            object_ = (object_prefix, object_lName)

                    triples['subject'] = subject
                    triples['predicate'] = predicate
                    # triples['object'] = str(object_)
                    triples['object'] = object_
                    triples['bgpType'] = 'Optional'
                    triples_list.append(triples)
            elif where_stmt.name == "Filter":  ## Optional triples
                filter_expr = where_stmt['expr']
                while len(filter_expr.keys()) != 3:
                    filter_expr = filter_expr['expr']
                if len(filter_expr.keys()) == 3:
                    filter_variable = filter_expr['expr']['expr']['expr']
                    filter_operator = filter_expr['op']
                    filter_value = filter_expr['other']['expr']['expr']
                    triples_list.append({'variable': filter_variable, 'operator': str(filter_operator),
                                             'value':filter_value, 'bgpType':'filter'})
            elif where_stmt.name == "GroupOrUnionGraphPattern":  ## Nested Select
                if where_stmt['graph'][0].name=='SubSelect':
                    SubSelect=where_stmt['graph'][0]
        where_limit = where_part['limitoffset']['limit'] if 'limitoffset' in where_part.keys() and 'limit' in where_part['limitoffset'].keys() else None
        where_offset = where_part['limitoffset']['offset'] if 'limitoffset' in where_part.keys() and 'offset' in where_part['limitoffset'].keys() else None
        return  modifier,projection_variable_lst,from_graph,triples_list,SubSelect,where_limit,where_offset
    def exec_query_plan(self):
        st=datetime.now()
        self.parse_select()
        DAG = self.build_DAG()
        # self.draw_DAG(DAG)
        decomposedSubqueries = self.DecomposeSubqueries(DAG)
        sparqlSubqueries = {}
        exectuted_Queries={}
        for subquery in decomposedSubqueries.keys():
            if subquery == "SQ1":
                sparqlSubqueries[subquery] = self.ReWriteSubqueryToSPARQL(decomposedSubqueries[subquery], DAG)
            else:
                sparqlSubqueries[subquery] = None
        print(f"Parse and Rewrite Time:{(datetime.now()-st).total_seconds()} Sec.")
        ##################################################
        st=datetime.now()
        KGMeta_Governer_ins = KGMeta_Governer(endpointUrl="http://206.12.98.118:8890/sparql", KGMeta_URI=Constants.KGNET_Config.KGMeta_IRI,RDFEngine=Constants.RDFEngine.OpenlinkVirtuoso)
        #################### get target node list ##########################
        res_df=KGMeta_Governer_ins.executeSparqlquery(sparqlSubqueries['SQ1'])
        exectuted_Queries["SQ1"]=sparqlSubqueries['SQ1']
        res_df=res_df.applymap(lambda x:str(x)[1:-1]) ## remove starting and ending qoutes from each cell
        target_node_res_lst=res_df[str(DAG['root'][0].var)].tolist()
        # target_node_res_lst=[str(elem).replace("\"","").strip() for elem in target_node_res_lst] ## replace starting and ending qoutes
        gml_vars = [var for var in DAG.keys() if var != 'root' and DAG[var].is_gml == True]
        dependent_gml_vars = [var for var in DAG.keys() if var != 'root' and (DAG[var].is_gml == False and DAG[var].gml_parent is not None)]
        print(f"SQ1 Exec Time:{(datetime.now() - st).total_seconds()} Sec.")
        print(f"# Traget Nodes={len(target_node_res_lst)}")
        ################### Loop on SQ2 infernce Queries #####################
        decomposedSubqueries['SQ2'].reverse()
        ######################
        for query in decomposedSubqueries['SQ2']:
            st=datetime.now()
            api_triples=[]
            non_api_triples=[]
            if 'triples' in query:
                for triple in query['triples']:
                    if triple['bgpType'] in ['Optional', 'NonOptional']:
                        s, p, o, bgpType = triple.values()
                        if str(s) in gml_vars or str(p) in gml_vars or str(o) in gml_vars:
                            api_triples.append(triple)
                        else:
                            non_api_triples.append(triple)
                    elif triple['bgpType'] in ['filter']:
                        var, op, val, bgpType = triple.values()
                        if str(var) in gml_vars or str(val) in gml_vars:
                            api_triples.append(triple)
                        else:
                            non_api_triples.append(triple)
            ################## get task and model id #################
            task_id_query,target_variable,predictetd_variable=getGMLOperatorTaskId(api_triples)
            task_df=KGMeta_Governer_ins.executeSparqlquery(task_id_query)
            exectuted_Queries[predictetd_variable+"_task_id"] = task_id_query
            if len(task_df) == 0:
                raise Exception("there is no trained model exist for GML operator: " + query)
            tid = task_df["tid"].values[0].replace('"', "")
            best_mid = KGMeta_Governer_ins.OptimizeForBestModel(tid)
            model_url = Constants.KGNET_Config.GML_API_URL + "gml_inference/mid/" + str(best_mid).replace('"', "")
            print(f"{predictetd_variable} Var Task/Model ID Fetch Time:{(datetime.now() - st).total_seconds()} Sec.")
            ################## peform model inference #################
            st=datetime.now()
            myobj = {"model_id": best_mid,
                     "named_graph_uri": "http://wikikg-v2",
                     "sparqlEndpointURL": "http://206.12.98.118:8890/sparql",
                     "RDFEngine": Constants.RDFEngine.OpenlinkVirtuoso,
                     "targetNodesList": target_node_res_lst,
                     "TOSG_Pattern": "d1h1",
                     "topk": 1}
            inference_res_dic = json.loads(dopost(model_url, myobj).decode("utf-8"))
            pred_values=["<"+elem+">" for elem in set([v for k,v in inference_res_dic.items() if k.startswith("http")])]
            print(f"{predictetd_variable} API Inference Time:{(datetime.now() - st).total_seconds()} Sec.")
            print(f"{predictetd_variable} #Predicted Nodes={len(inference_res_dic)-1}")
            print(f"{predictetd_variable} #Predicted Classes={len(pred_values)}")
            ################### filter target nodes based one Inferences results #################
            st=datetime.now()
            descendent_query=query.copy()
            descendent_query['triples']=non_api_triples
            descendent_query['select'].append(predictetd_variable)
            descendent_query_sparql=self.ReWriteSubqueryToSPARQL(descendent_query,DAG)
            descendent_query_sparql=descendent_query_sparql.replace("}",f" values {set_syntax(predictetd_variable)}"+"{"+" ".join(list(pred_values))+"}\n}")
            descendent_query_res_df = KGMeta_Governer_ins.executeSparqlquery(descendent_query_sparql)
            exectuted_Queries[predictetd_variable + "_descendent_query"] = descendent_query_sparql
            descendent_query_res_df = descendent_query_res_df.applymap(lambda x: str(x)[1:-1])  ## remove starting and ending qoutes from each cell
            descendent_query_filtered_pred_lst=descendent_query_res_df[str(predictetd_variable)].tolist()
            print(f"{predictetd_variable} predictions filter query Time:{(datetime.now() - st).total_seconds()} Sec.")
            ################# filter traget nodes ###################################################
            target_node_res_lst=[k for k,v in inference_res_dic.items() if v in descendent_query_filtered_pred_lst]
            res_df = res_df[res_df[str(DAG['root'][0].var)].isin(target_node_res_lst)]
            print(f"# Target Nodes After Filter :{len(target_node_res_lst)}")
            ################ Set Projection Columns Values #############################
            st=datetime.now()
            for col in descendent_query_res_df.columns.tolist():
                if col in res_df.columns.tolist():
                    res_df[col]=res_df[str(DAG['root'][0].var)].apply(lambda x:inference_res_dic[str(x)]) ## join with the parent varariable
                    col_dict=dict(zip(descendent_query_res_df[str(predictetd_variable)].tolist(),descendent_query_res_df[col].tolist())) ## join with the intermediate join var (predictetd_variable)
                    res_df[col] = res_df[col].apply(lambda x: col_dict[str(x)])
            print(f"Projection Columns Filling  Time:{(datetime.now() - st).total_seconds()} Sec.")
        return res_df,exectuted_Queries
    def parse_select(self):
        query = rdflibParseQuery(self.gmlquery)
        # formatted_gml_query = gmlQueryFormatter.format_gml_query_tree(query)
        # formatted_gml_query=gmlQueryFormatter.format_gml_query_str(str(query))
        flag_prefix = False
        query_type = ""
        distinct=False
        if len(query) >= 2 and query[0][0].name == 'PrefixDecl':  # check if prefix exist in the query
            flag_prefix = True
            prefix_part = query[0]
            where_part = query[1]
            # print(where_part)
        else:
            where_part = query[0]

        self.query_statments["query_type"] = where_part.name
        if (flag_prefix):  # store prefix in the dictionary
            dict_prefix = {}
            for p in prefix_part:
                dict_prefix[p['prefix']] = str(p['iri'])
            self.query_statments['prefixes'] = dict_prefix

        modifier,projection_variable_lst, from_graph, triples_list, SubSelect,where_limit,where_offset = self.breakdown_Sparql_Select_statment(where_part)
        self.query_statments['select'] = projection_variable_lst
        if modifier:
            self.query_statments['modifier'] = modifier
        self.query_statments['from'] = from_graph
        self.query_statments['triples'] = triples_list
        self.query_statments['limit'] = where_limit
        self.query_statments['offset'] = where_offset
        SubSelect_stmts=self.query_statments
        while SubSelect is not None:
            SubSelect_stmts['SubSelect'] = {}
            SubSelect_stmts["SubSelect"]["query_type"] = SubSelect.name
            modifier,projection_variable_lst, from_graph, triples_list, SubSelect, where_limit, where_offset = self.breakdown_Sparql_Select_statment(SubSelect)
            SubSelect_stmts['SubSelect']['select'] = projection_variable_lst
            if modifier:
                SubSelect_stmts['SubSelect']['modifier'] = modifier
            SubSelect_stmts['SubSelect']['from'] = from_graph
            SubSelect_stmts['SubSelect']['triples'] = triples_list
            SubSelect_stmts['SubSelect']['limit'] = where_limit
            SubSelect_stmts['SubSelect']['offset'] = where_offset
            SubSelect_stmts=SubSelect_stmts['SubSelect']
        return self.query_statments
    def parse_insert(self):
        self.query_statments["queryType"] = 'insertQuery'
        prefix_dic={}
        prefixes_str_lst=re.findall("\s+prefix\s+.*:<.*>\s*\n",self.gmlquery.lower())
        insert_command_str=re.findall("\n\s*insert\s+into\s+<kgnet>\s*\n", self.gmlquery.lower())[0]
        where_criteria_str=re.findall("\n\s*where\s*[\{]\s*(.*\n)*",self.gmlquery,re.DOTALL)[0]
        train_gml_json_str=re.findall("\s*select\s+\*\s+from\s+kgnet\s*\.\s*TrainGML\s*\(\s*(.*)", where_criteria_str, re.DOTALL)[0]+"}"
        try:
            gml_json_dict=json.loads(train_gml_json_str)
        except Exception as e:
            print(str(e))
            print("TrainGML JSON Str:",train_gml_json_str)
            raise Exception("TrainGML JSON Object not correctly formatted")

        self.query_statments["insertJSONObject"]=gml_json_dict

        for pref in prefixes_str_lst:
            prefix_parts=re.split('\s|:<|>', pref)
            prefix_parts=[x for x in prefix_parts if x]
            prefix_dic[prefix_parts[1]]=prefix_parts[2]
        self.query_statments["prefixes"]=prefix_dic
        return self.query_statments
    def parse_delete(self):
        self.query_statments["queryType"] = 'deleteQuery'
        return self.query_statments
    def getQueryType(self):
        return self.query_statments["queryType"] if "queryType" in self.query_statments.keys() else None
    def extractQueryStatmentsDict (self):
        """ The extract() function takes the raw SPARQL query and prases it using the function provided by rdf
            and returns a dictionary containing different modules of the SPAEQL query"""
        if re.findall("\n\s*insert\s+into\s+<kgnet>\s*\n",self.gmlquery.lower()).__len__() >0:
            return  self.parse_insert()
        elif re.findall("\n\s*delete\s+<kgnet>\s*\n",self.gmlquery.lower()).__len__():
            return  self.parse_delete()
        else:
            return self.parse_select()

