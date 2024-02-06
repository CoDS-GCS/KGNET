import json
import rdflib
import re
from rdflib.plugins.sparql.parser import parseQuery as rdflibParseQuery
from  SparqlMLaasService.QueryParser import  gmlQueryParser
from  SparqlMLaasService.QueryParser import  set_syntax
# from QueryFormatter import  gmlQueryFormatter
from rdflib.plugins.sparql.parserutils import CompValue
from pyparsing.results import ParseResults
import Constants

""" Production """
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
############################################### Constants ###############################
NODECLASSIFIER = 'nodeclassifier'
LINKPREDICTOR = 'linkpredictor'
GRAPHCLASSIFIER = 'graphclassifier'
MISC = 'MISC'
lis_classTypes = ['nodeclassifier','linkpredictor','kgnet:types/nodeclassifier','kgnet:types/linkpredictor']
dict_classifTypes = {NODECLASSIFIER:['nodeclassifier','kgnet:types/nodeclassifier',],
                   LINKPREDICTOR:['linkpredictor','kgnet:types/linkpredictor'],
                   GRAPHCLASSIFIER:['graphclassifier']}

lis_targetNodeTypes = ['kgnet:targetnode']
lis_labelTypes = ['kgnet:labelnode']
lis_targetEdgeTypes = ['kgnet:targetedge']

clf_fxn = {  # 'nodeclassifier':'getNodeClass',
    'nodeclassification': 'getNodeClass_v2',
    'type/nodeclassification': 'getNodeClass_v2',
    'kgnet:type/nodeclassification': 'getNodeClass_v2',
    'type/linkprediction': 'getLinkPred',
    'kgnet:type/linkprediction': 'getLinkPred',
    'linkprediction': 'getLinkPred'}
##############################################################################################


class gmlQueryRewriter:
    def __init__(self,query_dict,KGMeta_Governer):
       self.query_dict=query_dict
       self.KGMeta_Governer=KGMeta_Governer

    def rewrite_gml_select_queries(self, KGMeta_prefix="kgnet",modelURI=None):
        """return data query a candidate  form 2 for sparqlML query and KGMeta query"""
        string_q = ""
        string_gml = ""
        bool_gml = False
        if len(self.query_dict['prefixes']) > 0:
            for prefix, uri in self.query_dict['prefixes'].items():
                if prefix.lower() == KGMeta_prefix:
                    bool_gml = True
                s = f"PREFIX {prefix}: <{uri}>\n"
                # string_q.join(s)
                string_q += s
                string_gml += s
        if len(self.query_dict['select']) > 0:
            string_q += 'SELECT'
            for item in self.query_dict['select']:
                if item['type']=='variable':
                    s = f" ?{item['name']} "
                elif item['type'] == 'function':
                    s = f" {item['name']}( "
                    for k,v in item['args'].items():
                        s+=f" ?{v}, "
                string_q += s
        string_q += "\n WHERE { \n"

        data_triples, gml_triples,gml_and_dependant_variables = self.split_data_gml_triples()
        dict_gml_var_dict,prep_gml_vars = self.prep_gml_vars(gml_triples)
        # string_gml += self.get_KGMeta_gmlOperatorQuery(dict_gml_var_dict, data_triples)
        # gml_query_res_df=self.KGMeta_Governer.executeSparqlquery(string_gml)
        tasks_queries_dict=self.getGMLOperatorTaskId(dict_gml_var_dict, data_triples)
        kgmeta_model_queries_dict= {}
        for gml_op in tasks_queries_dict.keys():
            query = string_gml+tasks_queries_dict[gml_op]
            kgmeta_model_queries_dict[str(gml_op)]=query
            task_df=self.KGMeta_Governer.executeSparqlquery(query)
            if len(task_df)==0:
                raise Exception("there is no trained model exist for GML operator: "+str(gml_op))
            tid = task_df["tid"].values[0].replace('"', "")
            best_mid = self.KGMeta_Governer.OptimizeForBestModel(tid)
            # dict_gml_var_dict['$m']=Constants.KGNET_Config.GML_API_URL+"gml_inference/mid/"+gml_query_res_df['mID'].values[0].replace('"', "")
            dict_gml_var_dict[gml_op]['$m'] = Constants.KGNET_Config.GML_API_URL + "gml_inference/mid/" + str(best_mid).replace('"', "")
        sparql_candidate_query,kg_data_query,gmlop_targetNode_queries_dict,model_ids = self.get_breakdown_queries(dict_gml_var_dict, data_triples,gml_and_dependant_variables)
        return (sparql_candidate_query, kg_data_query,gmlop_targetNode_queries_dict,kgmeta_model_queries_dict,model_ids)

    def split_data_gml_triples(self):
        '''split query triples into data and GML triples lists. return list of data triples,gml triples, and gml variables '''
        KGNET_LOOKUP = ['kgnet', '<kgnet']
        data_triples = []
        gml_triples = []
        ########## split based on KGNET Prefix ################
        q_dict = self.query_dict
        while True:
            for triple_type in ['triples']:
                triples=q_dict[triple_type]
                for t in triples:
                    if 'bgpType' in t.keys() and t['bgpType'] in['NonOptional','Optional']:
                        values = t.values()
                        flagT_gml = False
                        # print(t,'\n')
                        for v in values:
                            if isinstance(v, str) and v.split(':')[0] in KGNET_LOOKUP:
                                gml_triples.append(t)
                                flagT_gml = True
                                break
                        if not flagT_gml:
                            data_triples.append(t)
            if "SubSelect" in q_dict.keys():
                q_dict=q_dict["SubSelect"]
            else:
                break
        ################## list GML variables ###################
        gml_and_dependant_variables=[]
        for triple in gml_triples:
            s,p,o,bgp_type=triple.values()
            if isinstance(s,rdflib.term.Variable) and s not in gml_and_dependant_variables:
                gml_and_dependant_variables.append(s)
            if isinstance(o,rdflib.term.Variable) and o not in gml_and_dependant_variables:
                gml_and_dependant_variables.append(o)
        ######## Dependent Object Variables ###############
        q_dict = self.query_dict
        query_levels_lst=[]
        while True: ## order is important , start by nested select varibles (buttom up)
            query_levels_lst.insert(0,q_dict)
            if "SubSelect" in q_dict.keys():
                q_dict = q_dict["SubSelect"]
            else:
                break
        for q_level in query_levels_lst:
            q_level_triples=q_level['triples']
            for triple in q_level_triples:
                s, p, o,bgp_type = triple.values()
                if bgp_type in['Optional','NonOptional'] and s in gml_and_dependant_variables:
                    # if isinstance(p, rdflib.term.Variable) and p not in gml_variables:
                    #     gml_variables.append(p)
                    if isinstance(o, rdflib.term.Variable) and o not in gml_and_dependant_variables:
                        gml_and_dependant_variables.append(o)
                if bgp_type in['Optional','NonOptional']  and p in gml_and_dependant_variables:
                    # if isinstance(s, rdflib.term.Variable) and s not in gml_variables:
                    #     gml_variables.append(s)
                    if isinstance(o, rdflib.term.Variable) and o not in gml_and_dependant_variables:
                        gml_and_dependant_variables.append(o)
            ####### Dependent select variables ################
            for var in q_dict['select']:
                if var['type']=='variable':
                    if 'alias' in var.keys() and var['name'] in gml_and_dependant_variables and var['alias'] not in gml_and_dependant_variables:
                        gml_and_dependant_variables.append(var['alias'])
                elif var['type'] == 'function' and 'alias' in var.keys() and 'args' in var.keys() and var['alias'] not in gml_and_dependant_variables:
                    for val in var['args'].values():
                        if val in gml_and_dependant_variables and var['alias'] not in gml_and_dependant_variables:
                            gml_and_dependant_variables.append(var['alias'])
                            break

        # ################## remove GML dependant triples from data triples ###################
        # to_remove_triples=[]
        # set()
        # for triple in data_triples:
        #     s, p, o = triple.values()
        #     if (s in gml_and_dependant_variables or p in gml_and_dependant_variables or o in gml_and_dependant_variables) and p not in gml_variables: ## keep GML user defined perdicate in data triples
        #         to_remove_triples.append(triple)
        # for remove_t in to_remove_triples:
        #     data_triples.remove(remove_t)
        #     # gml_triples.append(remove_t)
        #####################################################################
        return (data_triples, gml_triples,gml_and_dependant_variables)

    def prep_gml_vars(self, gml_dict):
        """The  prep_gml_vars () function takes input the gml dictionary of SPARQL-ML BGP statements and identifies the classification type,
                targets and labels per GML Operator and returns a dictionary containing the aforementioned information
                of the triples per GML Operator. """

        dict_vars = {}
        # dict_vars[MISC]=[]
        """ Identify Classification Type """
        gml_operator_type = []
        for triple in gml_dict:
            keys_lst=[key for key in dict_classifTypes if isinstance(triple['object'], rdflib.term.URIRef) \
             and triple['object'].lower() in dict_classifTypes.get(key, [])]
            if len(keys_lst)>0:
                gml_operator_type.append(keys_lst[0])

        gml_operator_type=list(set(gml_operator_type))
        if len(gml_operator_type) ==0:
            return

        for triple in gml_dict:
            if isinstance(triple['object'], rdflib.term.URIRef) and triple['object'].lower() in lis_classTypes:
                dict_vars[triple["subject"]]={}
                dict_vars[triple["subject"]]['gml_operator_class']={'subject': triple["subject"],"predicate":triple["predicate"],"object":rdflib.term.URIRef('kgnet:type/GMLTask')}
                if str(triple['object']) == 'kgnet:types/NodeClassifier':
                    dict_vars[triple["subject"]]['gml_operator_type'] = {'subject': triple["subject"], "predicate": rdflib.term.URIRef('kgnet:GMLTask/taskType'),"object": rdflib.term.URIRef('kgnet:type/nodeClassification')}
                elif str(triple['object']) == 'kgnet:types/LinkPredictor':
                    dict_vars[triple["subject"]]['gml_operator_type'] = {'subject': triple["subject"],"predicate": rdflib.term.URIRef('kgnet:GMLTask/taskType'),"object": rdflib.term.URIRef('kgnet:type/linkPrediction')}
                else :
                    raise 'GML operator Not supported: \"'+triple['object'].lower() +'\"'
                continue
            elif isinstance(triple['predicate'], str) and triple['predicate'].lower() in lis_targetNodeTypes:
                triple['predicate']=rdflib.term.URIRef('kgnet:GMLTask/targetNode')
                dict_vars[triple["subject"]]['target'] = triple  # ['object']         # if the triple is the target node type
                continue

            elif isinstance(triple['predicate'], str) and triple['predicate'].lower() in lis_targetEdgeTypes:
                triple['predicate'] = rdflib.term.URIRef('kgnet:GMLTask/targetEdge')
                dict_vars[triple["subject"]]['targetEdge'] = triple  # ['object']         # if the triple is the target edge type
                continue

            elif isinstance(triple['predicate'], str) and triple['predicate'].lower() in lis_labelTypes:
                triple['predicate'] = rdflib.term.URIRef('kgnet:GMLTask/labelNode')
                dict_vars[triple["subject"]]['label'] = triple  # ['object']          # if the triple is the label
                continue

            elif isinstance(triple['predicate'], str) and triple['predicate'].lower() in ["kgnet:gnnmethod"]:
                triple['predicate'] = rdflib.term.URIRef('kgnet:GMLModel/GNNMethod')
                dict_vars[triple["subject"]]['GNNMethod'] = triple  # ['object']         # if the triple is the GNN Method
                continue
            elif isinstance(triple['predicate'], str) and triple['predicate'].lower() in ["kgnet:topk"]:
                triple['predicate'] = rdflib.term.URIRef('kgnet:topk')
                dict_vars[triple["subject"]]['topk'] = triple  # ['object']         # if the triple is the GNN Method
                continue

            else:
                if MISC not in dict_vars.keys():
                    dict_vars[MISC] = []
                dict_vars[MISC].append(triple)
        return dict_vars,gml_operator_type

    def get_rdfType (self,list_data_T,var,rdf_type='http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
        """ The get_rdfType () function takes inputs a list of triples and a subject variable
                and traverses through the triples to identiy the rdf type of the subject """
        rdf_type = rdf_type.lower()
        var_string=  var.string.lower() if isinstance(var,rdflib.plugins.sparql.parserutils.CompValue)  else var.lower()
        for t in list_data_T:
            if  str(t['subject']).lower() == var_string and str(t['predicate']).lower() == rdf_type:
                return set_syntax(t['object'])

    def get_KGMeta_gmlOperatorQuery (self,dict_vars,data_vars):
        """ The function takes dictionary of variables required for the generation of GML operator query and returns the the query """
        # SELECT {set_syntax(dict_vars['gml_operator']['subject'])}
        string_Q = f"""
        SELECT max(?mID) as ?mID
        from <"""+Constants.KGNET_Config.KGMeta_IRI+""">
        WHERE
        """
        try :
            target_type = dict_vars['target']['object'] if 'target' in dict_vars else dict_vars['targetEdge']['object']  if 'targetEdge' in dict_vars else None
            target_type = set_syntax(self.get_rdfType(data_vars, target_type.split(':')[1]) if (isinstance(target_type,str) and self.get_rdfType(data_vars, target_type.split(':')[1]) is not None) else target_type)

            string_Q+="{"
            for key in dict_vars.keys():
                if len (dict_vars[key]) ==0:
                    continue
                elif key in['target','label']:
                    string_temp = f"{set_syntax(dict_vars[key]['subject'])} {set_syntax(dict_vars[key]['predicate'])} <{set_syntax(dict_vars[key]['object'])}> .\n"
                    string_Q+=string_temp
                    continue
                elif key == 'targetEdge':
                    string_temp = f"{set_syntax(dict_vars[key]['subject'])} {set_syntax(dict_vars[key]['predicate'])} {set_syntax(dict_vars[key]['object'])} .\n"
                    string_Q += string_temp
                    continue

                elif isinstance(dict_vars[key],list):
                    for list_triple in dict_vars[key]:
                        if list_triple['predicate'].lower()!='kgnet:topk':
                            string_temp=f"{set_syntax(list_triple['subject'])} {set_syntax(list_triple['predicate'])} {set_syntax(list_triple['object'])} .\n"
                            string_Q+=string_temp
                    continue
                else:
                    string_temp = f"{set_syntax(dict_vars[key]['subject'])} {set_syntax(dict_vars[key]['predicate'])} {set_syntax(dict_vars[key]['object'])} .\n"
                    string_Q+=string_temp

            string_Q+=f"{set_syntax(dict_vars['gml_operator_class']['subject'])} <kgnet:GMLTask/modelID> ?gmlModel .\n"
            string_Q+="""?gmlModel <kgnet:GMLModel/id> ?mID .  """
            string_Q +="}"
        except Exception as e :
            print(e)
            raise Exception("GML specifications are incomplete in the query")

        return string_Q

    def getGMLOperatorTaskId(self, dict_vars, data_vars):
        """ The function takes dictionary of parsed GML operators and returns a dictionay of KGMeta task-id query per operator """
        GML_taskId_Query_dict= {}
        for gml_op in dict_vars:
            string_Q = f"""
               SELECT ?tid
               from <""" + Constants.KGNET_Config.KGMeta_IRI + """>
               WHERE
               """
            try:
                target_type = dict_vars[gml_op]['target']['object'] if 'target' in dict_vars[gml_op] else dict_vars[gml_op]['targetEdge'][
                    'object'] if 'targetEdge' in dict_vars[gml_op] else None
                if target_type:
                    target_type = set_syntax(self.get_rdfType(data_vars, target_type.split(':')[-1]) if (
                                isinstance(target_type, str) and self.get_rdfType(data_vars, target_type.split(':')[
                            -1]) is not None) else target_type)

                string_Q += "{"
                for key in dict_vars[gml_op].keys():
                    if len(dict_vars[gml_op][key]) == 0:
                        continue
                    elif key in ['target', 'label']:
                        obj=set_syntax(dict_vars[gml_op][key]['object'])
                        obj= ("<"+str(obj)+">") if (str(obj).startswith("<") or str(obj).startswith("http") or ":" in str(obj) ) else str(obj)
                        string_temp = f"{set_syntax(dict_vars[gml_op][key]['subject'])} {set_syntax(dict_vars[gml_op][key]['predicate'])} {obj} .\n"
                        string_Q += string_temp
                        continue
                    elif key == 'targetEdge':
                        string_temp = f"{set_syntax(dict_vars[gml_op][key]['subject'])} {set_syntax(dict_vars[gml_op][key]['predicate'])} {set_syntax(dict_vars[gml_op][key]['object'])} .\n"
                        string_Q += string_temp
                        continue
                    elif key.lower() in ['topk','gnnmethod']:
                        continue
                    elif isinstance(dict_vars[gml_op][key], list):
                        for list_triple in dict_vars[gml_op][key]:
                            if str(list_triple['predicate']).lower() not in ['kgnet:topk','kgnet:gml/gnnmethod']:
                                string_temp = f"{set_syntax(list_triple['subject'])} {set_syntax(list_triple['predicate'])} {set_syntax(list_triple['object'])} .\n"
                                string_Q += string_temp
                        continue
                    else:
                        string_temp = f"{set_syntax(dict_vars[gml_op][key]['subject'])} {set_syntax(dict_vars[gml_op][key]['predicate'])} {set_syntax(dict_vars[gml_op][key]['object'])} .\n"
                        string_Q += string_temp

                string_Q += f"{set_syntax(dict_vars[gml_op]['gml_operator_class']['subject'])} <kgnet:GMLTask/id> ?tid .\n"
                # string_Q += """?gmlModel <kgnet:GMLModel/id> ?mID .  """
                string_Q += "}"
                GML_taskId_Query_dict[gml_op]=string_Q
            except Exception as e:
                print(e)
                raise Exception("GML specifications are incomplete in the query")

        return GML_taskId_Query_dict
    def get_target_label_variables(self,query,userDefinedPredicate):
        query_part=query
        while True:
            query_triples=query_part['triples']
            for triple in query_triples:
                if triple['predicate']==userDefinedPredicate:
                    return triple['subject'],triple['object']
            if 'SubSelect' in query_part.keys():
                query_part=query_part['SubSelect']
            else:
                break
        return None

    def ReWrite_OuterQuery_Statments(self, query,userDefinedPredicate_lst, gmlOperatorType_dict, userDefinedPredicate_dict,target_variables_dict, label_variables_dict,list_data_T, model_uri_dict,gml_target_label_postfix,gml_and_dependant_variables,rewriteForm=1):
        ''' rewrite Parsed SPARQL-ML Outer Query statments '''
        string_q=string_q_dataonly=string_q_target_nodes=""
        gml_and_dependant_variables_names=[str(elem) for elem in gml_and_dependant_variables]
        ############################## Projection Variables #######################
        if len(query['select'])>0:
            string_q+='SELECT'
            string_q_dataonly+='SELECT'
            if 'modifier' in query.keys() and query['modifier'] is not None:
                string_q+=f" {query['modifier']} "
                string_q_dataonly += f" {query['modifier']} "
            string_q_target_nodes+= 'SELECT distinct ?s \n'
            labels_dict= { str(v):k for k,v in label_variables_dict.items()}
            for item in query['select']:
                if item['type']=='variable':
                    if item['name'] in labels_dict.keys():
                        target_label =  item['name']+gml_target_label_postfix
                        target_variable=str(target_variables_dict[labels_dict[item['name']]])
                        if rewriteForm==1: ## nested Call Form
                            string_q+= f'\n (kgnetML:getKeyValue_v2({set_syntax(rdflib.term.Variable(target_variable))},{set_syntax(rdflib.term.Variable(target_label))}) as {set_syntax(rdflib.term.Variable(item["name"]))} ) '
                        else: ## Call Per Target Form
                            string_q += f'\n (kgnetML:getSingleNodePrediction({set_syntax(rdflib.term.Variable(target_variable))},"$getSingleNodePrediction$") as {set_syntax(rdflib.term.Variable(item["name"]))} )'

                        continue
                    s = f" ?{ item['name']} "
                    string_q+=s
                    if (item['name'] not in gml_and_dependant_variables_names) and ("alias" not in item.keys() or item["alias"] not in gml_and_dependant_variables_names):
                        string_q_dataonly+=s
                elif item['type'] == 'function':
                    s = f"{item['name']}( "
                    for arg,val in item['args'].items():
                        if str(val) in labels_dict.keys():
                            target_label = str(val) + gml_target_label_postfix
                            target_variable = str(target_variables_dict[labels_dict[str(val)]])
                            if rewriteForm == 1:  ## nested Call Form
                                s+= f'kgnetML:getKeyValue_v2({set_syntax(rdflib.term.Variable(target_variable))},{set_syntax(rdflib.term.Variable(target_label))})'
                            else:  ## Call Per Target Form
                                s += f'\n (kgnetML:getSingleNodePrediction({set_syntax(rdflib.term.Variable(target_variable))},"$getSingleNodePrediction$") '
                            continue
                        else:
                            s+="?"+val+","
                    s=s[0:-1]+") "
                    if "alias" in item.keys():
                        s=s+") as ?"+item["alias"]+" "
                    else:
                        s = s + ") as ?fun_" + str(item["name"])+ ") "
                    string_q+=s
                    ########################## Check if function dependent on GML variable #####################
                    if (item['name'] not in gml_and_dependant_variables_names) and ("alias" not in item.keys() or str(item["alias"]) not in gml_and_dependant_variables_names):
                        has_gml_arg=False
                        for arg in item['args'].values():
                            if (str(arg)  in gml_and_dependant_variables_names):
                                has_gml_arg=True
                                break
                        if has_gml_arg==False:
                            string_q_dataonly+=s
        ############################## Projection Variables #######################
        if "from" in query.keys() and query['from'] is not None:
            string_q += "\nfrom <" + query['from'] + "> "
            string_q_dataonly += "\nfrom <" + query['from'] + "> "
            string_q_target_nodes += "\nfrom <" + query['from'] + "> "
        ############################## Where Block Start  #######################
        string_q += "$\nWHERE\n{\n$"
        string_q_dataonly += "\nWHERE\n{ \n"
        string_q_target_nodes += "\nWHERE\n{ \n"
        ############################## BGP Triples/Optinal  #######################
        gml_label_as_subject_triples={}
        # all_triples={'triples':query['triples'],'optional':query['optional_triples']}
        for triple in query['triples']:
            if triple["bgpType"] in['NonOptional','Optional'] and triple in list_data_T:
                string_t = ""
                s = triple['subject']  # if not isinstance(triple['subject'], tuple) else triple['subject'][1]
                p = triple['predicate']  # if not isinstance(triple['predicate'], tuple) else triple['predicate'][1]
                o = triple['object']  # if not isinstance(triple['object'], tuple) else triple['object'][1]
                s = set_syntax(s)
                p = set_syntax(p)
                o = set_syntax(o)
                if str(triple['predicate']) in userDefinedPredicate_lst:
                    gml_op = list(userDefinedPredicate_dict.keys())[userDefinedPredicate_lst.index(str(triple['predicate']))]
                    if rewriteForm==1:
                        sub_query = '\n{'
                        sub_query += f"SELECT (kgnetML:{clf_fxn[gmlOperatorType_dict[gml_op].lower()]}(\"{model_uri_dict[gml_op]}\",?$API_JSON_{str(gml_op)}$) \n as {set_syntax(label_variables_dict[str(gml_op)]) + gml_target_label_postfix} )"
                        sub_query += ' WHERE {}}\n'
                        string_q += sub_query
                    else:
                        sub_query =sub_query.replace("$getSingleNodePrediction$",f"\"{model_uri_dict[gml_op]}\",?$API_JSON_{str(gml_op)}$ ")
                    continue
                if ('SubSelect' not in query.keys() or str(triple['subject']) not in [str(elem['alias']) if 'alias' in elem.keys() else str(elem['name']) for elem in query['SubSelect']['select']])   and str(triple['subject']) in gml_and_dependant_variables_names:
                    if 'triples' not in gml_label_as_subject_triples.keys():
                        gml_label_as_subject_triples['triples']=[]
                    gml_label_as_subject_triples['triples'].append(triple)
                    continue
                # s=set_syntax(s)
                # p=set_syntax(p)
                if triple["bgpType"]=='Optional':
                    string_t += "optional{ "+f"{s} {p} {o}"+ ". }\n"
                else:
                    string_t += f"{s} {p} {o} .\n"
                string_q += string_t
                if str(triple['subject']) not in gml_and_dependant_variables_names and str(triple['predicate']) not in gml_and_dependant_variables_names and str(triple['object']) not in gml_and_dependant_variables_names:
                    string_q_dataonly += string_t
                    string_q_target_nodes += string_t
        ############################## BGP Filters  #######################
            elif triple["bgpType"] in['filter'] :
                string_t = f"filter({set_syntax(triple['variable'])} {triple['operator']}  {set_syntax(triple['value'])})\n"
                if  ('SubSelect' not in query.keys()) and str(triple['variable']) in gml_and_dependant_variables_names:
                    # or str(filter['variable']) not in [str(elem['alias']) if 'alias' in elem.keys() else str(elem['name']) for elem in query['SubSelect']['select']]
                    if 'filters' not in gml_label_as_subject_triples.keys():
                        gml_label_as_subject_triples['filters'] = []
                    gml_label_as_subject_triples['filters'].append(triple)
                else:
                    string_q += string_t
                    if str(triple['variable']) not in gml_and_dependant_variables_names:
                        string_q_dataonly += string_t
                        string_q_target_nodes += string_t
        ###################### Sub Select ##########################
        if "SubSelect" in query.keys():
            string_q += "$$SubSelect$$\n"
            string_q_dataonly += "$$SubSelect$$\n"
            string_q_target_nodes += "$$SubSelect$$\n"
        ################### Where Block End ########################
        string_q += "}"
        string_q_dataonly += "} "
        string_q_target_nodes += "} "
        ################### Limit and Offset ####################
        if "limit" in query.keys():
            string_q += f'\n LIMIT {query["limit"]}' if query["limit"] is not None else ""
            string_q_dataonly += f'\n LIMIT {query["limit"]}' if query["limit"] is not None else ""
            # string_q_target_nodes += f'\n LIMIT {query["limit"]}' if query["limit"] is not None else ""
        if "offset" in query.keys():
            string_q += f'\n offset {query["offset"]}' if query["offset"] is not None else ""
            string_q_dataonly += f'\n offset {query["offset"]}' if query["offset"] is not None else ""
            # string_q_target_nodes += f'\n offset {query["offset"]}' if query["offset"] is not None else ""
        # string_q_dataonly += "\n} filter(!isBlank(?o)). }"
        ################ gml_label_as_subject_of_triple must replae Model prediction string value with IRI ##################
        if len(gml_label_as_subject_triples.keys())>0:
            gml_label_as_subject_subselect_projection = "select distinct"
            gml_label_as_subject_subselect_triples = ""
            if 'triples' in gml_label_as_subject_triples.keys():
                for triple in gml_label_as_subject_triples['triples']:
                    s,p,o = triple['subject'],triple['predicate'],triple['object']
                    s,p,o = set_syntax(s), set_syntax(p),set_syntax(o)
                    gml_label_as_subject_subselect_triples+=f"{s} {p} {o} .\n"
                    if str(triple['subject']) in [str(elem) for elem in label_variables_dict.values()]:
                        gml_label_as_subject_subselect_projection+=" (IRI"
                        target_label = str(triple['subject'])  + gml_target_label_postfix
                        target_variable = str(target_variables_dict[labels_dict[str(triple['subject'])]])
                        gml_label_as_subject_subselect_projection += f'(kgnetML:getKeyValue_v2({set_syntax(rdflib.term.Variable(target_variable))},{set_syntax(rdflib.term.Variable(target_label))})) as {set_syntax(rdflib.term.Variable(triple["subject"]))} ) '
                for var in query['select']: ## add nested select variables
                    if var['type']=='variable' and var['name'] not in gml_and_dependant_variables_names:
                        gml_label_as_subject_subselect_projection+=" ?"+var['name']+" "
            if 'filters' in gml_label_as_subject_triples.keys():
                for filter in gml_label_as_subject_triples['filters']:
                    gml_label_as_subject_subselect_triples+= f"filter({set_syntax(filter['variable'])} {filter['operator']} {set_syntax(filter['value'])} ).\n"
            string_q=string_q.replace("$\nWHERE\n{\n$","\nWHERE\n{\n"+gml_label_as_subject_subselect_triples+"\n{\n"+gml_label_as_subject_subselect_projection+"\nwhere{\n")
            string_q+="\n}\n}"
        else:
            string_q = string_q.replace("$\nWHERE\n{\n$","\nWHERE\n{\n")

        return string_q,string_q_dataonly,string_q_target_nodes

    def ReWrite_OuterQuery_Statments_to_ast(self, query,userDefinedPredicate_lst, gmlOperatorType_dict, userDefinedPredicate_dict,target_variables_dict, label_variables_dict,list_data_T, model_uri_dict,gml_target_label_postfix,gml_and_dependant_variables,rewriteForm=1):
        ''' rewrite Parsed SPARQL-ML Outer Query statments '''
        gml_and_dependant_variables_names=[str(elem) for elem in gml_and_dependant_variables]
        ############################## Projection Variables #######################
        sparqlml_ast={}
        data_triples_ast = {}
        target_nodes_ast = {}
        if len(query['select'])>0:
            sparqlml_ast['type']=data_triples_ast['type']=target_nodes_ast['type']='select'
            if 'modifier' in query.keys() and query['modifier'] is not None:
                sparqlml_ast['modifier'] = query['modifier']
                data_triples_ast['modifier'] =  query['modifier']
            target_nodes_ast['modifier']='distinct'
            labels_dict= { str(v):k for k,v in label_variables_dict.items()}
            sparqlml_ast['projection'] =[]
            data_triples_ast['projection']=[]
            target_nodes_ast['projection'] = []
            target_nodes_ast['projection'].append({'type':'variable','is_target_label':False,'is_gml_or_dependant_variables':False,'name':rdflib.term.Variable("s")})
            for item in query['select']:
                var = {}
                var['is_target_label'] = False
                var['is_gml_or_dependant_variables'] = False
                if (item['name']  in gml_and_dependant_variables_names):
                    var['is_gml_or_dependant_variables'] = True
                if item['type']=='variable':
                    var['type'] = 'variable'
                    var['name'] = rdflib.term.Variable(item['name'])
                    if item['name'] in labels_dict.keys():
                        var['is_target_label'] = True
                        var['target_label'] =rdflib.term.Variable(item['name']+gml_target_label_postfix)
                        var['target_variable'] = rdflib.term.Variable(str(target_variables_dict[labels_dict[item['name']]]))
                        var['alias'] = rdflib.term.Variable(item["name"])
                    sparqlml_ast['projection'].append(var)
                    if (item['name'] not in gml_and_dependant_variables_names) and ("alias" not in item.keys() or item["alias"] not in gml_and_dependant_variables_names):
                        data_triples_ast['projection'].append(var)
                elif item['type'] == 'function':
                    var['type'] = 'function'
                    var['name'] = rdflib.term.Variable(item['name'])
                    var['args'] = {}
                    for idx,(arg,val) in enumerate(item['args'].items()):
                        fun_arg={}
                        fun_arg['type'] = 'variable'
                        fun_arg['is_target_label'] = False
                        fun_arg['name'] = rdflib.term.Variable(val)
                        if str(val) in labels_dict.keys():
                            fun_arg['is_target_label'] = True
                            fun_arg['target_label'] = rdflib.term.Variable(str(val) + gml_target_label_postfix)
                            fun_arg['target_variable'] = rdflib.term.Variable(str(target_variables_dict[labels_dict[str(val)]]))
                        var['arg'+str(idx)]=fun_arg
                    if "alias" in item.keys():
                        var['alias']=rdflib.term.Variable(item["alias"])
                    else:
                        var['alias'] = rdflib.term.Variable("fun_" + str(item["name"]))
                    sparqlml_ast['projection'].append(var)
                    ########################## Check if function dependent on GML variable #####################
                    if (item['name'] not in gml_and_dependant_variables_names) and ("alias" not in item.keys() or str(item["alias"]) not in gml_and_dependant_variables_names):
                        has_gml_arg=False
                        for arg in item['args'].values():
                            if (str(arg)  in gml_and_dependant_variables_names):
                                has_gml_arg=True
                                break
                        if has_gml_arg==False:
                            data_triples_ast['projection'].append(var)
        ############################## Projection Variables #######################
        if "from" in query.keys() and query['from'] is not None:
            sparqlml_ast['from']=query['from']
            data_triples_ast['from']=query['from']
            target_nodes_ast['from'] = query['from']
        ############################## Where Block Start  #######################
        sparqlml_ast['where'] =[]
        data_triples_ast['where'] =[]
        target_nodes_ast['where'] = []
        ############################## BGP Triples/Optinal  #######################
        gml_label_as_subject_triples={}
        # all_triples={'triples':query['triples'],'optional':query['optional_triples']}
        # for k,triples in all_triples.items():
        for triple in query['triples']:
            if triple["bgpType"] in['NonOptional','Optional'] and triple in list_data_T:
                BGP={}
                BGP['type']='triple'
                BGP['triple'] = {'subject':triple['subject'],'predicate':triple['predicate'],'object':triple['object']}
                BGP['is_userDefinedPredicate']=False
                if str(triple['predicate']) in userDefinedPredicate_lst:
                    gml_op = list(userDefinedPredicate_dict.keys())[userDefinedPredicate_lst.index(str(triple['predicate']))]
                    BGP['is_userDefinedPredicate'] = True
                    BGP['gml_udf']="kgnetML:"+clf_fxn[gmlOperatorType_dict[gml_op].lower()]
                    BGP['model_uri'] = model_uri_dict[gml_op]
                    BGP['json_request_body'] = f"?$API_JSON_{str(gml_op)}$"
                if ('SubSelect' not in query.keys() or str(triple['subject']) not in [str(elem['alias']) if 'alias' in elem.keys() else str(elem['name']) for elem in query['SubSelect']['select']])   and str(triple['subject']) in gml_and_dependant_variables_names:
                    if 'triples' not in gml_label_as_subject_triples.keys():
                        gml_label_as_subject_triples['triples']=[]
                    gml_label_as_subject_triples['triples'].append(triple)
                    continue
                if triple["bgpType"] in['Optional']:
                    BGP['optional']='optional'
                sparqlml_ast['where'].append(BGP)
                if str(triple['subject']) not in gml_and_dependant_variables_names and str(triple['predicate']) not in gml_and_dependant_variables_names and str(triple['object']) not in gml_and_dependant_variables_names:
                    data_triples_ast['where'].append(BGP)
                    if str(triple['subject']) in [str(elem) for elem in target_variables_dict.values()]:
                        BGP['triple']['subject']=rdflib.term.Variable("s")
                    if str(triple['object']) in [str(elem) for elem in target_variables_dict.values()]:
                        BGP['triple']['object'] = rdflib.term.Variable("s")
                    target_nodes_ast['where'].append(BGP)
        ############################## BGP Filters  #######################
            elif triple["bgpType"] in['filter']:
                BGP = {}
                string_t = ""
                BGP['type'] = 'filter'
                sparqlml_ast['where'].append(BGP)
                BGP['variable'] = triple['variable']
                BGP['operator'] = triple['operator']
                BGP['depends_on_gml_variable'] = False
                string_t = f"filter({set_syntax(triple['variable'])} {triple['operator']}  {set_syntax(triple['value'])})\n"
                if  ('SubSelect' not in query.keys()) and str(triple['variable']) in gml_and_dependant_variables_names:
                    BGP['depends_on_gml_variable'] = True
                    # or str(filter['variable']) not in [str(elem['alias']) if 'alias' in elem.keys() else str(elem['name']) for elem in query['SubSelect']['select']]
                    if 'filters' not in gml_label_as_subject_triples.keys():
                        gml_label_as_subject_triples['filters'] = []
                    gml_label_as_subject_triples['filters'].append(triple)
                else:
                    sparqlml_ast['where'].append(BGP)
                    if str(triple['variable']) not in gml_and_dependant_variables_names:
                        data_triples_ast['where'].append(BGP)
                        target_nodes_ast['where'].append(BGP)
        ###################### Sub Select ##########################
        if "SubSelect" in query.keys():
            sparqlml_ast['SubSelect']={}
            data_triples_ast['SubSelect'] = {}
            target_nodes_ast['SubSelect'] = {}
        ################### Limit and Offset ####################
        if "limit" in query.keys() and query["limit"] is not None:
            sparqlml_ast["limit"]=data_triples_ast["limit"]=target_nodes_ast["limit"] = query["limit"]
        if "offset" in query.keys() and query["offset"] is not None:
            sparqlml_ast["offset"] = data_triples_ast["offset"] = target_nodes_ast["offset"] = query["offset"]
        return sparqlml_ast,data_triples_ast,target_nodes_ast

    def ReWrite_SPARQL_ML_Query(self, query, gmlOperatorType_dict, userDefinedPredicate_dict, target_node_edge_dict, list_data_T, model_uri_dict,gml_and_dependant_variables, rewriteForm=1):
        '''
        rewrite SPARQL_ML query to generate  (candidateSparqlQuery,KG_DataQuery, and KG_kgTargetNodesQueries) and list mof model URIs
        '''
        string_q = ""
        string_q_dataonly=""
        string_q_target_nodes = ""
        string_gml = ""
        target_label_postfix = '_dic'

        if len(query['prefixes']) >0:
            for prefix,uri in query['prefixes'].items():
                s = f"PREFIX {prefix}: <{uri}>\n"
                # string_q.join(s)
                string_q+=s
                string_gml+=s
                string_q_dataonly+=s

        if self.KGMeta_Governer.RDFEngine==Constants.RDFEngine.stardog:
            string_q+="prefix kgnetML:<tag:stardog:api:kgnet:>\n"
        else:
            string_q += "prefix kgnetML:<sql:>\n"

        string_q_target_nodes=string_q_dataonly
        # string_q_dataonly +="SELECT ?s ?p ?o \n"+ " from <" + query['from'] + "> \n" if 'from' in query else ""
        # string_q_dataonly +="where { ?s ?p ?o { \n"
        target_variables_dict={}
        label_variables_dict={}
        for gml_op in gmlOperatorType_dict.keys():
            target_node = target_node_edge_dict[gml_op].split(':')[1] if ':' in target_node_edge_dict[gml_op] else target_node_edge_dict[gml_op]
            target_variable,label_variable=self.get_target_label_variables(query,userDefinedPredicate_dict[gml_op])
            target_variables_dict[str(gml_op)]=target_variable
            label_variables_dict[str(gml_op)]=label_variable
        ################### Query and one SubQuery level ##########################
        query_level=query
        userDefinedPredicate_lst = [str(elem) for elem in list(userDefinedPredicate_dict.values())]
        outer_string_q,outer_string_q_dataonly,outer_string_q_target_nodes=self.ReWrite_OuterQuery_Statments(query_level,userDefinedPredicate_lst, gmlOperatorType_dict, userDefinedPredicate_dict, target_variables_dict,label_variables_dict,list_data_T, model_uri_dict,target_label_postfix,gml_and_dependant_variables,rewriteForm)
        outer_ast, outer_dataonly_ast, outer_target_nodes_ast = self.ReWrite_OuterQuery_Statments_to_ast(query_level, userDefinedPredicate_lst, gmlOperatorType_dict, userDefinedPredicate_dict,target_variables_dict, label_variables_dict, list_data_T, model_uri_dict, target_label_postfix,gml_and_dependant_variables, rewriteForm)
        if "SubSelect" in query_level.keys():
            sub_string_q, sub_string_q_dataonly, sub_string_q_target_nodes = self.ReWrite_OuterQuery_Statments(query_level['SubSelect'],userDefinedPredicate_lst, gmlOperatorType_dict, userDefinedPredicate_dict, target_variables_dict,label_variables_dict,list_data_T, model_uri_dict,target_label_postfix,gml_and_dependant_variables,rewriteForm)
            sub_ast, sub_dataonly_ast, sub_target_nodes_ast = self.ReWrite_OuterQuery_Statments_to_ast(query_level['SubSelect'], userDefinedPredicate_lst, gmlOperatorType_dict, userDefinedPredicate_dict,target_variables_dict, label_variables_dict, list_data_T, model_uri_dict, target_label_postfix,gml_and_dependant_variables, rewriteForm)
            outer_string_q=outer_string_q.replace("$$SubSelect$$","\n{\n"+sub_string_q+"\n}\n")
            outer_string_q_dataonly="\n"+outer_string_q_dataonly.replace("$$SubSelect$$", "\n{\n"+sub_string_q_dataonly+"\n}\n")
            outer_string_q_target_nodes="\n"+outer_string_q_target_nodes.replace("$$SubSelect$$", "\n{\n"+sub_string_q_target_nodes+"\n}\n" )
        string_q+=outer_string_q
        string_q_dataonly+=outer_string_q_dataonly
        string_q_target_nodes+=outer_string_q_target_nodes
        ####################### create inference API-JSON object #####################
        API_JSON_dict={}
        for gml_op in model_uri_dict.keys():
            kg_df=self.KGMeta_Governer.getModelKGMetadata(mid=str(model_uri_dict[gml_op].split("/")[-1]))
            for col in kg_df.columns.tolist():
                kg_df[col]=kg_df[col].apply(lambda x:str(x)[1:-1] if str(x).startswith("<") or str(x).startswith("\"") else x)
            kg_df_p_list=kg_df["p"].tolist()
            API_JSON = "\"\"\"{\"model_id\" : \"" + str(model_uri_dict[gml_op].split("/")[-1]) + "\", "
            API_JSON += "\"RDFEngine\" : \"" + self.KGMeta_Governer.RDFEngine + "\", "
            if "kgnet:graph/namedGraphURI" in kg_df_p_list:
                API_JSON+= "\"named_graph_uri\" : \""+kg_df[kg_df["p"] == "kgnet:graph/namedGraphURI"]["o"].values[0].replace("\"","")+"\", "
            if "kgnet:graph/sparqlendpoint" in  kg_df_p_list:
                # API_JSON += "\"sparqlEndpointURL\" : \"" + kg_df[kg_df["p"] == "kgnet:graph/sparqlendpoint"]["o"].values[0].replace("\"","") + "\", "
                API_JSON += "\"sparqlEndpointURL\" : \"" + self.KGMeta_Governer.endpointUrl + "\", "
            for triple in query['triples'] :
                if triple['bgpType'] not in['filter'] and str(triple['subject'])==str(gml_op) and str(triple['predicate']).lower()=='kgnet:topk':
                    API_JSON += "\"topk\" : "+str(triple['object'])+", "
            API_JSON_dict[gml_op]=API_JSON

        string_q_target_nodes_dict={}
        string_q_dataonly_dict = {}
        for gml_op in API_JSON_dict.keys():
            API_JSON = API_JSON_dict[gml_op]
            target_variable=target_variables_dict[str(gml_op)]

            op_string_q_dataonly = string_q_dataonly.replace(set_syntax(target_variable), "?s")
            op_string_q_dataonly=op_string_q_dataonly.replace("\"", "'").replace("\n", " ")
            string_q_dataonly_dict[str(gml_op)]=op_string_q_dataonly
            API_JSON += "\"dataQuery\" : [\"" + op_string_q_dataonly + "\"] , "

            op_target_nodes_query= string_q_target_nodes.replace("\"", "'").replace("\n", " ").replace(set_syntax(target_variable), "?s")

            string_q_target_nodes_dict[str(gml_op)] =op_target_nodes_query
            API_JSON += "\"targetNodesQuery\" : \"" +op_target_nodes_query+ "\" "
            API_JSON+="}\"\"\""
            string_q=string_q.replace("?$API_JSON_"+str(gml_op)+"$",API_JSON)
            # string_q = string_q.replace(set_syntax(target_variable), "?s")

        return string_q ,string_q_dataonly,string_q_target_nodes_dict,[elem.split("/")[-1] for elem in model_uri_dict.values()]
    def ReWrite_SPARQL_ML_Query_to_ast(self, query, gmlOperatorType_dict, userDefinedPredicate_dict, target_node_edge_dict, list_data_T, model_uri_dict,gml_and_dependant_variables, rewriteForm=1):
        '''
        rewrite SPARQL_ML query to generate  (candidateSparqlQuery,KG_DataQuery, and KG_kgTargetNodesQueries) and list mof model URIs
        '''
        sparqlml_prefix = {}
        data_triples_prefix = {}
        target_nodes_prefix = {}

        string_q = ""
        string_q_dataonly=""
        string_q_target_nodes = ""
        string_gml = ""
        target_label_postfix = '_dic'

        if len(query['prefixes']) >0:
            sparqlml_prefix['prefix']=[]
            data_triples_prefix['prefix'] = []
            target_nodes_prefix['prefix'] = []
            for prefix,uri in query['prefixes'].items():
                s = f"{prefix}: <{uri}>"
                sparqlml_prefix['prefix'].append(s)
                data_triples_prefix['prefix'].append(s)
                target_nodes_prefix['prefix'].append(s)

        if self.KGMeta_Governer.RDFEngine==Constants.RDFEngine.stardog:
            sparqlml_prefix['prefix'].append("kgnetML:<tag:stardog:api:kgnet:>")
        else:
            sparqlml_prefix['prefix'].append("prefix kgnetML:<sql:>")

        target_variables_dict={}
        label_variables_dict={}
        for gml_op in gmlOperatorType_dict.keys():
            target_node = target_node_edge_dict[gml_op].split(':')[1] if ':' in target_node_edge_dict[gml_op] else target_node_edge_dict[gml_op]
            target_variable,label_variable=self.get_target_label_variables(query,userDefinedPredicate_dict[gml_op])
            target_variables_dict[str(gml_op)]=target_variable
            label_variables_dict[str(gml_op)]=label_variable
        ################### Query and one SubQuery level ##########################
        query_level=query
        userDefinedPredicate_lst = [str(elem) for elem in list(userDefinedPredicate_dict.values())]
        outer_ast, outer_dataonly_ast, outer_target_nodes_ast = self.ReWrite_OuterQuery_Statments_to_ast(query_level, userDefinedPredicate_lst, gmlOperatorType_dict, userDefinedPredicate_dict,target_variables_dict, label_variables_dict, list_data_T, model_uri_dict, target_label_postfix,gml_and_dependant_variables, rewriteForm)
        outer_ast['prefix']=sparqlml_prefix['prefix']
        outer_dataonly_ast['prefix'] = data_triples_prefix['prefix']
        outer_target_nodes_ast['prefix'] = target_nodes_prefix['prefix']
        if "SubSelect" in query_level.keys():
            sub_ast, sub_dataonly_ast, sub_target_nodes_ast = self.ReWrite_OuterQuery_Statments_to_ast(query_level['SubSelect'], userDefinedPredicate_lst, gmlOperatorType_dict, userDefinedPredicate_dict,target_variables_dict, label_variables_dict, list_data_T, model_uri_dict, target_label_postfix,gml_and_dependant_variables, rewriteForm)
            outer_ast['SubSelect'] = sub_ast
            outer_dataonly_ast['SubSelect'] =sub_dataonly_ast
            outer_target_nodes_ast['SubSelect'] = sub_target_nodes_ast
        ####################### create inference API-JSON object #####################
        API_JSON_dict={}
        for gml_op in model_uri_dict.keys():
            kg_df=self.KGMeta_Governer.getModelKGMetadata(mid=str(model_uri_dict[gml_op].split("/")[-1]))
            for col in kg_df.columns.tolist():
                kg_df[col]=kg_df[col].apply(lambda x:str(x)[1:-1] if str(x).startswith("<") or str(x).startswith("\"") else x)
            kg_df_p_list=kg_df["p"].tolist()
            API_JSON = "\"\"\"{\"model_id\" : \"" + str(model_uri_dict[gml_op].split("/")[-1]) + "\", "
            API_JSON += "\"RDFEngine\" : \"" + self.KGMeta_Governer.RDFEngine + "\", "
            if "kgnet:graph/namedGraphURI" in kg_df_p_list:
                API_JSON+= "\"named_graph_uri\" : \""+kg_df[kg_df["p"] == "kgnet:graph/namedGraphURI"]["o"].values[0].replace("\"","")+"\", "
            if "kgnet:graph/sparqlendpoint" in  kg_df_p_list:
                # API_JSON += "\"sparqlEndpointURL\" : \"" + kg_df[kg_df["p"] == "kgnet:graph/sparqlendpoint"]["o"].values[0].replace("\"","") + "\", "
                API_JSON += "\"sparqlEndpointURL\" : \"" + self.KGMeta_Governer.endpointUrl + "\", "
            for triple in query['triples'] :
                if str(triple['subject'])==str(gml_op) and str(triple['predicate']).lower()=='kgnet:topk':
                    API_JSON += "\"topk\" : "+str(triple['object'])+", "
            API_JSON_dict[gml_op]=API_JSON

        string_q_target_nodes_dict={}
        string_q_dataonly_dict = {}
        for gml_op in API_JSON_dict.keys():
            API_JSON = API_JSON_dict[gml_op]
            target_variable=target_variables_dict[str(gml_op)]
            op_string_q_dataonly = string_q_dataonly.replace(set_syntax(target_variable), "?s")
            op_string_q_dataonly=op_string_q_dataonly.replace("\"", "'").replace("\n", " ")
            string_q_dataonly_dict[str(gml_op)]=op_string_q_dataonly
            API_JSON += "\"dataQuery\" : [\"" + op_string_q_dataonly + "\"] , "
            op_target_nodes_query= string_q_target_nodes.replace("\"", "'").replace("\n", " ").replace(set_syntax(target_variable), "?s")
            string_q_target_nodes_dict[str(gml_op)] =op_target_nodes_query
            API_JSON += "\"targetNodesQuery\" : \"" +op_target_nodes_query+ "\" "
            API_JSON+="}\"\"\""
            string_q=string_q.replace("?$API_JSON_"+str(gml_op)+"$",API_JSON)
            # string_q = string_q.replace(set_syntax(target_variable), "?s")

        return string_q ,string_q_dataonly,string_q_target_nodes_dict,[elem.split("/")[-1] for elem in model_uri_dict.values()]
    def get_breakdown_queries(self, dict_gml_var, list_data_T,gml_and_dependant_variables):
        gmlOperatorType={}
        userDefinedPredicate={}
        target_node_edge={}
        model_uri_dict={}
        for gml_op in dict_gml_var:
            gmlOperatorType[gml_op] =  dict_gml_var[gml_op]['gml_operator_type']['object'] if not isinstance(dict_gml_var[gml_op]['gml_operator_type']['object'], (tuple,list)) else dict_gml_var[gml_op]['gml_operator_type']['object'][1]
            userDefinedPredicate[gml_op] = dict_gml_var[gml_op]['gml_operator_type']['subject'] if not isinstance(dict_gml_var[gml_op]['gml_operator_type']['object'], (tuple, list)) else  dict_gml_var[gml_op]['gml_operator_type']['object'][1]
            model_uri_dict[gml_op]=dict_gml_var[gml_op]['$m']
            if gmlOperatorType[gml_op].lower() == "kgnet:type/nodeclassification":
                target_label = dict_gml_var[gml_op]['label']['object'] if not isinstance(dict_gml_var[gml_op]['label']['object'],tuple) else dict_gml_var[gml_op]['label']['object'][1]
                target_label = target_label.split(':')[1] if ':' in target_label else target_label
                target_node_edge[gml_op] = dict_gml_var[gml_op]['target']['object'] if not isinstance(dict_gml_var[gml_op]['target']['object'],tuple) else  dict_gml_var[gml_op]['target']['object'][1]
                # return data_infer_query, data_query , target_nodes_query,model_id
            elif gmlOperatorType[gml_op].lower() == "kgnet:type/linkprediction":
                target_node_edge[gml_op] = dict_gml_var[gml_op]['targetEdge']['object'] if not isinstance(dict_gml_var[gml_op]['targetEdge']['object'],tuple) else  dict_gml_var[gml_op]['targetEdge']['object'][1]
                # data_infer_query,data_query,target_nodes_query,model_id = self.format_data_query(self.query_dict,gmlOperatorType, userDefinedPredicate,target_node_edge_dict=target_edge, list_data_T=list_data_T, model_uri_dict=dict_gml_var['$m'])
        data_infer_query,data_query,target_nodes_query,model_id = self.ReWrite_SPARQL_ML_Query(self.query_dict, gmlOperatorType, userDefinedPredicate, target_node_edge_dict=target_node_edge, list_data_T=list_data_T, model_uri_dict=model_uri_dict,gml_and_dependant_variables=gml_and_dependant_variables)
        return data_infer_query ,data_query,target_nodes_query,model_id

    def rewrite_gml_query (self):
        # print("gml_query_type=",self.query_dict["query_type"])
        return self.rewrite_gml_select_queries(self.query_dict)

if __name__ == '__main__':
    aifb_NC="""  
            prefix aifb:<http://swrc.ontoware.org/>
            prefix kgnet:<http://kgnet/>
            select ?person ?aff
            from <http://www.aifb.uni-karlsruhe.de>
            where {
            ?person a aifb:ontology#Person.
            ?person ?NodeClassifier ?aff.
            ?NodeClassifier a <kgnet:types/NodeClassifier>.
            ?NodeClassifier <kgnet:targetNode> <dblp:ontology#Person>.
            ?NodeClassifier <kgnet:labelNode> <dblp:ontology#ResearchGroup>.
            }
            limit 100 
            """

    dblp_LP="""
    prefix dblp:<https://dblp.org/>
    prefix kgnet:<https://kgnet/>
    select ?author ?affiliation
    where {
    ?author a dblp:Person.    
    ?author ?LinkPredictor ?affiliation.    
    ?LinkPredictor  a <kgnet:types/LinkPredictor>.
    ?LinkPredictor  <kgnet:targetEdge> "http://swrc.ontoware.org/ontology#publication".
    ?LinkPredictor <kgnet:GNNMethod> "MorsE" .
    }
    limit 10
    offset 0
    """
    dblp_NC = """
       prefix dblp:<https://dblp.org/>
       prefix kgnet: <https://kgnet/>
       select ?Publication ?title ?venue 
       where {
       ?Publication a dblp:Publication.
       ?Publication dblp:title ?title.
       ?Publication <https://dblp.org/rdf/schema#publishedIn> ?o.
       ?Publication ?NodeClassifier ?venue.
       ?NodeClassifier a <kgnet:types/NodeClassifier>.
       ?NodeClassifier <kgnet:targetNode> <dblp:Publication>.
       ?NodeClassifier <kgnet:labelNode> <dblp:venue>.
       }
       limit 100
       """

    kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.98.118:8890/sparql', KGMeta_URI="http://kgnet")
    gmlqp = gmlQueryParser(dblp_NC)
    (dataInferq,dataOnlyq,kmetaq)=gmlQueryRewriter(gmlqp.extractQueryStatmentsDict(),kgmeta_govener).rewrite_gml_query()
    # print(dataInferq)
    # print(dataOnlyq)
    # print(kmetaq)

    # insert_task_dict = gmlQueryParser(Insert_Query).extractQueryStatmentsDict()
    ieeecis_NC = """
    prefix ieeecis:<https://ieee-cis-fraud-detection/>
    select  ?Transaction ?is_fraud
    where
    {
    ?Transaction ieeecis:ProductCD ?prod.
    ?prod ?NodeClassifier ?is_fraud.
    ?NodeClassifier a <kgnet:types/NodeClassifier>.
    ?NodeClassifier <kgnet:GML/TargetNode> <ieeecis:Transaction>.
    ?NodeClassifier <kgnet:GML/NodeLabel>  <ieeecis:is_fraud>.
    }
    limit 100
    """
    mag_NC = """
    prefix mag:<http://mag.graph/>
    select  ?paper ?venue
    
    where
    {
    ?paper mag:has_venue ?o.
    ?paper ?NodeClassifier ?venue.
    ?NodeClassifier a <kgnet:types/NodeClassifier>.
    ?NodeClassifier <kgnet:GML/TargetNode> <mag:paper>.
    ?NodeClassifier <kgnet:GML/NodeLabel> <mag:venue> .
    }
    limit 100
    """
    YAGO_LP = """
    prefix yago3: <http://www.yago3-10/>
    prefix kgnet: <https://www.kgnet.com/>
    select ?airport1 ?airport2
    where { 
    ?airport1 yago3:isConnectedTo ?o.
    ?airport1 ?LinkPredictor ?airport2.
    ?LinkPredictor a <kgnet:types/LinkPredictor>.
    ?LinkPredictor <kgnet:GML/SourceNode> <yago3:airport1>.
    ?LinkPredictor <kgnet:GML/DestinationNode> <yago3:airport2>.
    ?LinkPredictor <kgnet:GML/EdgeType> <yago3:isConnectedTo> .
    ?LinkPredictor <kgnet:GML/TopK-Links> 10.}
    limit 10
    """

    double_nc = """
    prefix dblp:<https://dblp.org/rdf/schema#>
    prefix kgnet: <https://www.kgnet.ai/>
    select ?title ?venue ?discipline
    where {
    ?paper a dblp:Publication.
    ?paper dblp:title ?title.
    ?paper <https://dblp.org/rdf/schema#publishedIn> ?o.
    ?paper <https://dblp.org/has_gnn_model> 1.
    ?paper ?NodeClassifier ?venue.
    ?NodeClassifier a <kgnet:types/NodeClassifier>.
    ?NodeClassifier <kgnet:GML/TargetNode> <dblp:paper>.
    ?NodeClassifier <kgnet:GML/NodeLabel> <dblp:venue>.
    ?paper ?NodeClassifier ?discipline.
    ?NodeClassifier_dic a <kgnet:types/NodeClassifier>.
    ?NodeClassifier_dic <kgnet:GML/TargetNode> <dblp:paper>.
    ?NodeClassifier_dic <kgnet:GML/NodeLabel> <dblp:discipline>.
    }
    limit 100
    
    """

