import json
import rdflib
import re
from rdflib.plugins.sparql.parser import parseQuery as rdflibParseQuery
""" Production """
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
""" DEBUG """
# import KGMetaG as KGMetaG

# g = Graph()
# with open (r"C:\Users\walee\Desktop\RDF\dblp.rdf.gz" , 'rb') as f:
#   gzip_fd = gzip.GzipFile(fileobj=f)
#   g.parse(gzip_fd.read())

# g.parse("sampleFile.rdf")
# g.parse('D:/dataset/RDF/sparql.rdf')

# for item in g.query(query):
#     print(item)

# result = parseQuery(query)
# result = parseQuery(gml_query)

""" Prefix part """
# print(len(result)) # if len = 2 then there exist prefixes

# print(len(result[0])) # based on number of prefixes

# print (result [0][0]['prefix']) #prefix variable
# print (result [0][0]['iri']) #URI of the prefix

""" Select part """

# print (result [1].keys()) # projection & where

# for v in result[1]['projection']:  #for selecting variables
#     print (v['var'])

""" Triples Part """
# print (result[1]['where']['part'][0]['triples'][0][0]) # [0]-> nTriple ,[0]->subject
# print (result[1]['where']['part'][0]['triples'][0][1]['part'][0]['part'][0]['part']) #[0]->nTriple, [1]->predicate

# print (result[1]['where']['part'][0]['triples'][0][2]['prefix']) # object prefix
# print (result[1]['where']['part'][0]['triples'][0][2]['localname']) # object name

# for r in result[0]:
#     print(r)
#     print("\n")


############################################### Constants ###############################3
NODECLASSIFIER = 'nodeclassifier'
LINKPREDICTOR = 'linkpredictor'
GRAPHCLASSIFIER = 'graphclassifier'
MISC = 'MISC'
lis_classTypes = ['nodeclassifier','linkprediction','kgnet:types/nodeclassifier','kgnet:types/linkpredictor']
dict_classifTypes = {NODECLASSIFIER:['nodeclassifier','kgnet:types/nodeclassifier',],
                   LINKPREDICTOR:['linkprediction','linkpredictor','kgnet:types/linkpredictor'],
                   GRAPHCLASSIFIER:['graphclassifier']}

lis_targetTypes = ['classifiertarget','kgnet:gml/targetnode','sourcenode','kgnet:gml/sourcenode']
lis_labelTypes = ['classifierlabel','kgnet:gml/nodelabel','targetnode','kgnet:gml/targetnode','sourcenode','kgnet:gml/destinationnode']

clf_fxn = {#'nodeclassifier':'getNodeClass',
            'nodeclassifier':'getNodeClass_v2',
            'types/nodeclassifier':'getNodeClass_v2',
            'kgnet:types/NodeClassifier':'getNodeClass_v2'
          }

class gmlQueryParser:
    """
    This module will take a basic SPARQL query and return two queries:
        1. SPARQL GML Query
        2. SPARQL Data Query

    The flow of of this module is as following:
        -> extract ()
        -> gen_queries ()
        -> filter_triples ()
        -> prep_gml_vars ()
        -> construct_gml_q ()
        -> construct_data_q ()
    """

    def __init__(self,gmlquery):
       self.gmlquery=gmlquery
       self.query_statments ={}
    def parse_select(self):
        query = rdflibParseQuery(self.gmlquery)
        flag_prefix = False

        query_type = ""
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

        select = []
        for s in where_part['projection']:  # SELECT variables of the query
            # print(s)
            select.append(str(s['var']))
        self.query_statments['select'] = select
        triples_list = []
        for t in where_part['where']['part'][0]['triples']:  # iterating through the triples
            # for t in where_part['where']['part']: # iterating through the triples
            # t = t['triples'][0]
            triples = {}
            # subject = str(t[0])
            subject = t[0]
            if isinstance(t[1], rdflib.term.Variable):  # if predicate is a variable
                # predicate = str(t[1])
                predicate = t[1]
            elif isinstance(t[1]['part'][0]['part'][0]['part'], rdflib.term.URIRef):  # else if predicate is a URI
                predicate = str(t[1]['part'][0]['part'][0]['part'])
            else:  # else it is a prefix:postfix pair
                p_prefix = str(t[1]['part'][0]['part'][0]['part']['prefix'])
                p_lName = str(t[1]['part'][0]['part'][0]['part']['localname'])
                predicate = (p_prefix, p_lName)

            object_ = t[2]
            if not isinstance(object_, (
            rdflib.term.Variable, rdflib.term.URIRef, rdflib.term.Literal)):  # if object is not a URI or Variabel
                # object_prefix = str(object_['prefix'])
                object_prefix = object_['prefix']
                object_lName = object_['localname']
                object_ = (object_prefix, object_lName)

            triples['subject'] = subject
            triples['predicate'] = predicate
            # triples['object'] = str(object_)
            triples['object'] = object_
            triples_list.append(triples)
        self.query_statments['triples'] = triples_list
        self.query_statments['limit'] = where_part['limitoffset']['limit'] if 'limitoffset' in where_part.keys() and 'limit' in \
                                                                where_part['limitoffset'].keys() else None
        self.query_statments['offset'] = where_part['limitoffset']['offset'] if 'limitoffset' in where_part.keys() and 'offset' in \
                                                                  where_part['limitoffset'].keys() else None
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
                s = f" ?{item}"
                string_q += s
        string_q += "\n WHERE { \n"

        data_triples, gml_triples = self.filter_triples()
        dict_gml_var_dict,prep_gml_vars = self.prep_gml_vars(gml_triples)
        string_gml += self.get_KGMeta_gmlOperatorQuery(dict_gml_var_dict, data_triples)
        gml_query_res_df=self.kgmeta_govener.executeSparqlquery(string_gml)
        #dict_gml_var_dict['$m'] = gml_query_res_df[' modelURI if modelURI is not None else 'http://127.0.0.1:64646/all'
        dict_gml_var_dict['$m']=gml_query_res_df['apiUrl'].values[0].replace('"', "")
        string_q = self.get_data_query(dict_gml_var_dict, data_triples)
        return (string_q, string_gml)

    def filter_triples(self):
        KGNET_LOOKUP = ['kgnet', '<kgnet']
        data_triples = []
        gml_triples = []
        for t in self.query_dict['triples']:
            values = t.values()
            flagT_gml = False
            # print(t,'\n')
            for v in values:
                # if isinstance(v,tuple) and v[0].lower() in KGNET_LOOKUP:
                if isinstance(v, str) and v.split(':')[0] in KGNET_LOOKUP:
                    gml_triples.append(t)
                    flagT_gml = True
                    break
            if not flagT_gml:
                data_triples.append(t)
        return (data_triples, gml_triples)

    def prep_gml_vars(self, gml_dict):
        """The  prep_gml_vars () function takes input the gml dictionary and identifies the classification type,
                targets and labels from the query and returns a dictionary containing the aforementioned information
                of the triples. """

        dict_vars = {}
        # dict_vars[MISC]=[]
        """ Identify Classification Type """
        gml_operator_type = None
        for triple in gml_dict:
            gml_operator_type = [key for key in dict_classifTypes if isinstance(triple['object'], rdflib.term.URIRef) \
                                   and triple['object'].lower() in dict_classifTypes.get(key, [])][0]
            if gml_operator_type is not None:
                # dict_vars['gml_operator_type'] = triple
                break
        if gml_operator_type == NODECLASSIFIER:
            for triple in gml_dict:
                if isinstance(triple['object'], rdflib.term.URIRef) and triple['object'].lower() in lis_classTypes:
                    dict_vars['gml_operator']= triple#['object']
                    continue

                if isinstance(triple['predicate'], str) and triple['predicate'].lower() in lis_targetTypes:
                    dict_vars['target'] = triple  # ['object']         # if the triple is the target type
                    continue

                elif isinstance(triple['predicate'], str) and triple['predicate'].lower() in lis_labelTypes:
                    dict_vars['label'] = triple  # ['object']          # if the triple is the label
                    continue
                else:
                    if MISC not in dict_vars.keys():
                        dict_vars[MISC] = []
                    dict_vars[MISC].append(triple)


        elif gml_operator_type == LINKPREDICTOR:
            for triple in gml_dict:
                # print(triple)
                if isinstance(triple['object'], rdflib.term.URIRef) and triple['object'].lower() in lis_classTypes:
                    dict_vars['gml_operator']= triple#['object']
                    continue

                if isinstance(triple['predicate'], str) and triple['predicate'].lower() in lis_targetTypes:
                    dict_vars['target'] = triple  # ['object']         # if the triple is the target type
                    continue

                elif isinstance(triple['predicate'], str) and triple['predicate'].lower() in lis_labelTypes:
                    dict_vars['label'] = triple  # ['object']          # if the triple is the label
                    continue

                else:
                    if MISC not in dict_vars.keys():
                        dict_vars[MISC] = []
                    dict_vars[MISC].append(triple)

        elif gml_operator_type == GRAPHCLASSIFIER:
            raise NotImplementedError("Graph Classification not yet supported")

        return dict_vars,gml_operator_type

    def set_syntax (self,var):
        """ The set_syntax () function takes in a variable from the query and adjusts its syntax according
                to its nature in the SPARQL query. This function is used in rebuilding the query according to the
                syntax of the query."""

        # if isinstance(var,str) and var[:4].lower()=="http": # return with angular brackets <> if the var is a URI
        if isinstance (var,str) and ('http' in var or ':' in var):
            return f'<{var}>'
        elif isinstance(var,rdflib.term.Literal):
            return str(var)
        return '?'+var if not isinstance(var,tuple) else f'{var[0]}:{var[1]}' # return with '?' apended to it if its a single term else return with prefix:postfix notation

    def get_rdfType (self,list_data_T,var,rdf_type='http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
        """ The get_rdfType () function takes inputs a list of triples and a subject variable
                and traverses through the triples to identiy the rdf type of the subject """
        rdf_type = rdf_type.lower()
        for t in list_data_T:
            if str(t['subject']).lower() == var.lower() and str(t['predicate']).lower() == rdf_type:
                return self.set_syntax(t['object'])

    def get_KGMeta_gmlOperatorQuery (self,dict_vars,data_vars):
        """ The function takes dictionary of variables required for the generation of GML operator query and returns the the query """
        # SELECT {self.set_syntax(dict_vars['gml_operator']['subject'])}
        string_Q = f"""
        SELECT distinct ?apiUrl ?gmlModel ?mID
        WHERE
        """
        try :
            target_type = dict_vars['target']['object']
            target_type = self.set_syntax(self.get_rdfType(data_vars, target_type.split(':')[1]) if (isinstance(target_type,str) and self.get_rdfType(data_vars, target_type.split(':')[1]) is not None) else target_type)

            string_Q+="{"
            for key in dict_vars.keys():
                if len (dict_vars[key]) ==0:
                    continue
                elif key=='target':
                    string_temp = f"{self.set_syntax(dict_vars[key]['subject'])} {self.set_syntax(dict_vars[key]['predicate'])} {self.set_syntax(dict_vars[key]['object'])} .\n"
                    string_Q+=string_temp
                    continue

                elif isinstance(dict_vars[key],list):
                    for list_triple in dict_vars[key]:
                        string_temp=f"{self.set_syntax(list_triple['subject'])} {self.set_syntax(list_triple['predicate'])} {self.set_syntax(list_triple['object'])} .\n"
                        string_Q+=string_temp
                    continue

                else:
                    string_temp = f"{self.set_syntax(dict_vars[key]['subject'])} {self.set_syntax(dict_vars[key]['predicate'])} {self.set_syntax(dict_vars[key]['object'])} .\n"
                    string_Q+=string_temp

            string_Q+=f"{self.set_syntax(dict_vars['gml_operator']['subject'])} <kgnet:term/uses> ?gmlModel .\n"
            string_Q+="""?gmlModel <kgnet:GML_ID> ?mID .
                         ?mID <kgnet:API_URL> ?apiUrl .   """
            string_Q +="}"
        except Exception as e :
            print(e)
            raise Exception("GML specifications are incomplete in the query")

        return string_Q

    def format_data_query(self,query,target_classifier,target_label,target_node,list_data_T,model_uri):
        string_q = ""
        string_gml = ""
        # target_label = dict_gml_var['label']['object'] if not isinstance(dict_gml_var['label']['object'], tuple) else dict_gml_var['label']['object'][1]
        # target_label = target_label.split(':')[1] if ':' in target_label else target_label
        target_label_mod = '_dic'


        # target_node =  dict_gml_var['target']['object'] if not isinstance(dict_gml_var['target']['object'], tuple) else dict_gml_var['target']['object'][1]
        target_node = target_node.split(':')[1] if ':' in target_node else target_node

        # target_classifier =  dict_gml_var['classifier']['object'] if not isinstance(dict_gml_var['classifier']['object'], (tuple,list)) else dict_gml_var['classifier']['object'][1]
        # target_classifier = target_classifier.split(':')[1] if ':' in target_classifier else target_classifier

        clf_fxn = {#'nodeclassifier':'getNodeClass',
                   'nodeclassifier':'getNodeClass_v2',
                   'types/nodeclassifier':'getNodeClass_v2',
                   'kgnet:types/NodeClassifier':'getNodeClass_v2',
                   'kgnet:types/LinkPredictor':'getLinkPred',
                   'kgnet:types/linkpredictor':'getLinkPred',
                   'linkpredictor':'getLinkPred'
                        }

        if len(query['prefixes']) >0:
            for prefix,uri in query['prefixes'].items():
                s = f"PREFIX {prefix}: <{uri}>\n"
                # string_q.join(s)
                string_q+=s
                string_gml+=s
        if len(query['select'])>0:
            string_q+='SELECT'
            for item in query['select']:
                if item.lower() in target_label.lower():
                    target_label += target_label_mod
                    string_q+= f'\n sql:getKeyValue_v2({self.set_syntax(target_node)},{self.set_syntax(target_label)}) as {self.set_syntax(item)} \n'
                    continue
                s = f" ?{item},"
                string_q+=s

        string_q+= "\n WHERE { \n"
        for triple in list_data_T:
            string_t = ""


            s = triple['subject'] #if not isinstance(triple['subject'], tuple) else triple['subject'][1]
            p = triple['predicate'] #if not isinstance(triple['predicate'], tuple) else triple['predicate'][1]
            o = triple['object'] #if not isinstance(triple['object'], tuple) else triple['object'][1]

            s = self.set_syntax(s)
            p = self.set_syntax(p)
            o = self.set_syntax(o)

            if p.replace('?','').lower() in self.set_syntax(target_classifier).lower():
                # target_type = get_rdfType(list_data_T, target_node)
                target_type = self.get_rdfType(list_data_T, target_node) if self.get_rdfType(list_data_T, target_node) is not None else target_node
                sub_query='{'
                sub_query+=f"SELECT sql:{clf_fxn[target_classifier.lower()]}('{model_uri}','{target_type}') \n as {self.set_syntax(target_label)}"
                sub_query+=' WHERE {} '
                string_q+=sub_query
                continue
            # s=set_syntax(s)
            # p=set_syntax(p)
            string_t += f"{s} {p} {o} .\n"
            string_q +=string_t

        string_q+="}" # for SELECT
        string_q+="}" # for WHERE
        string_q+= f'\n LIMIT {query["limit"]}' if query["limit"] is not None else ""
        return string_q

    def get_data_query(self,dict_gml_var,list_data_T):
        # string_q = ""
        # string_gml = ""
        target_label = dict_gml_var['label']['object'] if not isinstance(dict_gml_var['label']['object'], tuple) else dict_gml_var['label']['object'][1]
        target_label = target_label.split(':')[1] if ':' in target_label else target_label
        # target_label_mod = '_dic'

        target_node =  dict_gml_var['target']['object'] if not isinstance(dict_gml_var['target']['object'], tuple) else dict_gml_var['target']['object'][1]
        #target_node = target_node.split(':')[1] if ':' in target_node else target_node

        target_classifier =  dict_gml_var['gml_operator']['object'] if not isinstance(dict_gml_var['gml_operator']['object'], (tuple,list)) else dict_gml_var['gml_operator']['object'][1]
        target_classifier = target_classifier.split(':')[1] if ':' in target_classifier else target_classifier
        target_classifier = target_classifier.split('/')[1] if '/' in target_classifier else target_classifier

        if target_classifier.lower() == NODECLASSIFIER.lower():
            data_query = self.format_data_query(self.query_dict, target_classifier, target_label, target_node, list_data_T, dict_gml_var['$m'])
            return data_query

        elif target_classifier.lower() == LINKPREDICTOR.lower():
            data_query = self.format_data_query(self.query_dict, target_classifier, target_label, target_node, list_data_T, dict_gml_var['$m'])
            return data_query

    def rewrite_gml_query (self):
        print("gml_query_type=",self.query_dict["query_type"])
        return self.rewrite_gml_select_queries(self.query_dict)

if __name__ == '__main__':
    dblp_LP="""
    prefix dblp:<https://dblp.org/rdf/schema#>
    prefix kgnet: <https://www.kgnet.ai/>
    select ?author ?affiliation
    where {
    ?author a dblp:Person.    
    ?author ?LinkPredictor ?affiliation.    
    ?LinkPredictor  a <kgnet:types/LinkPredictor>.
    ?LinkPredictor  <kgnet:GML/SourceNode> <dblp:author>.
    ?LinkPredictor  <kgnet:GML/DestinationNode> <dblp:Affiliation>.
    ?LinkPredictor <kgnet:term/uses> ?gmlModel .
    ?gmlModel <kgnet:GML_ID> ?mID .
    ?mID <kgnet:API_URL> ?apiUrl.
    ?mID <kgnet:GMLMethod> <kgnet:GML/Method/MorsE>.
    }
    limit 10
    offset 0
    """
    dblp_NC = """
       prefix dblp:<https://dblp.org/rdf/schema#>
       prefix kgnet: <https://www.kgnet.ai/>
       select ?paper ?title ?venue 
       where {
       ?paper a dblp:Publication.
       ?paper dblp:title ?title.
       ?paper <https://dblp.org/rdf/schema#publishedIn> ?o.
       ?paper <https://dblp.org/has_gnn_model> 1.
       ?paper ?NodeClassifier ?venue.
       ?NodeClassifier a <kgnet:types/NodeClassifier>.
       ?NodeClassifier <kgnet:GML/TargetNode> <dblp:Publication>.
       ?NodeClassifier <kgnet:GML/NodeLabel> <dblp:venue>.
       }
       limit 100
       """

    Insert_Query = """
        prefix dblp:<https://www.dblp.org/>
        prefix kgnet:<https://www.kgnet.com/>
        Insert into <kgnet>
        where{
            select * from kgnet.TrainGML(
            {
            "Name":"MAG_Paper-Venue_Classifer",
            "GML-Task":{
                "TaskType":"kgnet:NodeClassifier",
                "TargetNode":"dblp:Publication",
                "NodeLabel":"dblp:venue",
                "GNN_Method":"G_SAINT"
                },
            "TaskBudget":{
                "MaxMemory":"50GB",
                "MaxTime":"1h",
                "Priority":"ModelScore"}
            })}"""
    # Insert_Query="""prefix dblp:<https://www.dblp.org/>
    #     prefix kgnet:<https://www.kgnet.com/>
    #     #Insert {GRAPH <kgnet> {?s ?p ?o}}
    #     select *
    #     where
    #     {
    #         GRAPH <http://dblp.org>
    #         {
    #             SELECT ?s ?p ?o
    #             { ?s ?p ?o.}
    #             limit 1
    #         }
    #     }
    #     """
    kgmeta_govener = KGMeta_Governer(endpointUrl='http://206.12.98.118:8890/sparql', KGMeta_URI="http://kgnet")

    # gmlqp = gmlQueryParser(dblp_NC)
    # (dataq,kmetaq)=gmlQueryRewriter(gmlqp.extractQueryStatmentsDict(),kgmeta_govener).rewrite_gml_query()

    insert_task_dict = gmlQueryParser(Insert_Query).extractQueryStatmentsDict()
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
