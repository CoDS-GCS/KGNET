 # from rdflib import Graph
import rdflib
# import gzip
from rdflib.plugins.sparql.parser import parseQuery
# from rdflib.plugins.sparql import parser
""" Production """
import GMLQueryRewriter.KG_Meta as KG_Meta

""" DEBUG """
# import KG_Meta as KG_Meta

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


""" The extract() function takes the raw SPARQL query and prases it using the function provided by rdf
    and returns a dictionary containing different modules of the SPAEQL query"""
def extract (query):
    query = parseQuery(query)
    flag_prefix = False
    result = {}
    if len(query)>=2: # check if prefix exist in the query
        flag_prefix = True
        prefix_part= query[0]
        where_part = query[1]
        # print(where_part)
    else:
        where_part = query[0]
    
    if (flag_prefix): #store prefix in the dictionary
        dict_prefix = {}
        for p in prefix_part:
            dict_prefix[p['prefix']] = str(p['iri'])
        result['prefixes'] = dict_prefix
    
    select =[] 
    for s in where_part['projection']: # SELECT variables of the query
        # print(s)
        select.append(str(s['var']))
    result['select'] = select
    
    triples_list = []
    for t in where_part['where']['part'][0]['triples']: # iterating through the triples 
    # for t in where_part['where']['part']: # iterating through the triples 
        # t = t['triples'][0]
        triples = {}
        # subject = str(t[0])
        subject = t[0]
        if isinstance(t[1],rdflib.term.Variable):   # if predicate is a variable
            # predicate = str(t[1])
            predicate = t[1]
        elif isinstance(t[1]['part'][0]['part'][0]['part'],rdflib.term.URIRef): # else if predicate is a URI
            predicate = str(t[1]['part'][0]['part'][0]['part'])
        else:                                                                   # else it is a prefix:postfix pair
            p_prefix = str(t[1]['part'][0]['part'][0]['part'] ['prefix'])   
            p_lName = str(t[1]['part'][0]['part'][0]['part'] ['localname'])
            predicate = (p_prefix,p_lName)
            
        object_ = t[2]
        if not isinstance(object_,(rdflib.term.Variable,rdflib.term.URIRef,rdflib.term.Literal)):  # if object is not a URI or Variabel
            # object_prefix = str(object_['prefix'])
            object_prefix = object_['prefix']
            object_lName = object_['localname']
            object_ = (object_prefix,object_lName)
            
        triples['subject'] = subject
        triples['predicate']  = predicate
        # triples['object'] = str(object_)
        triples ['object'] =object_
        triples_list.append(triples)
    result['triples'] = triples_list
    result ['limit'] = where_part['limitoffset']['limit'] if 'limitoffset' in where_part.keys() else None
    return result
     


""" The set_syntax () function takes in a variable from the query and adjusts its syntax according 
    to its nature in the SPARQL query. This function is used in rebuilding the query according to the 
    syntax of the query."""
def set_syntax (var):
    # if isinstance(var,str) and var[:4].lower()=="http": # return with angular brackets <> if the var is a URI 
    if isinstance (var,str) and ('http' in var or ':' in var):
        return f'<{var}>'
    elif isinstance(var,rdflib.term.Literal):
        return str(var)
    return '?'+var if not isinstance(var,tuple) else f'{var[0]}:{var[1]}' # return with '?' apended to it if its a single term else return with prefix:postfix notation


""" The get_rdfType () function takes inputs a list of triples and a subject variable 
    and traverses through the triples to identiy the rdf type of the subject """
def get_rdfType (list_data_T,var):
    rdf_type = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'.lower()
    for t in list_data_T:
        if str(t['subject']).lower() == var.lower() and str(t['predicate']).lower() == rdf_type:
            return set_syntax(t['object'])


""" The construct_gml_q () function takes dictionary of variables required for the generation of GML query and returns the gml based query """
def construct_gml_q (dict_vars,data_vars):
    string_Q = f"""
    #SELECT {set_syntax(dict_vars['classifier']['subject'])}
    SELECT ?apiUrl
    WHERE
    """
    try :
        target_type = dict_vars['target']['object']
        target_type = set_syntax(get_rdfType(data_vars, target_type.split(':')[1]) if (isinstance(target_type,str) and get_rdfType(data_vars, target_type.split(':')[1]) is not None) else target_type)

        string_Q+="{"
        for key in dict_vars.keys():
            if len (dict_vars[key]) ==0:
                continue
            elif key=='target':
                string_temp = f"{set_syntax(dict_vars[key]['subject'])} {set_syntax(dict_vars[key]['predicate'])} {target_type} .\n"
                string_Q+=string_temp
                continue

            elif isinstance(dict_vars[key],list):
                for list_triple in dict_vars[key]:
                    string_temp=f"{set_syntax(list_triple['subject'])} {set_syntax(list_triple['predicate'])} {set_syntax(list_triple['object'])} .\n"
                    string_Q+=string_temp
                continue
            
            else:
                string_temp = f"{set_syntax(dict_vars[key]['subject'])} {set_syntax(dict_vars[key]['predicate'])} {set_syntax(dict_vars[key]['object'])} .\n"
                string_Q+=string_temp
     
        string_Q+=f"{set_syntax(dict_vars['classifier']['subject'])} <kgnet:term/uses> ?gmlModel .\n"
        string_Q+="""?gmlModel <kgnet:GML_ID> ?mID .
                     ?mID <kgnet:API_URL> ?apiUrl .   """
        string_Q +="}"
    except:
        raise Exception("GML specifications are incomplete in the query")
    
    return string_Q


"""The  prep_gml_vars () function takes input the gml dictionary and identifies the classification type,
    targets and labels from the query and returns a dictionary containing the aforementioned information 
    of the triples. """
def prep_gml_vars(gml_dict):

    dict_vars = {}
    # dict_vars[MISC]=[]
    """ Identify Classification Type """
    classification_type = None
    
    for triple in gml_dict:
        classification_type = [key for key in dict_classifTypes if isinstance(triple['object'],rdflib.term.URIRef) \
                               and triple['object'].lower()  in dict_classifTypes.get(key, [])][0]
        if classification_type is not None:
            dict_vars['classifier']= triple
            break
        
    if classification_type==NODECLASSIFIER:
        for triple in gml_dict:   
            if isinstance(triple['object'],rdflib.term.URIRef) and triple['object'].lower() in lis_classTypes:
                # dict_vars['classifier']= triple#['object'] 
                continue
                    
            if isinstance(triple['predicate'],str) and triple['predicate'].lower() in lis_targetTypes:
                dict_vars['target'] = triple#['object']         # if the triple is the target type
                continue
            
            elif isinstance(triple['predicate'],str) and triple ['predicate'].lower() in lis_labelTypes:
                dict_vars['label'] = triple#['object']          # if the triple is the label 
                continue 
            else:
                if MISC not in dict_vars.keys():
                    dict_vars[MISC] = []
                dict_vars[MISC].append(triple)
                
    
    elif classification_type == LINKPREDICTOR:
        for triple in gml_dict:
            # print(triple)
            if isinstance(triple['object'],rdflib.term.URIRef) and triple['object'].lower() in lis_classTypes:
                # dict_vars['classifier']= triple#['object'] 
                continue
            
            if isinstance(triple['predicate'],str) and triple['predicate'].lower() in lis_targetTypes:
                dict_vars['target'] = triple#['object']         # if the triple is the target type
                continue

            elif isinstance(triple['predicate'],str) and triple ['predicate'].lower() in lis_labelTypes:
                dict_vars['label'] = triple#['object']          # if the triple is the label 
                continue 

            else:
                if MISC not in dict_vars.keys():
                    dict_vars[MISC] = []
                dict_vars[MISC].append(triple)

    elif classification_type == GRAPHCLASSIFIER:
        raise NotImplementedError("Classification not yet supported")
        
    return dict_vars
    
def format_data_query(query,target_classifier,target_label,target_node,list_data_T,model_uri):
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
                string_q+= f'\n sql:getKeyValue_v2({set_syntax(target_node)},{set_syntax(target_label)}) as {set_syntax(item)} \n'
                continue
            s = f" ?{item},"
            string_q+=s

    string_q+= "\n WHERE { \n"
    for triple in list_data_T:
        string_t = ""


        s = triple['subject'] #if not isinstance(triple['subject'], tuple) else triple['subject'][1]
        p = triple['predicate'] #if not isinstance(triple['predicate'], tuple) else triple['predicate'][1]
        o = triple['object'] #if not isinstance(triple['object'], tuple) else triple['object'][1]
                   
        s = set_syntax(s)
        p = set_syntax(p)
        o = set_syntax(o)

        if p.replace('?','').lower() in set_syntax(target_classifier).lower():
            # target_type = get_rdfType(list_data_T, target_node)
            target_type = get_rdfType(list_data_T, target_node) if get_rdfType(list_data_T, target_node) is not None else target_node
            sub_query='{'
            sub_query+=f"SELECT sql:{clf_fxn[target_classifier.lower()]}('{model_uri}','{target_type}') \n as {set_syntax(target_label)}"
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
    
    
def construct_data_q(query,dict_gml_var,list_data_T):
    # string_q = ""
    # string_gml = ""
    target_label = dict_gml_var['label']['object'] if not isinstance(dict_gml_var['label']['object'], tuple) else dict_gml_var['label']['object'][1]
    target_label = target_label.split(':')[1] if ':' in target_label else target_label
    # target_label_mod = '_dic'
    
    target_node =  dict_gml_var['target']['object'] if not isinstance(dict_gml_var['target']['object'], tuple) else dict_gml_var['target']['object'][1]
    #target_node = target_node.split(':')[1] if ':' in target_node else target_node
    
    target_classifier =  dict_gml_var['classifier']['object'] if not isinstance(dict_gml_var['classifier']['object'], (tuple,list)) else dict_gml_var['classifier']['object'][1]
    target_classifier = target_classifier.split(':')[1] if ':' in target_classifier else target_classifier
    target_classifier = target_classifier.split('/')[1] if '/' in target_classifier else target_classifier
    
    if target_classifier.lower() == NODECLASSIFIER.lower():
        data_query = format_data_query(query, target_classifier, target_label, target_node, list_data_T, dict_gml_var['$m'])
        return data_query
    
    elif target_classifier.lower() == LINKPREDICTOR.lower():
        data_query = format_data_query(query, target_classifier, target_label, target_node, list_data_T, dict_gml_var['$m'])
        return data_query
    

    
def filter_triples (query):
    KGNET_LOOKUP = ['kgnet','<kgnet']
    data_triples = []
    gml_triples = []
    for t in query['triples']:
        values = t.values()
        flagT_gml= False
        # print(t,'\n')
        
        for v in values:

            # if isinstance(v,tuple) and v[0].lower() in KGNET_LOOKUP:
            if isinstance(v, str) and v.split(':')[0] in KGNET_LOOKUP:

                gml_triples.append(t)
                flagT_gml = True
                break
        if not flagT_gml:
            data_triples.append(t)
    return (data_triples,gml_triples)
    
def gen_queries(query,KEYWORD_KG = "kgnet"):
    string_q = ""
    string_gml = ""
    
    bool_gml = False
    if len(query['prefixes'])>0:
        for prefix,uri in query['prefixes'].items():
            if prefix.lower() == KEYWORD_KG:
                bool_gml = True
            s = f"PREFIX {prefix}: <{uri}>\n"
            # string_q.join(s)
            string_q+=s
            string_gml+=s
    if len(query['select'])>0:
        string_q+='SELECT'
        for item in query['select']:
            s = f" ?{item}"
            string_q+=s
    string_q+= "\n WHERE { \n"
    
    
    data_triples,gml_triples = filter_triples(query)

    dict_gml_var = prep_gml_vars(gml_triples)
    
    string_gml += construct_gml_q(dict_gml_var,data_triples)
    
    modelURI = KG_Meta.kgnet_getModelURI(string_gml)
    
    """DEBUG"""
    dict_gml_var['$m'] = modelURI #if modelURI is not None else 'http://127.0.0.1:64646/all'
    """ """    
    
    
    string_q = construct_data_q(query, dict_gml_var,data_triples)
    
    return (string_q,string_gml)



def execute (query):
    query_dict = extract(query)
    data_query,gml_query = gen_queries (query_dict)
    return data_query,gml_query


# group_query = """
# prefix dblp:<https://dblp.org/rdf/schema#>
# prefix kgnet: <https://www.kgnet.ai/>
# select ?title ?venue 
# where {
# ?paper a dblp:Publication.
# ?paper dblp:title ?title.
# ?paper <https://dblp.org/rdf/schema#publishedIn> ?o.


# ?paper ?NodeClassifier ?venue.

# ?NodeClassifier a <kgnet:types/NodeClassifier>.
# ?NodeClassifier <kgnet:GML/TargetNode> <dblp:paper>.
# ?NodeClassifier <kgnet:GML/NodeLabel> <dblp:venue>.}
# limit 10
# """
# test_query = """ 
# prefix dblp:<https://dblp.org/rdf/schema#>
# prefix kgnet: <https://www.kgnet.ai/>
# select ?title ?venue 
# where {
# ?paper a dblp:Publication.
# ?paper dblp:title ?title.
# ?paper <https://dblp.org/rdf/schema#publishedIn> ?o.


# ?paper ?NodeClassifier ?venue.

# ?NodeClassifier a <kgnet:types/NodeClassifier>.
# ?NodeClassifier <kgnet:GML/TargetNode> <dblp:paper>.
# ?NodeClassifier <kgnet:GML/NodeLabel> <dblp:venue>.}
# limit 10
#   """

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
}
limit 10
"""

dblp_NC= """
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
?NodeClassifier <kgnet:GML/NodeLabel> <dblp:venue>.

}
limit 100
"""

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
"""

# print("*"*20,"INPUT QUERY","*"*20)
# query_dict = extract(mag_NC) 
# output_2 = gen_queries(query_dict)
# print("*"*20,"DATA QUERY","*"*20)
# print(output_2[0])
# print("*"*20,"GML QUERY","*"*20)
# print(output_2[1])
# output_3 = prep_gml_vars (output_2)