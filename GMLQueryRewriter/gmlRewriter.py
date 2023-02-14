from rdflib import Graph
import rdflib
import gzip
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql import parser
g = Graph()
# with open (r"C:\Users\walee\Desktop\RDF\dblp.rdf.gz" , 'rb') as f:
#   gzip_fd = gzip.GzipFile(fileobj=f)
#   g.parse(gzip_fd.read())

g.parse("sampleFile.rdf")
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
    -> const_gml_q ()
    -> const_data_q ()
"""


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
    else:
        where_part = query[0]
    
    if (flag_prefix): #store prefix in the dictionary
        dict_prefix = {}
        for p in prefix_part:
            dict_prefix[p['prefix']] = str(p['iri'])
        result['prefixes'] = dict_prefix
    
    select =[] 
    for s in where_part['projection']: # SELECT variables of the query
        select.append(str(s['var']))
    result['select'] = select
    
    triples_list = []
    for t in where_part['where']['part'][0]['triples']: # iterating through the triples 
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
        if not isinstance(object_,(rdflib.term.Variable,rdflib.term.URIRef)):  # if object is not a URI or Variabel
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
    return result
     


query_1 = "SELECT ?x WHERE { ?x a <http://example.org/Person> }"

           
query_2 = """select ?s ?p ?o
where  {
 ?s ?p ?o.
}
"""

query_3 = """
prefix dblp: <https://www.dblp.org/>
prefix kgnet: <https://www.kgnet.com/>
select ?title ?venue 
where {
?paper a dblp:Publication.
?paper dblp:title ?title.
?paper ?NodeClassifier ?venue.
?NodeClassifier a kgnet:NodeClassifier.
?NodeClassifier kgnet:classifierTarget dblp:paper.
?NodeClassifier kgnet:classifierLabel dblp:venue.}
"""      



query_4 = """
prefix dblp: <https://www.dblp.org/>
prefix kgnet: <https://www.kgnet.com/>
select ?title ?cite 
where {
?paper a dblp:Publication.
?paper dblp:title ?title.
?paper ?LinkPrediction  ?cite.
?LinkPrediction a kgnet:LinkPrediction.
?LinkPrediction kgnet:classifierTarget dblp:cite.
?LinkPrediction kgnet:classifierLabel dblp:paper.}
"""   


output = extract(query_3)

# output =parseQuery(query_3)
# print(output)

""" The set_syntax () function takes in a variable from the query and adjusts its syntax according 
    to its nature in the SPARQL query. This function is used in rebuilding the query according to the 
    syntax of the query."""
def set_syntax (var):
    if isinstance(var,str) and var[:4].lower()=="http": # return with angular brackets <> if the var is a URI 
        return f'<{var}>'
    return '?'+var if not isinstance(var,tuple) else f'<{var[0]}:{var[1]}>' # return with '?' apended to it if its a single term else return with prefix:postfix notation


""" The const_gml_q () function takes dictionary of variables and returns the gml based query """
def const_gml_q (dict_vars):
    string_Q = f"""
    SELECT {set_syntax(dict_vars['classifier']['subject'])}
    WHERE
    """
    try :
        string_Q+="{"
        string_Q+=f"""
        {set_syntax(dict_vars['classifier']['subject'])} {set_syntax(dict_vars['classifier']['predicate'])} {set_syntax(dict_vars['classifier']['object'])} .
        {set_syntax(dict_vars['target']['subject'])} {set_syntax(dict_vars['target']['predicate'])} {set_syntax(dict_vars['target']['object'])} .
        {set_syntax(dict_vars['label']['subject'])} {set_syntax(dict_vars['label']['predicate'])} {set_syntax(dict_vars['label']['object'])} .
        """
        string_Q +="}"
    except:
        print("GML specifications not complete")
    
    return string_Q

""" The get_rdfType () function takes inputs a list of triples and a subject variable 
    and traverses through the triples to identiy the rdf type of the subject """
def get_rdfType (list_data_T,var):
    rdf_type = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'.lower()
    for t in list_data_T:
        if t['subject'].lower() == var.lower() and t['predicate'].lower() == rdf_type:
            return set_syntax(t['object'])


"""The  prep_gml_vars () function takes input the gml dictionary and identifies the classification type,
    targets and labels from the query and returns a dictionary containing the aforementioned information 
    of the triples. """
def prep_gml_vars(gml_dict):
    lis_classTypes = ['nodeclassifier','linkprediction',]
    lis_targetTypes = ['classifiertarget']
    lis_labelTypes = ['classifierlabel']
    dict_vars = {}
    for triple in gml_dict: # traverse the list and identify the label types
        if isinstance(triple['object'],tuple) and triple['object'][1].lower() in lis_classTypes:
            dict_vars['classifier']= triple#['object']      # if the triple is the classifier type
            continue
        
        if isinstance(triple['predicate'],tuple) and triple ['predicate'][1].lower() in lis_targetTypes:
            dict_vars['target'] = triple#['object']         # if the triple is the target type
            continue
        
        if isinstance(triple['predicate'],tuple) and triple ['predicate'][1].lower() in lis_labelTypes:
            dict_vars['label'] = triple#['object']          # if the triple is the label 
            continue  
    return dict_vars

def const_data_q(query,dict_gml_var,list_data_T):
    string_q = ""
    string_gml = ""
    target_label = dict_gml_var['label']['object'] if not isinstance(dict_gml_var['label']['object'], tuple) else dict_gml_var['label']['object'][1]
    target_label_mod = '_dic'
    target_node =  dict_gml_var['target']['object'] if not isinstance(dict_gml_var['target']['object'], tuple) else dict_gml_var['target']['object'][1]
    target_classifier =  dict_gml_var['classifier']['object'] if not isinstance(dict_gml_var['classifier']['object'], tuple) else dict_gml_var['classifier']['object'][1]
    clf_fxn = {'nodeclassifier':'getNodeClass',
                    }
    
    
    bool_var_label = False
    if len(query['prefixes']) >0:
        for prefix,uri in query['prefixes'].items():
            s = f"PREFIX {prefix}: <{uri}>\n"
            # string_q.join(s)
            string_q+=s
            string_gml+=s
    if len(query['select'])>0:
        string_q+='SELECT'
        for item in query['select']:
            if item.lower() == target_label:
                target_label += target_label_mod
                string_q+= f'\n sql:UDFS.getKeyValue({set_syntax(target_label)} , {set_syntax(target_node)}) as {set_syntax(item)} \n'
                continue
            s = f" ?{item}"
            string_q+=s
    
    string_q+= "\n WHERE { \n"
    for triple in list_data_T:
        string_t = ""

        s = ""
        p = ""
        o = ""
        subject = triple['subject']
        predicate = triple['predicate']
        object_ = triple['object']
                   
        s+= set_syntax(subject)

        p+= set_syntax(predicate)

        if p.lower() == set_syntax(target_classifier.lower()):
            target_type = get_rdfType(list_data_T, target_node)
            sub_query='{'
            sub_query+=f'SELECT sql:UDFS.{clf_fxn[target_classifier.lower()]}($m,{target_type}) \n as {set_syntax(target_label)}'
            sub_query+=' WHERE {} '
            string_q+=sub_query
            continue                   
        
        o+=set_syntax(object_)
        string_t += f"{s} {p} {o} .\n"
        string_q +=string_t
    
    string_q+="}"
    
    return string_q
    
def filter_triples (query):
    data_triples = []
    gml_triples = []
    for t in query['triples']:
        values = t.values()
        # if bool_gml:
        flagT_gml= False
        for v in values:
            if isinstance(v,tuple) and v[0].lower()=="kgnet":
                # print(v)
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
    string_gml = const_gml_q(dict_gml_var)
    
    string_q = const_data_q(query, dict_gml_var,data_triples)
    return (string_q,string_gml)



        
# print(gen_queries(output))
output_2 = gen_queries(output)
print("*"*20,"DATA QUERY","*"*20)
print(output_2[0])
print("*"*20,"GML QUERY","*"*20)
print(output_2[1])
# output_3 = prep_gml_vars (output_2)