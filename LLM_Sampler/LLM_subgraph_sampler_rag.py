import requests
import pandas as pd
from io import StringIO

#rquirements
# !pip install llama-index
# pip install llama-index-llms-opena

"""#SPARQL Query"""

def executeSparqlQuery(query,SPARQLendpointUrl,firstRowHeader=True):
    """
    Execute sparql query through dopost and return results in form of datafarme.
    :param query:the sparql query string.
    :type query: str
    :param firstRowHeader: wherther to assume frist line as the dataframe columns header.

    """
    body = {'query': query}
    headers = {
        # 'Content-Type': 'application/sparql-update',
        'Content-Type': "application/x-www-form-urlencoded",
        'Accept-Encoding': 'gzip',
        'Accept':  ('text/tab-separated-values; charset=UTF-8')
        # 'Accept': 'text/tsv'
    }
    r = requests.post(SPARQLendpointUrl, data=body, headers=headers)
    if firstRowHeader:
        res_df=pd.DataFrame([x.split('\t') for x in r.text.split('\n')[1:] if x],columns=r.text.split('\n')[0].replace("\"","").replace("?","").split('\t'))
    else:
        res_df=pd.DataFrame([x.split('\t') for x in r.text.split('\n')])
    for col in res_df.columns:
      res_df[col] = res_df[col].str.strip('"')
    return res_df

SPARQLendpointUrl = "http://206.12.98.118:8890/sparql/"
NamedGraph_URI="https://dblp2022.org"
# KG_Schema_df=getKGNodeEdgeTypes(NamedGraph_URI,SPARQLendpointUrl)

"""# Install OpenAI Libraries & API"""
API_KEY=""
import openai
import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]
llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.llm = llm
Settings.chunk_size = 512
############### Supported LLMs ################33
# model_name="gpt-4o-mini"
# model_name="deepseek-r1"
# model_name="deepseek-reasoner"
# model_name="deepseek-chat"
model_name="gemini-1.5-flash"

import requests
import json
def query_ollama(model, prompt):
  # url = "http://192.168.41.218:11434/api/generate"
  url = "http://206.12.96.43:11434/api/generate"
  headers = {"Content-Type": "application/json"}
  data = {
      "model": model,
      "prompt": prompt,
      "stream": False
  }
  response = requests.post(url, headers=headers, data=json.dumps(data))
  if response.status_code == 200:
          return response.json()
  else:
      return {"error": f"Request failed with status code {response.status_code}"}

def chat(model="o1-mini",prompt_in=""):
  if model in ["o1-mini","gpt-4o-mini"]:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
    model,
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": prompt_in
                  },
              ],
          }
      ]
  )
    return response.choices[0].message.content,response.usage,response
  elif "deepseek-r1" in model:
    response = query_ollama("deepseek-r1", prompt_in)
    # print(response['response'].split("</think>")[-1].strip())
    # print(response)
    usage_keys=['total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']
    return response['response'].split("</think>")[-1].strip(),{key:response[key] for key in usage_keys},response
  elif model=="deepseek-chat":
    from openai import OpenAI
    llm = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a knoweldge reasoner system"},
            {"role": "user", "content":prompt_in },
        ],
        stream=False
    )
    return response.choices[0].message.content,response.usage,response
  elif model=="deepseek-reasoner":
    from openai import OpenAI
    llm = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    response = llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a knoweldge reasoner system"},
            {"role": "user", "content":prompt_in },
        ],
        stream=False
    )
    return response.choices[0].message.content,response.usage,response
  elif model in ["gemini-1.5-flash","gemini-2.5-flash"]:
      import google.generativeai as genai
      from google.colab import userdata
      GEMINI_API_KEY=API_KEY
      genai.configure(api_key=GEMINI_API_KEY)
      model = genai.GenerativeModel(model)
      response = model.generate_content(prompt_in)
      return response.text,response.usage_metadata,response

"""# Subgraph Sampling: Prompts To SPARQL"""

def suggest_features_prompt(task="classify a publication into a certian publication venue",model_name=model_name,topK=25):
  message=f""""You are an expert in machine learning feature selection specifically for classification tasks.
  Think about information required to accurately {task}.
  return up to {topK} features.
  return the a numbered key list of items without explination
  sort the list according to item imprtance
  """
  suggested_features,usage,full_response=chat(model=model_name,prompt_in=message)
  return suggested_features,usage,full_response,message
  # suggested_features.split('\n')

def match_features_prompt(schema,suggested_features,KG="DBLP",model_name=model_name,topk=25):
  message=f""""You are an expert in machine learning feature selection for graph neural networks, specifically for graph node classification inference tasks.
  The following describes the {KG} knowledge graph schema, detailing the relationships between graph entities in a series of triples, one triple per line.
  ---------- {KG} Schema ----
  {schema}
  ---------------------------
  Given the folwing list of key features select the matching relations from the previouse schema.
  --------------- Keys Features ----------------
  {suggested_features}
  ---------------------------------------------
  1- Think carefully and refine your selected/matching items
  2- Give high periorty to directly connected nodes than fare away connected nodes
  2- Return the top {topk} matched schema triples sorted by the importance
  Output only one selected triple per line without any explination.
  """
  # Output only one selected triple per line.
  # output each choosen triple and the proper explination why did you choose it.
  BGP_lst,usage,full_response=chat(model=model_name,prompt_in=message)
  return BGP_lst,usage,full_response,message

def generate_sparql_query(BGP_lst,KG="DBLP",VT="Publication",
                          SPARQL_Example=f""" PREFIX dblp: <https://dblp.org/rdf/schema#>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
  select ?s ?p ?o
  from <https://dblp2022.org>
  where
  {{
    {{select ?s ?p ?o where {{?s a dblp:Publication. ?s dblp:title  ?o. BIND('dblp:title' AS ?p). }}}}
    {{select ?s ?p ?o where {{?s a dblp:Publication. ?s rdfs:label  ?o. BIND('rdfs:label' AS ?p). }}}}
  }}""",model_name=model_name):
  sparql_prompt=f"""-You are an expert SPARQL query writer.
  - Given the following triples list from the {KG} knoweldge graph schema , Write SPARQL query to selects the {VT} and its associated information given int the following triples list.
  - the triples are directed, make sure to fullfill the direction and relation type.
  - the query must return the union of sub select statment in the form ?s ?p ?o.
  - Each triple is Subject Entity - relation - Object Entity
  - Start with the {VT} node.
  -------- BGP List ----------------
  {BGP_lst.replace(",","-")}
  ----------------- SPARQL query example ---------------
  {SPARQL_Example}
  -----------------
  1 write nested select subqueries and Union them
  2 in single-hop nested select , make sure to start the first BGP with the variable ?s.
  4 in tow-hop or more nested select:
      4.1 start the frist BGP with the variable ?s then use other variable names for next BGPs.
      4.2 used the last connected entity as subject as shown in the perviouse example
  5 Generate only the SPARQL query without any explination.
  6 make sure to use each given BGP triple
  7 add the BGP:  'Values ?s {{<VT-List>}}.' to the end of each subquery
  8 Refine all rules  and the query sysntax.
  10 Do invent new relations i.e, dblp:authoredBy can not be dblp:Authored , But you can start with ?o instead of ?s.
    Example:
      ?s a dblp:Publication.
      ?s dblp:authoredBy ?o.
      --------- Can Be ---------
      ?o a dblp:author.
      ?s dblp:authoredBy ?o.
  """
  sparql_v0,usage,full_response=chat(model=model_name,prompt_in=sparql_prompt)
  return sparql_v0,usage,full_response,sparql_prompt

# - Rule2: in nested select queries, use the last  BGP relation name as predicate variable  ,i.e, SELECT ?s ?p ?o -> SELECT ?s 'dblp:title' as ?p ?o.
def  refine_sparql_query(sparql_query,model_name=model_name):
  sparql_refine_prompt=f"""-You are an expert SPARQL query writer.
                  Given the following SPARQL query , ReWrite it to follow the following rules.
                  - Rule1: Keep the nested selects and thier Union statments
                  - Rule2: restructure the n-hop sub-select to choose lastest BGP subject and object and as the select items
                    Example:  {{
                                ?s a prefix:x.
                                ?s prefix:y ?y.
                                ?y prefix:z ?z.
                              }}
                      the lastest BGP is ?y prefix:z ?z , then the select items must be:
                      1- ?y as ?s
                      2- ?p
                      3- ?z as ?o

  -------- SPARQL Query ----------------
  {sparql_query}

  --Refine The Rules and Examples Carefully
  -- Return only the Query, do not return any explaination
  <Answer>"""
  # print(sparql_refine_prompt)
  sparql_v1,usage,full_response=chat(model=model_name,prompt_in=sparql_refine_prompt)
  return sparql_v1,usage,full_response,sparql_refine_prompt

def validate_BGP_Schema(BGP_lst_str,KG_Schema_str):
  BGP_lst_str=BGP_lst_str.lower()
  BGP_lst_df=pd.read_csv(StringIO(BGP_lst_str), sep=',',names=["stype","rel","otype"])
  KG_Schema_str=KG_Schema_str.lower()
  KG_Schema_df=pd.read_csv(StringIO(KG_Schema_str), sep=',',names=["stype","rel","otype"])
  non_exist_BGP_df=BGP_lst_df[~BGP_lst_df.isin(KG_Schema_df).all(axis=1)]
  print(non_exist_BGP_df)

if __name__ == '__main__':
    ############################### DBLP #######################
    # DBLP Schema
    DBLP2022_KG_str_Schema="""schema:AmbiguousCreator,schema:possibleActualCreator,schema:Creator
    schema:Creator,schema:orcid,xsd:anyUri
    schema:Creator,schema:creatorName,xsd:string
    schema:Creator,schema:primaryCreatorName,xsd:string
    schema:Creator,schema:creatorNote,xsd:string
    schema:Creator,schema:affiliation,xsd:string
    schema:Creator,schema:primaryAffiliation,xsd:string
    schema:Creator,schema:awardWebpage,Document
    schema:Creator,schema:homepage,Document
    schema:Creator,schema:primaryHomepage,Document
    schema:Creator,schema:creatorOf,schema:Publication
    schema:Creator,schema:authorOf,schema:Publication
    schema:Creator,schema:editorOf,schema:Publication
    schema:Creator,schema:coCreatorWith,schema:Creator
    schema:Creator,schema:coAuthorWith,schema:Creator
    schema:Creator,schema:coEditorWith,schema:Creator
    schema:Creator,schema:proxyAmbiguousCreator,schema:AmbiguousCreator
    schema:Entity,schema:identifier,xsd:anyUri
    schema:Entity,schema:wikidata,xsd:anyUri
    schema:Entity,schema:webpage,Document
    schema:Entity,schema:archivedWebpage,Document
    schema:Entity,schema:wikipedia,Document
    schema:Publication,schema:doi,xsd:anyUri
    schema:Publication,schema:isbn,xsd:anyUri
    schema:Publication,schema:title,xsd:string
    schema:Publication,schema:bibtexType,bibtex#Entry
    schema:Publication,schema:createdBy,schema:Creator
    schema:Publication,schema:authoredBy,schema:Creator
    schema:Publication,schema:editedBy,schema:Creator
    schema:Publication,schema:numberOfCreators,xsd:integer
    schema:Publication,schema:documentPage,Document
    schema:Publication,schema:primaryDocumentPage,Document
    schema:Publication,schema:listedOnTocPage,Document
    schema:Publication,schema:publishedInStream,schema:Stream
    schema:Publication,schema:publishedIn,xsd:string
    schema:Publication,schema:publishedInSeries,xsd:string
    schema:Publication,schema:publishedInSeriesVolume,xsd:string
    schema:Publication,schema:publishedInJournal,xsd:string
    schema:Publication,schema:publishedInJournalVolume,xsd:string
    schema:Publication,schema:publishedInJournalVolumeIssue,xsd:string
    schema:Publication,schema:publishedInBook,xsd:string
    schema:Publication,schema:publishedInBookChapter,xsd:string
    schema:Publication,schema:pagination,xsd:string
    schema:Publication,schema:yearOfEvent,xsd:gYear
    schema:Publication,schema:yearOfPublication,xsd:gYear
    schema:Publication,schema:monthOfPublication,xsd:gMonth
    schema:Publication,schema:publishedBy,xsd:string
    schema:Publication,schema:publishersAddress,xsd:string
    schema:Publication,schema:thesisAcceptedBySchool,xsd:string
    schema:Publication,schema:publicationNote,xsd:string
    schema:Publication,schema:publishedAsPartOf,schema:Publication
    schema:Publication,schema:isVersionOf,schema:Publication"""
    DBLP2022_KG_Schema_df = pd.read_csv(StringIO(DBLP2022_KG_str_Schema), sep='\,',names=["stype","rel","otype"])
    DBLP2022_KG_Schema_df.sort_values(by=["stype"])

    dblp_filtred_schema=DBLP2022_KG_str_Schema
    KG="DBLP"
    ####################### PV NC################################
    dblp_pv_suggested_features,dblp_pv_suggested_features_usage,dblp_pv_suggested_features_full_response,dblp_pv_suggested_features_p=suggest_features_prompt(task="classify a publication into a certian publication venue on DBLP KG.")
    dblp_pv_matching_BGP_lst,dblp_pv_matching_p_usage,dblp_pv_matching_p_full_response,dblp_pv_matching_p=match_features_prompt(dblp_filtred_schema,dblp_pv_suggested_features,KG)
    print('dblp_pv_matching_BGP_lst=',dblp_pv_matching_BGP_lst.split('\n'))
    dblp_SPARQL_example=f""" PREFIX schema: <https://dblp.org/rdf/schema#>
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
      select ?s ?p ?o
      from <https://dblp2022.org>
      where
      {{
        {{select ?s ?p ?o where {{?s a schema:Publication. ?s schema:title  ?o. BIND('dblp:title' AS ?p). }}}}
        {{select ?s ?p ?o where {{?s a schema:Publication. ?s rdfs:label  ?o. BIND('rdfs:label' AS ?p). }}}}
      }}"""
    sparql_v0,sparql_v0_p_usage,sparql_v0_p_full_response,sparql_v0_p=generate_sparql_query(dblp_pv_matching_BGP_lst,KG="DBLP",VT="Publication",SPARQL_Example=dblp_SPARQL_example)
    print("sparql_v0=",sparql_v0)
    final_sparql,final_sparql_p_usage,final_sparql_p_full_response,final_sparql_p=refine_sparql_query(sparql_v0)
    print("final_sparql=",final_sparql)
    ######################### DBLP AA LP##############################
    dblp_AA_suggested_features,dblp_AA_suggested_features_usage,dblp_AA_suggested_features_full_response,dblp_AA_suggested_features_p=suggest_features_prompt(task="predict the affaliation link of an author (creator) on DBLP KG , a link prediction task")
    print("dblp_AA_suggested_features=",dblp_AA_suggested_features)
    dblp_AA_matching_BGP_lst,dblp_AA_matching_p_usage,dblp_AA_matching_p_full_response,dblp_AA_matching_p=match_features_prompt(DBLP2022_KG_str_Schema,dblp_AA_suggested_features,KG)
    print("dblp_AA_matching_BGP_lst=",dblp_AA_matching_BGP_lst)
    VT="Person"
    dblp_SPARQL_example=f""" PREFIX schema: <https://dblp.org/rdf/schema#>
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
      select ?s ?p ?o
      from <https://dblp2022.org>
      where
      {{
        {{select ?s ?p ?o where {{?s a schema:Publication. ?s schema:title  ?o. BIND('dblp:title' AS ?p). }}}}
        {{select ?s ?p ?o where {{?s a schema:Publication. ?s rdfs:label  ?o. BIND('rdfs:label' AS ?p). }}}}
      }}"""
    sparql_v0,sparql_v0_p_usage,sparql_v0_p_full_response,sparql_v0_p=generate_sparql_query(dblp_AA_matching_BGP_lst,VT=VT,KG="DBLP",SPARQL_Example=dblp_SPARQL_example)
    print("sparql_v0=",sparql_v0)
    final_sparql,final_sparql_p_usage,final_sparql_p_full_response,final_sparql_p=refine_sparql_query(sparql_v0)
    print("final_sparql=",final_sparql)

    ######################### YAGO4  ############################
    yago4_schema="""schema:Thing,rdfs:comment,rdf:langString
    schema:Thing,rdfs:label,rdf:langString
    schema:Thing,schema:alternateName,rdf:langString
    schema:Thing,schema:image,xsd:anyURI
    schema:Thing,schema:mainEntityOfPage,xsd:anyURI
    schema:Thing,schema:sameAs,xsd:anyURI
    schema:Thing,schema:url,xsd:anyURI
    schema:CreativeWork,schema:about, schema:Thing
    yago:Creator,yago:influencedBy, schema:Thing
    schema:Event,schema:about, schema:Thing
    schema:MusicGroup,yago:influencedBy, schema:Thing
    schema:Person,schema:owns, schema:Thing
    schema:CreativeWork,schema:about,schema:Thing
    schema:CreativeWork,schema:author,schema:Organization
    schema:CreativeWork,schema:author,schema:Person
    schema:CreativeWork,schema:award,yago:Award
    schema:CreativeWork,schema:contentLocation,schema:Place
    schema:CreativeWork,schema:dateCreated,xsd:dateTime
    schema:CreativeWork,schema:dateCreated,xsd:date
    schema:CreativeWork,schema:dateCreated,xsd:gYearMonth
    schema:CreativeWork,schema:dateCreated,xsd:gYear
    schema:CreativeWork,schema:inLanguage,schema:Language
    yago:Creator,yago:notableWork, schema:CreativeWork
    yago:FictionalEntity,yago:appearsIn, schema:CreativeWork
    schema:PerformingGroup,yago:notableWork, schema:CreativeWork
    schema:Book,schema:editor,schema:Person
    schema:Book,schema:illustrator,schema:Person
    schema:Book,schema:isbn,xsd:string
    schema:Book,schema:numberOfPages,xsd:decimal
    schema:Book,schema:publisher,schema:Organization
    schema:Book,schema:publisher,schema:Person
    schema:Movie,schema:actor,schema:Person
    schema:Movie,schema:director,schema:Person
    schema:Movie,schema:duration,xsd:decimal
    schema:Movie,schema:locationCreated,schema:Place
    schema:Movie,schema:musicBy,schema:MusicGroup
    schema:Movie,schema:musicBy,schema:Person
    schema:Movie,schema:productionCompany,schema:Organization
    schema:MusicComposition,schema:iswcCode,xsd:string
    schema:MusicComposition,schema:lyricist,schema:Person
    schema:MusicComposition,schema:lyricist,schema:MusicGroup
    schema:MusicComposition,schema:musicBy,schema:Person
    schema:MusicComposition,schema:musicBy,schema:MusicGroup
    schema:Newspaper,schema:publisher,schema:Organization
    schema:Newspaper,schema:publisher,schema:Person
    schema:Newspaper,schema:sponsor,schema:Organization
    schema:Newspaper,schema:sponsor,schema:Person
    schema:TVSeries,schema:actor,schema:Person
    schema:TVSeries,schema:director,schema:Person
    schema:TVSeries,schema:locationCreated,schema:Place
    schema:TVSeries,schema:musicBy,schema:MusicGroup
    schema:TVSeries,schema:musicBy,schema:Person
    schema:TVSeries,schema:numberOfEpisodes,xsd:decimal
    schema:TVSeries,schema:numberOfSeasons,xsd:decimal
    schema:TVSeries,schema:productionCompany,schema:Organization
    schema:Event,schema:about,schema:Thing
    schema:Event,schema:endDate,xsd:dateTime
    schema:Event,schema:endDate,xsd:date
    schema:Event,schema:endDate,xsd:gYearMonth
    schema:Event,schema:endDate,xsd:gYear
    schema:Event,schema:location,schema:Place
    schema:Event,schema:organizer,schema:Person
    schema:Event,schema:organizer,schema:Organization
    schema:Event,schema:sponsor,schema:Organization
    schema:Event,schema:sponsor,schema:Person
    schema:Event,schema:startDate,xsd:dateTime
    schema:Event,schema:startDate,xsd:date
    schema:Event,schema:startDate,xsd:gYearMonth
    schema:Event,schema:startDate,xsd:gYear
    schema:Event,schema:superEvent,schema:Event
    schema:Event,yago:follows,schema:Event
    schema:Event,yago:participant,schema:Organization
    schema:Event,yago:participant,schema:Person
    schema:Event,yago:follows, schema:Event
    schema:Event,schema:superEvent, schema:Event
    yago:Politician,yago:candidateIn, schema:Event
    yago:SportsPerson,yago:playsIn, schema:Event
    schema:Country,yago:officialLanguage, schema:Language
    schema:CreativeWork,schema:inLanguage, schema:Language
    schema:PerformingGroup,schema:knowsLanguage, schema:Language
    schema:Person,schema:knowsLanguage, schema:Language
    yago:Award,yago:conferredBy,schema:Organization
    yago:Award,yago:conferredBy,schema:Person
    schema:CreativeWork,schema:award, yago:Award
    schema:Organization,schema:award, yago:Award
    schema:Person,yago:academicDegree, yago:Award
    schema:Person,schema:award, yago:Award
    schema:Product,schema:award, yago:Award
    schema:Person,yago:beliefSystem, yago:BeliefSystem
    schema:Person,schema:gender, yago:Gender
    schema:Organization,schema:address,xsd:string
    schema:Organization,schema:award,yago:Award
    schema:Organization,schema:dateCreated,xsd:dateTime
    schema:Organization,schema:dateCreated,xsd:date
    schema:Organization,schema:dateCreated,xsd:gYearMonth
    schema:Organization,schema:dateCreated,xsd:gYear
    schema:Organization,schema:dissolutionDate,xsd:dateTime
    schema:Organization,schema:dissolutionDate,xsd:date
    schema:Organization,schema:dissolutionDate,xsd:gYearMonth
    schema:Organization,schema:dissolutionDate,xsd:gYear
    schema:Organization,schema:duns,xsd:string
    schema:Organization,schema:founder,schema:Person
    schema:Organization,yago:leader,schema:Person
    schema:Organization,schema:leiCode,xsd:string
    schema:Organization,schema:location,schema:Place
    schema:Organization,schema:locationCreated,schema:Place
    schema:Organization,schema:logo,xsd:anyURI
    schema:Organization,schema:memberOf,schema:Organization
    schema:Organization,schema:motto,xsd:string
    schema:Organization,schema:numberOfEmployees,xsd:decimal
    schema:Organization,yago:ownedBy,schema:Organization
    schema:Organization,yago:ownedBy,schema:Person
    schema:AdministrativeArea,schema:memberOf, schema:Organization
    schema:Movie,schema:productionCompany, schema:Organization
    schema:MusicGroup,schema:recordLabel, schema:Organization
    schema:Organization,schema:memberOf, schema:Organization
    schema:Person,schema:affiliation, schema:Organization
    schema:Person,schema:worksFor, schema:Organization
    schema:Person,schema:alumniOf, schema:Organization
    schema:Person,schema:memberOf, schema:Organization
    schema:TVSeries,schema:productionCompany, schema:Organization
    schema:Product,schema:manufacturer, schema:Corporation
    schema:Airline,schema:iataCode,xsd:string
    schema:Airline,schema:icaoCode,xsd:string
    schema:EducationalOrganization,yago:studentsCount,xsd:decimal
    schema:PerformingGroup,schema:knowsLanguage,schema:Language
    schema:PerformingGroup,yago:director,schema:Person
    schema:PerformingGroup,yago:notableWork,schema:CreativeWork
    schema:MusicGroup,yago:influencedBy,schema:Thing
    schema:MusicGroup,schema:recordLabel,schema:Organization
    schema:Person,schema:affiliation,schema:Organization
    schema:Person,schema:alumniOf,schema:Organization
    schema:Person,schema:award,yago:Award
    schema:Person,schema:birthDate,xsd:dateTime
    schema:Person,schema:birthDate,xsd:date
    schema:Person,schema:birthDate,xsd:gYearMonth
    schema:Person,schema:birthDate,xsd:gYear
    schema:Person,schema:birthPlace,schema:Place
    schema:Person,schema:children,schema:Person
    schema:Person,schema:deathDate,xsd:dateTime
    schema:Person,schema:deathDate,xsd:date
    schema:Person,schema:deathDate,xsd:gYearMonth
    schema:Person,schema:deathDate,xsd:gYear
    schema:Person,schema:deathPlace,schema:Place
    schema:Person,schema:gender,yago:Gender
    schema:Person,schema:homeLocation,schema:Place
    schema:Person,schema:knowsLanguage,schema:Language
    schema:Person,schema:memberOf,schema:Organization
    schema:Person,schema:nationality,schema:Country
    schema:Person,schema:owns,schema:Thing
    schema:Person,schema:spouse,schema:Person
    schema:Person,schema:worksFor,schema:Organization
    schema:Person,yago:academicDegree,yago:Award
    schema:Person,yago:beliefSystem,yago:BeliefSystem
    yago:Academic,yago:studentOf, schema:Person
    yago:Academic,yago:doctoralAdvisor, schema:Person
    schema:AdministrativeArea,yago:leader, schema:Person
    schema:Book,schema:illustrator, schema:Person
    schema:Book,schema:editor, schema:Person
    yago:FictionalEntity,schema:performer, schema:Person
    schema:Movie,schema:actor, schema:Person
    schema:Movie,schema:director, schema:Person
    schema:Organization,schema:founder, schema:Person
    schema:Organization,yago:leader, schema:Person
    schema:PerformingGroup,yago:director, schema:Person
    schema:Person,schema:children, schema:Person
    schema:Person,schema:spouse, schema:Person
    schema:TVSeries,schema:actor, schema:Person
    schema:TVSeries,schema:director, schema:Person
    yago:Academic,yago:doctoralAdvisor,schema:Person
    yago:Academic,yago:studentOf,schema:Person
    yago:Creator,yago:influencedBy,schema:Thing
    yago:Creator,yago:notableWork,schema:CreativeWork
    yago:Politician,yago:candidateIn,schema:Event
    yago:SportsPerson,yago:playsIn,schema:Event
    yago:SportsPerson,yago:sportNumber,xsd:string
    schema:Place,schema:area,xsd:decimal
    schema:Place,schema:elevation,xsd:decimal
    schema:Place,schema:geo,geo:wktLiteral
    schema:Place,yago:highestPoint,schema:Place
    schema:Place,schema:location,schema:Place
    schema:Place,yago:lowestPoint,schema:Place
    schema:Place,yago:neighbors,schema:Place
    schema:CreativeWork,schema:contentLocation, schema:Place
    schema:Event,schema:location, schema:Place
    schema:Movie,schema:locationCreated, schema:Place
    schema:Organization,schema:locationCreated, schema:Place
    schema:Organization,schema:location, schema:Place
    schema:Person,schema:birthPlace, schema:Place
    schema:Person,schema:deathPlace, schema:Place
    schema:Person,schema:homeLocation, schema:Place
    schema:Place,schema:location, schema:Place
    schema:Place,yago:highestPoint, schema:Place
    schema:Place,yago:lowestPoint, schema:Place
    schema:Place,yago:neighbors, schema:Place
    schema:TVSeries,schema:locationCreated, schema:Place
    yago:Way,yago:terminus, schema:Place
    schema:AdministrativeArea,schema:dateCreated,xsd:dateTime
    schema:AdministrativeArea,schema:demonym,xsd:string
    schema:AdministrativeArea,yago:leader,schema:Person
    schema:AdministrativeArea,schema:memberOf,schema:Organization
    schema:AdministrativeArea,schema:motto,xsd:string
    schema:AdministrativeArea,schema:populationNumber,xsd:decimal
    schema:AdministrativeArea,schema:postalCode,xsd:string
    schema:AdministrativeArea,yago:capital,schema:City
    schema:AdministrativeArea,yago:replaces,schema:AdministrativeArea
    schema:AdministrativeArea,yago:replaces, schema:AdministrativeArea
    schema:AdministrativeArea,yago:capital, schema:City
    schema:Country,schema:humanDevelopmentIndex,xsd:decimal
    schema:Country,yago:officialLanguage,schema:Language
    schema:Country,schema:unemploymentRate,xsd:decimal
    schema:Person,schema:nationality, schema:Country
    schema:BodyOfWater,yago:flowsInto,schema:BodyOfWater
    schema:BodyOfWater,yago:flowsInto, schema:BodyOfWater
    yago:AstronomicalObject,yago:distanceFromEarth,xsd:decimal
    yago:AstronomicalObject,yago:luminosity,xsd:decimal
    yago:AstronomicalObject,yago:mass,xsd:decimal
    yago:AstronomicalObject,yago:parallax,xsd:decimal
    yago:AstronomicalObject,yago:parentBody,yago:AstronomicalObject
    yago:AstronomicalObject,yago:radialVelocity,xsd:decimal
    yago:AstronomicalObject,yago:parentBody, yago:AstronomicalObject
    yago:HumanMadeGeographicalEntity,schema:dateCreated,xsd:dateTime
    yago:HumanMadeGeographicalEntity,yago:ownedBy,schema:Organization
    yago:HumanMadeGeographicalEntity,yago:ownedBy,schema:Person
    schema:Airport,schema:iataCode,xsd:string
    schema:Airport,schema:icaoCode,xsd:string
    yago:Way,yago:length,xsd:decimal
    yago:Way,yago:terminus,schema:Place
    schema:Product,schema:award,yago:Award
    schema:Product,schema:dateCreated,xsd:dateTime
    schema:Product,schema:dateCreated,xsd:date
    schema:Product,schema:dateCreated,xsd:gYearMonth
    schema:Product,schema:dateCreated,xsd:gYear
    schema:Product,schema:gtin,xsd:string
    schema:Product,schema:manufacturer,schema:Corporation
    schema:Product,schema:material,schema:Product
    schema:Product,schema:material, schema:Product
    schema:Taxon,schema:parentTaxon,schema:Taxon
    schema:Taxon,yago:consumes,schema:Taxon
    schema:Taxon,schema:parentTaxon, schema:Taxon
    schema:Taxon,yago:consumes, schema:Taxon
    yago:FictionalEntity,schema:author,schema:Organization
    yago:FictionalEntity,schema:author,schema:Person
    yago:FictionalEntity,schema:performer,schema:Person
    yago:FictionalEntity,yago:appearsIn,schema:CreativeWork"""
    yago4_schema_df=pd.DataFrame(data=[x.split(',') for x in yago4_schema.split('\n')],columns=["stype","ptype","otype"])
    yago4_schema_df
    YAGO4_KG_Place_Schema_df=yago4_schema_df[(yago4_schema_df["stype"]=="schema:Place") | (yago4_schema_df["otype"]=="schema:Place")]
    YAGO4_KG_Place_Schema_txt=YAGO4_KG_Place_Schema_df.to_csv(index=False,header=None)
    ################### YAGO4 PC NC ###########################
    yag4_pc_suggested_features,yag4_pc_suggested_features_usage,yag4_pc_suggested_features_full_response,yag4_pc_suggested_features_p=suggest_features_prompt(task="classify a place into crossponding country from YAGO4 KG.")
    KG="YAGO4"
    print("yag4_pc_suggested_features=",yag4_pc_suggested_features)
    yag4_pc_matching_BGP_lst,yag4_pc_matching_p_usage,yag4_pc_matching_p_full_response,yag4_pc_matching_p=match_features_prompt(YAGO4_KG_Place_Schema_txt,yag4_pc_suggested_features,KG)
    print("yag4_pc_matching_BGP_lst=",yag4_pc_matching_BGP_lst)
    VT="Place"
    dblp_SPARQL_example=f"""Prefix schema:<http://schema.org/>
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    select ?s ?p ?o
    from <https://yago-knowledge.org> where
      {{
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s schema:alternateName  ?o. BIND('schema:alternateName' AS ?p). }}}}
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s rdfs:label  ?o. BIND('rdfs:label' AS ?p). }}}}
      }}"""
    sparql_v0,sparql_v0_p_usage,sparql_v0_p_full_response,sparql_v0_p=generate_sparql_query(yag4_pc_matching_BGP_lst,VT=VT,KG="YAGO4",SPARQL_Example=dblp_SPARQL_example)
    print("sparql_v0=",sparql_v0)
    final_sparql,final_sparql_p_usage,final_sparql_p_full_response,final_sparql_p=refine_sparql_query(sparql_v0)
    print("final_sparql=",final_sparql)
    ######################## YAGO4 CW Genre NC #################################
    YAGO4_KG_CW_Schema_df=yago4_schema_df[(yago4_schema_df["stype"]=="schema:CreativeWork") | (yago4_schema_df["otype"]=="schema:CreativeWork")].drop_duplicates()
    YAGO4_KG_CW_Schema_txt=YAGO4_KG_Place_Schema_df.to_csv(index=False,header=None)
    YAGO4_KG_CW_Schema_df
    yag4_CWG_suggested_features,yag4_CWG_suggested_features_usage,yag4_CWG_suggested_features_full_response,yag4_CWG_suggested_features_p=suggest_features_prompt(task="classify a CreativeWork into crossponding Genre from YAGO4 KG.")
    KG="YAGO4"
    print("yag4_CWG_suggested_features=",yag4_CWG_suggested_features)
    yag4_CWG_matching_BGP_lst,yag4_CWG_matching_p_usage,yag4_CWG_matching_p_full_response,yag4_CWG_matching_p=match_features_prompt(YAGO4_KG_CW_Schema_df,yag4_CWG_suggested_features,KG)
    print("yag4_CWG_matching_BGP_lst=",yag4_CWG_matching_BGP_lst)
    VT="CreativeWork"
    dblp_SPARQL_example=f"""Prefix schema:<http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    select ?s ?p ?o
    from <https://yago-knowledge.org> where
      {{
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s schema:alternateName  ?o. BIND('schema:alternateName' AS ?p). }}}}
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s rdfs:label  ?o. BIND('rdfs:label' AS ?p). }}}}
      }}"""
    sparql_v0,sparql_v0_p_usage,sparql_v0_p_full_response,sparql_v0_p=generate_sparql_query(yag4_CWG_matching_BGP_lst,VT=VT,KG="YAGO4",SPARQL_Example=dblp_SPARQL_example)
    print("sparql_v0=",sparql_v0)
    final_sparql,final_sparql_p_usage,final_sparql_p_full_response,final_sparql_p=refine_sparql_query(sparql_v0)
    print("final_sparql=",final_sparql)
    ##################### YAGO4 CA Link Prediction #############################
    YAGO4_KG_CA_Schema_df_l1=yago4_schema_df[(yago4_schema_df["stype"]=="schema:Airport") | (yago4_schema_df["otype"]=="schema:Airport")]
    YAGO4_KG_CA_Schema_df_list=set(YAGO4_KG_CA_Schema_df_l1["stype"].tolist()).union(set(YAGO4_KG_CA_Schema_df_l1["otype"].tolist()))
    YAGO4_KG_CA_Schema_df_list.remove('xsd:string')
    print("YAGO4_KG_CA_Schema_df_l1:",YAGO4_KG_CA_Schema_df_list)
    YAGO4_KG_CA_Schema_df_l2_s=yago4_schema_df[yago4_schema_df["stype"].isin(YAGO4_KG_CA_Schema_df_list)]
    YAGO4_KG_CA_Schema_df_l2_o=yago4_schema_df[yago4_schema_df["otype"].isin(YAGO4_KG_CA_Schema_df_list)]
    YAGO4_KG_CA_Schema_df=pd.concat([YAGO4_KG_CA_Schema_df_l1,YAGO4_KG_CA_Schema_df_l2_s,YAGO4_KG_CA_Schema_df_l2_o])
    YAGO4_KG_CA_Schema_txt=YAGO4_KG_CA_Schema_df.to_csv(index=False,header=None)

    ################################ MAG ###############################
    mag_schema="""Author,creator,Paper
    Paper,cites,Paper
    Paper,hasCitingEntity,Citation
    Paper,hasCitedEntity,Citation
    Paper,recommends,Paper
    Affiliation,created,date
    Affiliation,rdf-schema#seeAlso,AnyURL
    Affiliation,owl#sameAs,AnyURL
    Affiliation,wgs84_pos#lat,latitude
    Affiliation,wgs84_pos#long,longitude
    Affiliation,homepage,AnyURL
    Affiliation,name,string
    Affiliation,isRelatedTo,Affiliation
    Affiliation,citationCount,int
    Affiliation,grid,string
    Affiliation,paperCount,int
    Affiliation,paperFamilyCount,int
    Affiliation,rank,int
    FieldOfStudy,level,string
    FieldOfStudy,hasParent,FieldOfStudy
    FieldOfStudy,diseaseHasDiseaseCause,FieldOfStudy
    FieldOfStudy,diseaseHasMedicalTreatment,FieldOfStudy
    FieldOfStudy,diseaseHasSymptom,FieldOfStudy
    FieldOfStudy,symptomHasDiseaseCause,FieldOfStudy
    FieldOfStudy,medicalTreatmentForDiseaseCause,FieldOfStudy
    FieldOfStudy,medicalTreatmentForSymptom,FieldOfStudy
    ConferenceSeries,issn,string
    Author,org#memberOf,Affiliation
    Paper,publicationDate,date
    Paper,title,string
    Paper,keyword,string
    Paper,hasDiscipline,FieldOfStudy
    Paper,publisher,publisher
    Paper,hasPubMedId,string
    Paper,appearsInJournal,ConferenceSeries
    Paper,estimatedCitationCount,int
    Paper,referenceCount,int
    Paper,doi,string
    Paper,endingPage,int
    Paper,issueIdentifier,int
    Paper,startingPage,int
    Paper,volume,int
    Author,hasORCID,string
    Paper,hasPubMedCentralId,string
    Paper,familyId,string
    Paper,hasPatentNumber,int
    Paper,patent,patent
    ConferenceInstance,location,location
    ConferenceInstance,timeline.owl#start,date
    ConferenceInstance,timeline.owl#end,date
    ConferenceInstance,isPartOf,ConferenceSeries
    ConferenceInstance,pageCount,int
    ConferenceInstance,submissionDeadlineDate,date
    Paper,appearsInConferenceSeries,ConferenceSeries
    ConferenceInstance,finalVersionDueDate,date
    ConferenceInstance,notificationDueDate,date
    Paper,citedResource,Work
    Paper,ownResource,Work
    """
    mag_schema_df=pd.DataFrame(data=[x.split(',') for x in mag_schema.split('\n')],columns=["stype","ptype","otype"])
    MAG_KG_Place_Schema_df=mag_schema_df[(mag_schema_df["stype"]=="Paper") | (mag_schema_df["otype"]=="Paper")]
    MAG_KG_Place_Schema_txt=MAG_KG_Place_Schema_df.to_csv(index=False,header=None)
    ################## MAG PV NC ##################
    MAG_pv_suggested_features,MAG_pv_suggested_features_usage,MAG_pv_suggested_features_full_response,MAG_pv_suggested_features_p=suggest_features_prompt(task="classify a paper into crossponding publication venue from MAG KG.")
    KG="MAG"
    print("MAG_pv_suggested_features=",MAG_pv_suggested_features)
    MAG_pv_matching_BGP_lst,MAG_pv_matching_p_usage,MAG_pv_matching_p_full_response,MAG_pv_matching_p=match_features_prompt(MAG_KG_Place_Schema_txt,MAG_pv_suggested_features,KG)
    print("MAG_pv_matching_BGP_lst=",MAG_pv_matching_BGP_lst)
    VT="Paper"
    dblp_SPARQL_example=f"""Prefix schema:<http://schema.org/>
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    select ?s ?p ?o
    from <https://yago-knowledge.org> where
      {{
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s schema:alternateName  ?o. BIND('schema:alternateName' AS ?p). }}}}
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s rdfs:label  ?o. BIND('rdfs:label' AS ?p). }}}}
      }}"""
    sparql_v0,sparql_v0_p_usage,sparql_v0_p_full_response,sparql_v0_p=generate_sparql_query(MAG_pv_matching_BGP_lst,VT=VT,KG="MAG",SPARQL_Example=dblp_SPARQL_example)
    print("sparql_v0=",sparql_v0)
    final_sparql,final_sparql_p_usage,final_sparql_p_full_response,final_sparql_p=refine_sparql_query(sparql_v0)
    print("final_sparql=",final_sparql)
    #################################### YAGO3 10 ##########################
    yago3_10_schema="""Place,isLocatedIn,Place
    Player,playsFor,Club
    Person,isAffiliatedTo,Organization
    Person,diedIn,Place
    Person,actedIn,Movie
    Person,graduatedFrom,Organization
    Person,wasBornIn,Place
    Person,hasGender,Gender
    Event,happenedIn,Place
    Person,hasMusicalRole,Role
    Place,isConnectedTo,Place
    Person,isMarriedTo,Person
    Place,participatedIn,Event
    Place,hasOfficialLanguage,Language
    Person,hasWonPrize,Prizw
    Person,influences,Person
    Person,worksAt,Place
    Person,created,CreativeWork
    Person,edited,CreativeWork
    Person,directed,CreativeWork
    Person,hasAcademicAdvisor,Person
    Place,imports,Things
    Person,isPoliticianOf,Place
    Person,wroteMusicFor,CreativeWork
    Person,isInterestedIn,CreativeWork
    Person,isCitizenOf,Place
    Person,hasChild,Person
    Person,isLeaderOf,Place
    Place,dealsWith,Place
    Person,livesIn,Place
    Place,hasCapital,Place
    Place,hasNeighbor,Place
    Place,exports,Thing
    Organization,owns,Organization
    Place,hasCurrency,hasCurrency
    Organization,hasWebsite,Website
    Person,isKnownFor,CreativeWork"""
    yago3_10_schema_df=pd.DataFrame(data=[x.split(',') for x in yago3_10_schema.split('\n')],columns=["stype","ptype","otype"])
    yago3_10_schema_df
    yago3_10_KG_Place_Schema_df=yago3_10_schema_df[(yago3_10_schema_df["stype"]=="Airport") | (yago3_10_schema_df["otype"]=="Airport")]
    yago3_10_KG_Place_Schema_txt=yago3_10_KG_Place_Schema_df.to_csv(index=False,header=None)
    ####################### YAGO3-10 CA LP ########################
    yago3_10_suggested_features,yago3_10_suggested_features_usage,yago3_10_suggested_features_full_response,yago3_10_suggested_features_p=suggest_features_prompt(task="predict a connection link between two Airports as link prediction task from yago3_10 KG.")
    KG="yago3_10"
    print("yago3_10_suggested_features=",yago3_10_suggested_features)
    yago3_10_matching_BGP_lst,yago3_10_matching_p_usage,yago3_10_matching_p_full_response,yago3_10_matching_p=match_features_prompt(yago3_10_KG_Place_Schema_txt,yago3_10_suggested_features,KG)
    print("yago3_10_matching_BGP_lst=",yago3_10_matching_BGP_lst)
    VT="Airport"
    dblp_SPARQL_example=f"""Prefix schema:<http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    select ?s ?p ?o
    from <https://yago3-10-knowledge.org> where
      {{
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s schema:alternateName  ?o. BIND('schema:alternateName' AS ?p). }}}}
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s rdfs:label  ?o. BIND('rdfs:label' AS ?p). }}}}
      }}"""
    sparql_v0,sparql_v0_p_usage,sparql_v0_p_full_response,sparql_v0_p=generate_sparql_query(yago3_10_matching_BGP_lst,VT=VT,KG="yago3_10",SPARQL_Example=dblp_SPARQL_example)
    print("sparql_v0=",sparql_v0)
    final_sparql,final_sparql_p_usage,final_sparql_p_full_response,final_sparql_p=refine_sparql_query(sparql_v0)
    print("final_sparql=",final_sparql)

    ############################# YAGO3-10 LivesIN #################################
    yago3_10_KG_Place_Schema_df=yago3_10_schema_df[(yago3_10_schema_df["stype"]=="Place") | (yago3_10_schema_df["otype"]=="Place")]
    yago3_10_KG_Place_Schema_txt=yago3_10_KG_Place_Schema_df.to_csv(index=False,header=None)
    yago3_10_suggested_features,yago3_10_suggested_features_usage,yago3_10_suggested_features_full_response,yago3_10_suggested_features_p=suggest_features_prompt(task="predict where a person lives in as a link prediction task from yago3_10 KG.")
    KG="yago3_10"
    print("yago3_10_suggested_features=",yago3_10_suggested_features)
    yago3_10_matching_BGP_lst,yago3_10_matching_p_usage,yago3_10_matching_p_full_response,yago3_10_matching_p=match_features_prompt(yago3_10_KG_Place_Schema_txt,yago3_10_suggested_features,KG)
    print("yago3_10_matching_BGP_lst=",yago3_10_matching_BGP_lst)
    VT="Person"
    dblp_SPARQL_example=f"""Prefix schema:<http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    select ?s ?p ?o
    from <https://yago3-10-knowledge.org> where
      {{
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s schema:alternateName  ?o. BIND('schema:alternateName' AS ?p). }}}}
        {{select ?s ?p ?o where {{?s a schema:{VT}. ?s rdfs:label  ?o. BIND('rdfs:label' AS ?p). }}}}
      }}"""
    sparql_v0,sparql_v0_p_usage,sparql_v0_p_full_response,sparql_v0_p=generate_sparql_query(yago3_10_matching_BGP_lst,VT=VT,KG="yago3_10",SPARQL_Example=dblp_SPARQL_example)
    print("sparql_v0=",sparql_v0)
    final_sparql,final_sparql_p_usage,final_sparql_p_full_response,final_sparql_p=refine_sparql_query(sparql_v0)
    print("final_sparql=",final_sparql)



    ##############################vWikiKG ###############################
    import pandas as pd
    wikiKG_KG_Schema_df=pd.read_csv("/content/WikiKG2015_v2_Types.csv",header=None)
    wikiKG_KG_Schema_df.columns=["stype","ptype","otype"]
    wikiKG_KG_Schema_df
    """## Person Ocupation"""
    WikiDataPropertyLabels_df=pd.read_csv("/content/WikiDataPropertyLabels.tsv",sep="\t")
    WikiDataPropertyLabels_dict=dict(zip(WikiDataPropertyLabels_df["PID"].astype(str).tolist(),WikiDataPropertyLabels_df["PName"].tolist()))
    wikiKG_KG_Schema_df["PName"]=wikiKG_KG_Schema_df["ptype"].apply(lambda x: WikiDataPropertyLabels_dict[str(x)[1:]] if str(x)[1:] in WikiDataPropertyLabels_dict.keys() else "")
    wikiKG_KG_Schema_df
    wiki_KG_Person_Schema_df=wikiKG_KG_Schema_df[(wikiKG_KG_Schema_df["stype"]=="person") | (wikiKG_KG_Schema_df["otype"]=="person")
    | (wikiKG_KG_Schema_df["stype"]=="occupation") | (wikiKG_KG_Schema_df["otype"]=="occupation")
    ]
    wiki_KG_Person_Schema_txt=wiki_KG_Person_Schema_df[["stype","ptype","otype"]].drop_duplicates().to_csv(index=False,header=None)
    wiki_KG_Person_Schema_df.drop_duplicates()
    ##################### WikiKG PO LP #########################
    wikiKG_suggested_features, wikiKG_suggested_features_usage, wikiKG_suggested_features_full_response, wikiKG_suggested_features_p=suggest_features_prompt(task="predict where a person occupaation (profession) as a link prediction task from  wikiKG knoweldge graph.")
    KG=" wikiKG"
    print(" wikiKG_suggested_features=", wikiKG_suggested_features)
    wikiKG_matching_BGP_lst, wikiKG_matching_p_usage, wikiKG_matching_p_full_response, wikiKG_matching_p=match_features_prompt( wiki_KG_Person_Schema_txt, wikiKG_suggested_features,KG)
    print(" wikiKG_matching_BGP_lst=", wikiKG_matching_BGP_lst)
    VT="Person"
    dblp_SPARQL_example=f"""
    PREFIX wiki: <http://www.wikidata.org/entity/>
    select ?s ?p ?o
    from <http://wikikg-v2> where
      {{
        {{select ?s ?p ?o where {{?s a wiki:{VT}. ?s wiki:alternateName  ?o. BIND('wiki:alternateName' AS ?p). }}}}
        {{select ?s ?p ?o where {{?s a wiki:{VT}. ?s wiki:label  ?o. BIND('wiki:label' AS ?p). }}}}
      }}"""
    sparql_v0,sparql_v0_p_usage,sparql_v0_p_full_response,sparql_v0_p=generate_sparql_query( wikiKG_matching_BGP_lst,VT=VT,KG=" wikiKG",SPARQL_Example=dblp_SPARQL_example)
    print("sparql_v0=",sparql_v0)
    final_sparql,final_sparql_p_usage,final_sparql_p_full_response,final_sparql_p=refine_sparql_query(sparql_v0)
    print("final_sparql=",final_sparql)