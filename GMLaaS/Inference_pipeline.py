import os
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_query as get_NC_d1h1_query
from SparqlMLaasService.GMLOperators import gmlOperator
from KGNET import KGNET
from KGNET import Constants
from GMLaaS.DataTransform.INFERENCE_TSV_TO_PYG import inference_transform_tsv_to_PYG
from GMLaaS.models.graph_saint_Shadow_KGTOSA import graphShadowSaint
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint


# output_path = Constants.KGNET_Config.inference_path
if not os.path.exists(Constants.KGNET_Config.inference_path):
    os.makedirs(Constants.KGNET_Config.inference_path)
inference_file = os.path.join(Constants.KGNET_Config.inference_path,'inference.tsv')
def get_MetaData(model_id):
    dict_params = {'model': {}, 'subG': {}}
    kgmeta_govener = KGMeta_Governer(endpointUrl=Constants.KGNET_Config.KGMeta_endpoint_url,
                                     KGMeta_URI=Constants.KGNET_Config.KGMeta_IRI)
    model_id_format = str(model_id)#"<kgnet:GMLModel/"+model_id+">"
    query="""
    prefix kgnet:<http://kgnet/>
    select * where
    {
        {
            select ?t as ?s  ?p as ?p  ?o as ?o
            from <http://kgnet/>
            where
            {
            ?t ?p ?o.
            ?t ?tp ?s.
            ?s <kgnet:GMLModel/id> """+model_id_format+""".
            }
        }        
        union
        {
            select ?m as ?s ?p as ?p  ?o as ?o
            from <http://kgnet/>
            where
            {
            ?m ?p ?o.
            ?m <kgnet:GMLModel/id> """+model_id_format+""".
            }
        }        
    }
    limit 100
    
    """
    # print(query)
    res_df = kgmeta_govener.executeSparqlquery(query)
    res_df = res_df.applymap(lambda x: x.strip('"'))
    dict_params['model']['embSize'] = int(res_df[res_df['p'] == Constants.GNN_KG_HParms.Emb_size]['o'].item())
    dict_params['model']['hiddenChannels'] = int(res_df[res_df['p'] == Constants.GNN_KG_HParms.HiddenChannels]['o'].item())
    dict_params['model']['Num_Layers'] = int(res_df[res_df['p'] == Constants.GNN_KG_HParms.Num_Layers]['o'].item())

    dict_params['subG']['targetEdge'] = str(res_df[res_df['p'] == Constants.GNN_SubG_Parms.targetEdge]['o'].item())
    dict_params['subG']['graphPrefix'] = str(res_df[res_df['p'] == Constants.GNN_SubG_Parms.prefix]['o'].item())

    return dict_params

def generate_subgraph(named_graph_uri,target_rel_uri,targetNode_filter_statements,sparqlEndpointURL):


    query = [get_NC_d1h1_query(graph_uri=named_graph_uri, target_rel_uri=target_rel_uri,
                               tragetNode_filter_statments=targetNode_filter_statements)]
    kg = KGNET(sparqlEndpointURL)
    subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(query,)
    subgraph_df = subgraph_df.applymap(lambda x : x.strip('"'))
    subgraph_df.to_csv(inference_file,index=None,sep='\t')
    return inference_file


def get_rel_types(named_graph_uri, graphPrefix, sparqlEndpointURL):
    types_file = os.path.join(Constants.KGNET_Config.datasets_output_path, graphPrefix + '_Types.csv')

    if os.path.exists(types_file):
        return types_file
    KG_sparqlEndpoint = sparqlEndpoint(endpointUrl=sparqlEndpointURL)
    gml_operator = gmlOperator(KG_sparqlEndpoint=KG_sparqlEndpoint)
    gml_operator.getKGNodeEdgeTypes(namedGraphURI=named_graph_uri, prefix=graphPrefix)

    if not os.path.exists(types_file):
        raise FileNotFoundError(f'Types file was not generated at expected location: {types_file}')

    else:
        return types_file


def perform_inference(model_id,named_graph_uri,targetNode_filter_statements,sparqlEndpointURL):
    meta_dict = get_MetaData(model_id)
    subgraph = generate_subgraph(named_graph_uri = named_graph_uri,
                                target_rel_uri = meta_dict['subG']['targetEdge'],
                                targetNode_filter_statements = targetNode_filter_statements,
                                sparqlEndpointURL = sparqlEndpointURL)

    rel_types = get_rel_types(named_graph_uri = named_graph_uri,
                              graphPrefix = meta_dict['subG']['graphPrefix'],
                              sparqlEndpointURL = sparqlEndpointURL)

    inference_transform_tsv_to_PYG(dataset_name='inference',
                                   dataset_name_csv='inference',
                                   dataset_types=rel_types,
                                   target_rel=meta_dict['subG']['targetEdge'],
                                   output_root_path=Constants.KGNET_Config.inference_path,
                                   Header_row=True)

    graphShadowSaint(dataset_name='inference',root_path=Constants.KGNET_Config.inference_path,loadTrainedModel=1)













    # subgraph.to_csv(Constants.)
    print ('pass!')
