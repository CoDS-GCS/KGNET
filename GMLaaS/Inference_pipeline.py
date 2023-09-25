import os
import shutil
from SparqlMLaasService.KGMeta_Governer import KGMeta_Governer
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_query as get_NC_d1h1_query
from SparqlMLaasService.GMLOperators import gmlOperator
from KGNET import KGNET
from KGNET import Constants
from GMLaaS.DataTransform.INFERENCE_TSV_TO_PYG import inference_transform_tsv_to_PYG
from GMLaaS.models.graph_saint_Shadow_KGTOSA import graphShadowSaint
from GMLaaS.models.graph_saint_KGTOSA import graphSaint
from GMLaaS.models.rgcn.rgcn_link_pred import rgcn_lp
# from GMLaaS.models.graph_saint_KGTOSA_DEMO import graphSaint
from GMLaaS.DataTransform.Transform_LP_Dataset import transform_LP_train_valid_test_subsets
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from model_manager import downloadModel,downloadDataset
import datetime
import pandas as pd

# output_path = Constants.KGNET_Config.inference_path
inference_file = os.path.join(Constants.KGNET_Config.inference_path, 'inference.tsv')


def delete_inference_cache(inference_path=Constants.KGNET_Config.inference_path):
    for entry in os.listdir(inference_path):
        entry_path = os.path.join(inference_path, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)


def get_MetaData(model_id):
    dict_params = {'model': {}, 'subG': {}}
    kgmeta_govener = KGMeta_Governer(endpointUrl=Constants.KGNET_Config.KGMeta_endpoint_url,
                                     KGMeta_URI=Constants.KGNET_Config.KGMeta_IRI)
    model_id_format = str(model_id)  # "<kgnet:GMLModel/"+model_id+">"
    query = """
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
            ?s <kgnet:GMLModel/id> """ + model_id_format + """.
            }
        }        
        union
        {
            select ?m as ?s ?p as ?p  ?o as ?o
            from <http://kgnet/>
            where
            {
            ?m ?p ?o.
            ?m <kgnet:GMLModel/id> """ + model_id_format + """.
            }
        }        
    }
    limit 100

    """
    # print(query)
    res_df = kgmeta_govener.executeSparqlquery(query)
    res_df = res_df.applymap(lambda x: x.strip('"'))
    # dict_params['model']['modelID'] = str(res_df[res_df['p'] == Constants.GNN_SubG_Parms.modelId]['o'].item()).split('/')[-1] + '.model'
    # dict_params['model']['GNNMethod'] = str(res_df[res_df['p'] == Constants.GNN_KG_HParms.GNN_Method]['o'].item())
    # dict_params['model']['embSize'] = int(res_df[res_df['p'] == Constants.GNN_KG_HParms.Emb_size]['o'].item())
    # dict_params['model']['hiddenChannels'] = int(res_df[res_df['p'] == Constants.GNN_KG_HParms.HiddenChannels]['o'].item())
    # dict_params['model']['Num_Layers'] = int(res_df[res_df['p'] == Constants.GNN_KG_HParms.Num_Layers]['o'].item())
    # dict_params['model']['TaskType'] = str (res_df[res_df['p'] == Constants.GNN_SubG_Parms.taskType]['o'].item())
    #
    # dict_params['subG']['targetEdge'] = str(res_df[res_df['p'] == Constants.GNN_SubG_Parms.targetEdge]['o'].item())
    # dict_params['subG']['graphPrefix'] = str(res_df[res_df['p'] == Constants.GNN_SubG_Parms.prefix]['o'].item())

    try:
        dict_params['model']['GNNMethod'] = str(res_df[res_df['p'] == Constants.GNN_KG_HParms.GNN_Method]['o'].item())
    except Exception as e:
        print("Error processing GNNMethod:", e)

    try:
        dict_params['model']['embSize'] = int(res_df[res_df['p'] == Constants.GNN_KG_HParms.Emb_size]['o'].item())
    except Exception as e:
        print("Error processing embSize:", e)

    try:
        dict_params['model']['hiddenChannels'] = int(
            res_df[res_df['p'] == Constants.GNN_KG_HParms.HiddenChannels]['o'].item())
    except Exception as e:
        print("Error processing hiddenChannels:", e)

    try:
        dict_params['model']['Num_Layers'] = int(res_df[res_df['p'] == Constants.GNN_KG_HParms.Num_Layers]['o'].item())
    except Exception as e:
        print("Error processing Num_Layers:", e)

    try:
        dict_params['model']['taskType'] = str(res_df[res_df['p'] == Constants.GNN_SubG_Parms.taskType]['o'].item())
    except Exception as e:
        print("Error processing TaskType:", e)

    try:
        dict_params['subG']['targetEdge'] = str(res_df[res_df['p'] == Constants.GNN_SubG_Parms.targetEdge]['o'].item())
    except Exception as e:
        print("Error processing targetEdge:", e)

    try:
        dict_params['subG']['graphPrefix'] = str(res_df[res_df['p'] == Constants.GNN_SubG_Parms.prefix]['o'].item())
    except Exception as e:
        print("Error processing graphPrefix:", e)

    return dict_params


def topKpred(pred_df, K=None):
    if K == 1:
        return pred_df.set_index(0)[1].to_dict()

    result_dict = {}
    for key, value in zip(pred_df[0], pred_df[1]):
        if key not in result_dict:
            result_dict[key] = []
        if K is None or len(result_dict[key]) < K:
            result_dict[key].append(value)

    for key in result_dict:
        result_dict[key] = str(result_dict[key])

    return result_dict
def generate_subgraph(named_graph_uri, target_rel_uri, dataQuery, sparqlEndpointURL):
    # query = [get_NC_d1h1_query(graph_uri=named_graph_uri, target_rel_uri=target_rel_uri,
    #                            tragetNode_filter_statments=targetNode_filter_statements)]
    query = ''
    for statement in dataQuery:
        query += statement
    kg = KGNET(sparqlEndpointURL)
    subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(query, )
    subgraph_df = subgraph_df.applymap(lambda x: x.strip('"'))
    subgraph_df.to_csv(inference_file, index=None, sep='\t')
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


def filterTargetNodes(predictions = pd.DataFrame(), targetNodesQuery = "", sparqlEndpointURL = "",named_graph_uri = "",apply = True,):
    kg = KGNET(sparqlEndpointURL,KG_NamedGraph_IRI=named_graph_uri)
    targetNodes = kg.KG_sparqlEndpoint.executeSparqlquery(targetNodesQuery, )
    targetNodes = targetNodes.applymap(lambda x: x.strip('"'))['s'].to_dict()
    targetNodes = {v: k for k, v in targetNodes.items()}

    if not apply:
        return targetNodes

    filtered_pred = {key: predictions[key] for key in predictions if key in targetNodes}
    return filtered_pred


def perform_inference(model_id, named_graph_uri, dataQuery, sparqlEndpointURL, targetNodesQuery,topk,demo = True):
    dict_time = {}
    if not os.path.exists(Constants.KGNET_Config.inference_path):
        os.makedirs(Constants.KGNET_Config.inference_path)

    meta_dict = get_MetaData(model_id)
    model_id = 'mid-' + Constants.utils.getIdWithPaddingZeros(model_id) + '.model'
    dataset_name = model_id.replace(".model","")


    ###### IF LINK PREDICTION #######
    if meta_dict['model']['taskType'] == 'kgnet:type/linkPrediction':
        if demo:
            preds = pd.read_csv(os.path.join(Constants.KGNET_Config.inference_path,'authored_by_predictions.tsv'),header=None,sep='\t')
            print("preds.columns=",preds.columns)
            kg = KGNET(sparqlEndpointURL, KG_NamedGraph_IRI=named_graph_uri)
            targetNodes = kg.KG_sparqlEndpoint.executeSparqlquery(targetNodesQuery, )
            print("targetNodes.columns=", targetNodes.columns)
            print("len(targetNodes)=",(len(targetNodes)))
            targetNodes['s']=targetNodes['s'].apply(lambda x:str(x).replace("\"",""))
            preds=preds[preds[0].isin(targetNodes['s'].tolist())]
            print("len(targetNodes)=", (len(targetNodes)))
            return topKpred(preds, topk)
            # return topKpred(preds[:1000],topk)

        targetNodes = list(filterTargetNodes(targetNodesQuery=targetNodesQuery, sparqlEndpointURL=sparqlEndpointURL, named_graph_uri=named_graph_uri, apply=False).keys())

        downloadDataset(dataset_name + '.tsv')

        transform_LP_train_valid_test_subsets(data_path=KGNET.KGNET_Config.inference_path,
                                              ds_name=dataset_name,
                                              target_rel=meta_dict['subG']['targetEdge'])

        dic_results = rgcn_lp(dataset_name=dataset_name,root_path=KGNET.KGNET_Config.inference_path,
                                loadTrainedModel=1,target_rel=meta_dict['subG']['targetEdge'],list_src_nodes=targetNodes,
                              modelID=model_id,K=topk)
        if topk==1:
            dic_results = {k : v[0] for k,v in dic_results.items()}

        return dic_results






    downloadDataset(dataset_name + '.zip')




    time_subgraph = datetime.datetime.now()
    # subgraph = generate_subgraph(named_graph_uri=named_graph_uri,
    #                              target_rel_uri=meta_dict['subG']['targetEdge'],
    #                              dataQuery=dataQuery,
    #                              sparqlEndpointURL=sparqlEndpointURL)
    dict_time['subgraph_generation_time'] = (datetime.datetime.now() - time_subgraph).total_seconds()
    # rel_types = get_rel_types(named_graph_uri=named_graph_uri,
    #                           graphPrefix=meta_dict['subG']['graphPrefix'],
    #                           sparqlEndpointURL=sparqlEndpointURL)

    time_dataTransform = datetime.datetime.now()
    # data_dict = inference_transform_tsv_to_PYG(dataset_name='inference',
    #                                            dataset_name_csv='inference',
    #                                            dataset_types=rel_types,
    #                                            target_rel=meta_dict['subG']['targetEdge'],
    #                                            output_root_path=Constants.KGNET_Config.inference_path,
    #                                            # Header_row=False
    #                                            )

    # dic_results = graphSaint(dataset_name='mid-0000092_', root_path=Constants.KGNET_Config.inference_path,
    #                          loadTrainedModel=1, target_mapping=data_dict['target_mapping'],
    #                          modelID=model_id)
    # return dic_results['y_pred']

    dict_time['transformation_time'] = (datetime.datetime.now() - time_dataTransform).total_seconds()

    time_download = datetime.datetime.now()
    if downloadModel(model_id) and downloadModel(model_id.replace('.model', '.param')):
        dict_time['model_download_time'] = (datetime.datetime.now() - time_download).total_seconds()
        print('Downloaded model successfully!')

        time_inference = datetime.datetime.now()
        if meta_dict['model']['GNNMethod'] == Constants.GNN_Methods.Graph_SAINT:
            dic_results = graphSaint(dataset_name=dataset_name, root_path=Constants.KGNET_Config.inference_path,
                                     loadTrainedModel=1, #target_mapping=data_dict['target_mapping'],
                                     modelID=model_id)



        elif meta_dict['model']['GNNMethod'] == Constants.GNN_Methods.ShaDowGNN:
            dic_results = graphShadowSaint(dataset_name='inference', root_path=Constants.KGNET_Config.inference_path,
                                           loadTrainedModel=1, #target_mapping=data_dict['target_mapping'],
                                           modelID=model_id)

    else:
        return {'error': 'Model not found'}
    dict_time['inference_time'] = (datetime.datetime.now() - time_inference).total_seconds()
    dic_results['y_pred'] = filterTargetNodes(dic_results['y_pred'],
                                              targetNodesQuery,
                                              sparqlEndpointURL,
                                              named_graph_uri)
    # shutil.rmtree(Constants.KGNET_Config.inference_path)

    # dict_time.update(dic_results['y_pred'])
    # dic_results['y_pred'] = dic_results['y_pred'].update({'Inference_Time':dict_time})
    dict_inference = dic_results['y_pred']
    dict_time_dic = {"Inference_Times": dict_time}
    dict_inference.update(dict_time_dic)
    return dict_inference
    print('pass!')
