import shutil
import Constants
from GMLaaS.models.graph_saint_Shadow_KGTOSA import graphShadowSaint
from GMLaaS.models.graph_saint_KGTOSA import graphSaint
from GMLaaS.models.rgcn.rgcn_link_pred import rgcn_lp
from GMLaaS.models.MorsE.main import morse
from GMLaaS.DataTransform.Transform_LP_Dataset import transform_LP_train_valid_test_subsets
from RDFEngineManager.sparqlEndpoint import sparqlEndpoint
from model_manager import downloadModel,downloadDataset
import datetime
import pandas as pd
from Constants import *

# output_path = Constants.KGNET_Config.inference_path
inference_file = os.path.join(Constants.KGNET_Config.inference_path, 'inference.tsv')


def delete_inference_cache(inference_path=Constants.KGNET_Config.inference_path):
    for entry in os.listdir(inference_path):
        entry_path = os.path.join(inference_path, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)


def get_MetaData(model_id,kg_endpoint):
    dict_params = {'model': {}, 'subG': {}}
    model_id_format = str(model_id)  # "<kgnet:GMLModel/"+model_id+">"
    query = """
    prefix kgnet:<http://kgnet/>
    select *
    from <http://kgnet/>
    where
    {
        {
            select (?t as ?s) ?p   ?o
            where
            {
            ?m <kgnet:GMLModel/id> ?mid.
            filter(str(?mid)="?mid_p").
            ?t ?tp ?m.
            ?t ?p ?o.
            }
        }        
        union
        {
            select (?m as ?s) ?p  ?o 
            where
            {
            ?m <kgnet:GMLModel/id> ?mid.
            filter(str(?mid)="?mid_p").
            ?m ?p ?o.
            }
        }        
    }
    limit 1000
    """
    query=query.replace("?mid_p",model_id_format)
    print("query=",query)
    res_df = kg_endpoint.executeSparqlquery(query)
    res_df["o"]=res_df["o"].apply(lambda x: str(x)[1:-1] if str(x).startswith("\"") or str(x).startswith("<") else x)
    res_df["p"]=res_df["p"].apply(lambda x: str(x)[1:-1] if  str(x).startswith("<") else x)
    res_df = res_df.applymap(lambda x: x.strip('"'))
    # print("res_df=",res_df)
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


def topKpred(pred_df, RDFEngine,K=None):
    if K == 1:
        return pred_df.set_index(0)[1].to_dict()

    result_dict = {}
    for key, value in zip(pred_df[0], pred_df[1]):
        if key not in result_dict:
            result_dict[key] = []
        if K is None or len(result_dict[key]) < K:
            result_dict[key].append(str(value))

    if RDFEngine==Constants.RDFEngine.stardog:
        for key in result_dict:
            result_dict[key] = str(result_dict[key]).replace("'","\"")
    else:
        for key in result_dict:
            result_dict[key] = str(result_dict[key])

    return result_dict
def generate_subgraph(kg_endpoint,dataQuery):
    # query = [get_NC_d1h1_query(graph_uri=named_graph_uri, target_rel_uri=target_rel_uri,
    #                            tragetNode_filter_statments=targetNode_filter_statements)]
    query = ''
    for statement in dataQuery:
        query += statement
    subgraph_df = kg_endpoint.executeSparqlquery(query, )
    subgraph_df = subgraph_df.applymap(lambda x: str(x)[1:-1] if str(x).startswith("\"") or str(x).startswith("<") else str(x))
    subgraph_df.to_csv(inference_file, index=None, sep='\t')
    return inference_file

def filterTargetNodes(kg_endpoint,predictions = pd.DataFrame(), targetNodesQuery = None,targetNodesList=None,TOSG_Pattern=TOSG_Patterns.d1h1,graph_uri=None,apply = True,):
    if targetNodesQuery is not None:
        print("targetNodesQuery=",targetNodesQuery)
        targetNodes = kg_endpoint.executeSparqlquery(targetNodesQuery)
        print("targetNodes.columns", targetNodes.columns)
        targetNodes["s"] = targetNodes["s"].apply(
            lambda x: str(x)[1:-1] if str(x).startswith("\"") or str(x).startswith("<") else str(x))
        # print("targetNodes.columns=",targetNodes.columns)
        targetNodes = targetNodes.applymap(lambda x: x.strip('"'))['s'].to_dict()
        targetNodes = {v: k for k, v in targetNodes.items()}
    elif targetNodesList is not None and len(targetNodesList)>0:
        targetNodes=[ str(elem).strip()[1:-1] if str(elem).strip().startswith("<") else str(elem).strip() for elem in targetNodesList]
    else:
        return None
    if not apply:
        return targetNodes

    filtered_pred = {key: predictions[key] for key in predictions if key in targetNodes}
    return filtered_pred


def perform_inference(model_id, named_graph_uri, dataQuery, sparqlEndpointURL, targetNodesQuery,topk,RDFEngine,demo = True,targetNodesList=None,TOSG_Pattern=TOSG_Patterns.d1h1):
    if RDFEngine:
        kg_endpoint = sparqlEndpoint(sparqlEndpointURL,RDFEngine=RDFEngine)
    else:
        kg_endpoint = sparqlEndpoint(sparqlEndpointURL)
    dict_time = {}
    if not os.path.exists(Constants.KGNET_Config.inference_path):
        os.makedirs(Constants.KGNET_Config.inference_path)
    meta_dict = get_MetaData(model_id,kg_endpoint)
    if Constants.utils.is_number(model_id):
        model_id = 'mid-' + Constants.utils.getIdWithPaddingZeros(model_id) + '.model'
    else:
        model_id = 'mid-' +model_id + '.model'
    dataset_name = model_id.replace(".model","")
    ###### IF LINK PREDICTION #######
    print("model meta_dict['model']=",meta_dict['model'])
    if meta_dict['model']['taskType'] == 'kgnet:type/linkPrediction':
        if demo:
            preds = pd.read_csv(os.path.join(Constants.KGNET_Config.inference_path,'authored_by_predictions.tsv'),header=None,sep='\t')
            # preds.columns=["s","o"]
            # print("preds.columns=",preds.columns)
            targetNodes = kg_endpoint.executeSparqlquery(targetNodesQuery)
            print("targetNodes.columns=", targetNodes.columns)
            print("len(targetNodes)=",(len(targetNodes)))
            targetNodes['s']=targetNodes['s'].apply(lambda x: str(x)[1:-1] if str(x).startswith("\"") or str(x).startswith("<") else str(x))
            preds=preds[preds[0].isin(targetNodes['s'].tolist())]
            #print("len(targetNodes)=", (len(targetNodes)))
            #print("preds=", preds)
            return topKpred(preds, RDFEngine,topk)
            # return topKpred(preds[:1000],topk)
        targetNodes = list(filterTargetNodes(kg_endpoint,targetNodesQuery=targetNodesQuery,apply=False).keys())
        if Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.remoteFileStore:
            downloadDataset(dataset_name + '.tsv')
        elif Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.S3:
            Constants.utils.DownloadFileFromS3(dataset_name + '.tsv', to_filepath=os.path.join(Constants.KGNET_Config.inference_path, dataset_name)+".tsv", file_type="metadata")


        transform_LP_train_valid_test_subsets(data_path=Constants.KGNET_Config.inference_path,
                                              ds_name=dataset_name,
                                              target_rel=meta_dict['subG']['targetEdge'])
        if meta_dict['model']['GNNMethod'] == Constants.GNN_Methods.MorsE:
            dic_results = morse(dataset_name=dataset_name,root_path=Constants.KGNET_Config.inference_path,modelID=model_id,)
        dic_results = rgcn_lp(dataset_name=dataset_name,root_path=Constants.KGNET_Config.inference_path,
                                loadTrainedModel=1,target_rel=meta_dict['subG']['targetEdge'],list_src_nodes=targetNodes,
                              modelID=model_id,K=topk)
        if topk==1:
            dic_results = {k : v[0] for k,v in dic_results.items()}

        return dic_results
    ########################## NC ##############################################
    if Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.remoteFileStore:
        downloadDataset(dataset_name + '.zip')
    elif Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.S3:
        Constants.utils.DownloadFileFromS3(dataset_name + '.zip',to_filepath=os.path.join(Constants.KGNET_Config.inference_path,dataset_name) + ".zip", file_type="metadata")
        Constants.utils.DownloadFileFromS3(dataset_name + '.param',to_filepath=os.path.join(Constants.KGNET_Config.inference_path,dataset_name) + ".param", file_type="metadata")
    elif Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.localfile:
        shutil.copyfile(os.path.join(Constants.KGNET_Config.datasets_output_path,dataset_name)+".zip", os.path.join(Constants.KGNET_Config.inference_path,dataset_name)+".zip")

    time_subgraph = datetime.datetime.now()
    # subgraph = generate_subgraph(named_graph_uri=named_graph_uri,
    #                              target_rel_uri=meta_dict['subG']['targetEdge'],
    #                              dataQuery=dataQuery,
    #                              sparqlEndpointURL=sparqlEndpointURL)
    dict_time['subgraph_generation_time'] = (datetime.datetime.now() - time_subgraph).total_seconds()
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
    downloaded=False
    # model_id = model_id+'_wise'
    filepath = os.path.join(Constants.KGNET_Config.trained_model_path, model_id)
    # emb_path = os.path.join(Constants.KGNET_Config.trained_model_path,'emb_store', model_id)
    if Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.localfile:
        if os.path.exists(filepath) and os.path.exists(filepath.replace('.model', '.param')):
            downloaded = True
    if Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.remoteFileStore:
        downloaded=downloadModel(model_id) and downloadModel(model_id.replace('.model', '.param'))
    elif Constants.KGNET_Config.fileStorageType == Constants.FileStorageType.S3:
        downloaded = Constants.utils.DownloadFileFromS3(model_id.replace('.model', ''),to_filepath=filepath) and \
                     Constants.utils.DownloadFileFromS3(model_id.replace('.model', '.param'),to_filepath=filepath.replace('.model', '.param'),file_type="metadata")

    if downloaded:
        dict_time['model_download_time'] = (datetime.datetime.now() - time_download).total_seconds()
        print('Downloaded model successfully!')

        time_inference = datetime.datetime.now()
        if meta_dict['model']['GNNMethod'] == Constants.GNN_Methods.Graph_SAINT:
            dic_results = graphSaint(dataset_name=dataset_name, root_path=Constants.KGNET_Config.inference_path,
                                     loadTrainedModel=1, #target_mapping=data_dict['target_mapping'],
                                     modelID=model_id,emb_size=meta_dict['model']['embSize'])



        elif meta_dict['model']['GNNMethod'] == Constants.GNN_Methods.ShaDowGNN:
            dic_results = graphShadowSaint(dataset_name='inference', root_path=Constants.KGNET_Config.inference_path,
                                           loadTrainedModel=1, #target_mapping=data_dict['target_mapping'],
                                           modelID=model_id,emb_size=meta_dict['model']['embSize'])

    else:
        return {'error': 'Model not found'}
    dict_time['inference_time'] = (datetime.datetime.now() - time_inference).total_seconds()
    dic_results['y_pred'] = filterTargetNodes(kg_endpoint,predictions=dic_results['y_pred'],targetNodesQuery=targetNodesQuery,targetNodesList=targetNodesList,graph_uri=named_graph_uri)
    # shutil.rmtree(Constants.KGNET_Config.inference_path)
    # dict_time.update(dic_results['y_pred'])
    # dic_results['y_pred'] = dic_results['y_pred'].update({'Inference_Time':dict_time})
    dict_inference = dic_results['y_pred']
    dict_time_dic = {"Inference_Times": dict_time}
    dict_inference.update(dict_time_dic)
    return dict_inference
    print('pass!')
