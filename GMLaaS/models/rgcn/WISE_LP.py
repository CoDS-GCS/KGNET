""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""
import pickle
import sys
GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GMLaaS/models/rgcn"
sys.path.insert(0,GMLaaS_models_path)
from Constants import *
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm
from rel_link_pred_dataset import RelLinkPredDataset
from torch_geometric.nn import GAE, RGCNConv
from resource import *
import datetime
import os
import json
import pandas as pd
import os.path as osp
from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor
from SparqlMLaasService.TaskSampler.TOSG_Extraction_NC import get_d1h1_TargetListquery
import shutil
from GMLaaS.DataTransform.Transform_LP_Dataset import transform_LP_train_valid_test_subsets
import pickle
def gen_model_name(dataset_name='',GNN_Method=''):
    return dataset_name
def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)

class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = Parameter(torch.Tensor(num_nodes, hidden_channels))
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=64,)

        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=64,)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.Tensor(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)



def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
    return neg_edge_index


def train():
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.edge_index, data.edge_type)

    pos_out = model.decode(z, data.train_edge_index, data.train_edge_type)

    neg_edge_index = negative_sampling(data.train_edge_index, data.num_nodes)
    neg_out = model.decode(z, neg_edge_index, data.train_edge_type)

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data.edge_index, data.edge_type)

    valid_mrr,valid_hits_10 = compute_mrr(z, data.valid_edge_index, data.valid_edge_type)
    test_mrr,test_hits_10 = compute_mrr(z, data.test_edge_index, data.test_edge_type)

    return valid_mrr,valid_hits_10 , test_mrr,test_hits_10


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr(z, edge_index, edge_type):
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)
    ranks = torch.tensor(ranks, dtype=torch.float)
    hits=[10]
    for hit in hits:
        avg_count = torch.mean((ranks <= hit).float())
        hits_10 = avg_count.item()
        print("Hits (filtered) @ {}: {:.6f}".format(hit, hits_10))


    return (1. / torch.tensor(ranks, dtype=torch.float)).mean(),hits_10

def execute_query_v2(batch, inference_file,kg,):

    subgraph_df = kg.KG_sparqlEndpoint.executeSparqlquery(batch)#kg.KG_sparqlEndpoint.execute_sparql_multithreads([query], inference_file)
    if len(subgraph_df.columns) != 3:
        print(subgraph_df)
        raise AssertionError
    subgraph_df = subgraph_df.applymap(lambda x: x.strip('"'))
    if os.path.exists(inference_file):
        subgraph_df.to_csv(inference_file, header=None, index=None, sep='\t', mode='a')
    else:
        subgraph_df.to_csv(inference_file, index=None, sep='\t', mode='a')

def batch_tosa_v2 (targetNodesList,inference_file,graph_uri,kg,BATCH_SIZE=2000):
    def batch_generator():
        ptr = 0
        while ptr < len(targetNodesList):
            yield targetNodesList[ptr:ptr + BATCH_SIZE]
            ptr += BATCH_SIZE

    queries = []
    for q in batch_generator():
        target_lst = ['<' + target + '>' for target in q]
        queries.extend(get_d1h1_TargetListquery(graph_uri=graph_uri, target_lst=target_lst))
    if len(targetNodesList) > BATCH_SIZE:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute_query_v2, batch, inference_file, kg, graph_uri) for batch in
                       queries]
            for future in tqdm(futures, desc="Downloading Raw Graph", unit="subgraphs"):
                # future.result()
                pass
    else:
        [execute_query_v2(batch,inference_file,kg) for batch in queries]
def generate_inference_subgraph(master_ds_name, graph_uri='',targetNodesList = [],labelNode = None,targetNodeType=None,
                                target_rel_uri='https://dblp.org/rdf/schema#publishedIn',
                                ds_types = '',
                                sparqlEndpointURL=KGNET_Config.KGMeta_endpoint_url,
                                output_file='inference_subG'):


    global download_end_time
    time_ALL_start = datetime.datetime.now()

    inference_root = os.path.join(KGNET_Config.inference_path, output_file)
    inference_tsv = os.path.join(KGNET_Config.inference_path, output_file + '.tsv')

    download_start_time = datetime.datetime.now()
    if os.path.exists(inference_tsv):
        os.remove(inference_tsv)

    if os.path.exists(inference_root):
        shutil.rmtree(inference_root)

    if os.path.exists(inference_root + '.zip'):
        os.remove(inference_root + '.zip')

    from KGNET import KGNET
    kg = KGNET(sparqlEndpointURL, KG_Prefix='http://kgnet/') #TODO: Parameterize

    batch_tosa_v2(targetNodesList,inference_file=inference_tsv,graph_uri=graph_uri,kg=kg)
    download_end_time = (datetime.datetime.now() - download_start_time).total_seconds()

    path_master_ds_root = os.path.join(KGNET_Config.datasets_output_path, master_ds_name)
    path_master_entities = os.path.join(path_master_ds_root,'entities.dict')
    path_master_relations = os.path.join(path_master_ds_root,'relations.dict')
    path_inference_root = os.path.join(KGNET_Config.inference_path, output_file)
    path_inference_entities = os.path.join(path_inference_root,'entities.dict')
    path_inference_relations = os.path.join(path_inference_root,'relations.dict')

    if not os.path.exists(path_master_ds_root):
        if os.path.exists(path_master_ds_root+'.zip'):
            shutil.unpack_archive(path_master_ds_root+'.zip', KGNET_Config.datasets_output_path)
        else:
            raise Exception('No Master Graph at {} '.format(path_master_ds_root))

    transform_LP_train_valid_test_subsets(KGNET_Config.inference_path,
                                  output_file,
                                  target_rel=target_rel_uri,
                                  split_rel=None,
                                  containHeader = True,
                                  inference=True)

    master_ent_df = pd.read_csv(path_master_entities,header=None,sep='\t',names=['ent_idx','ent_name'],dtype={'ent_idx':'int64'})
    master_rel_df = pd.read_csv(path_master_relations,header=None,sep='\t',names=['ent_idx','ent_name'],dtype={'ent_idx':'int64'})
    inf_ent_df = pd.read_csv(path_inference_entities,header=None,sep='\t',names=['ent_idx','ent_name'],dtype={'ent_idx':'int64'})
    inf_rel_df = pd.read_csv(path_inference_relations,header=None,sep='\t',names=['ent_idx','ent_name'],dtype={'ent_idx':'int64'})
    print('here')
    idx_cols = ['ent_idx_inf','ent_idx_orig']
    output_cols = ['ent_idx_orig','ent_name']
    mapping_ent = pd.merge(inf_ent_df,master_ent_df,on='ent_name',how='left',suffixes=('_inf','_orig'))
    mapping_rel = pd.merge(inf_rel_df,master_rel_df,on='ent_name',how='left',suffixes=('_inf','_orig'))
  #mapping_ent.dropna().to_csv()
    mapping_ent = mapping_ent.dropna()
    mapping_rel = mapping_rel.dropna()
    mapping_ent[idx_cols] = mapping_ent[idx_cols].astype('int64')
    mapping_rel[idx_cols] = mapping_rel[idx_cols].astype('int64')

    mapping_ent[output_cols].to_csv(path_inference_entities,header=None,index=None,sep='\t')
    mapping_rel[output_cols].to_csv(path_inference_relations, header=None, index=None, sep='\t')
    return output_file


def rgcn_lp(dataset_name,
            root_path=KGNET_Config.datasets_output_path,
            epochs=3,val_interval=2,
            hidden_channels=10,
            loadTrainedModel=0,
            target_rel = '',
            list_src_nodes = [],
            K = 1,modelID = ''
            ):

    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results = {}
    print('loading dataset..')
    start_data_t = datetime.datetime.now()
    dataset = RelLinkPredDataset(root_path, dataset_name)
    global data,model,optimizer
    data = dataset[0]
    load_data_t = str((datetime.datetime.now()-start_data_t).total_seconds())
    dict_results['Sampling_Time'] = load_data_t
    print(f'dataset loaded at {load_data_t}')




    if loadTrainedModel == 1:
        print('Initializing model...')
        path_model_param = os.path.join(KGNET_Config.trained_model_path, modelID.replace(".model",".param"))
        with open(path_model_param,'rb') as f:
            dict_model_param = pickle.load(f)

        model = GAE(
            RGCNEncoder(dict_model_param['data.num_nodes'], hidden_channels=dict_model_param['hidden_channels'],
                        num_relations=dict_model_param['dataset.num_relations']),
            DistMultDecoder(dict_model_param['dataset.num_relations'] // 2, hidden_channels=dict_model_param['hidden_channels']),
        )
        print('Model Initialized!')

        ###### LOAD ENTITIES AND REL DICT FROM "MASTER GRAPH" ##########
        path_master =osp.join(KGNET_Config.datasets_output_path,modelID.replace('.model',''))
        entities_df = pd.read_csv(osp.join(path_master, 'entities.dict'), header=None, sep="\t")
        entities_dict = dict(zip(entities_df[1], entities_df[0]))
        rev_entities_dict = {v:k for k,v in entities_dict.items()}

        relations_df = pd.read_csv(osp.join(path_master, 'relations.dict'), header=None, sep="\t")
        relations_dict = dict(zip(relations_df[1], relations_df[0]))
        if target_rel not in relations_dict:
            return {'error':['Unseen relation']}
        target_rel_ID = relations_dict[target_rel]

        ####### LOAD MODEL STATE ##############
        trained_model_path = os.path.join(KGNET_Config.trained_model_path,modelID)
        model.load_state_dict(torch.load(trained_model_path)); print(f'LOADED PRE-TRAINED MODEL {modelID}')
        model.eval()
        with torch.no_grad():
            edge_index = data.edge_index
            edge_type = data.edge_type
            y_pred = {}
            z = model.encode(data.edge_index, data.edge_type)

            for _src in list_src_nodes:
                if _src not in entities_dict:
                    y_pred[_src] = ['None']
                    continue
                src = entities_dict[_src]
                (_, dst), rel = edge_index[:, target_rel_ID], edge_type[target_rel_ID]

                # tail = torch.arange(data.num_nodes)
                tail = torch.arange(dict_model_param['data.num_nodes'])
                tail = torch.cat([torch.tensor([dst]), tail])
                head = torch.full_like(tail, fill_value=src)
                eval_edge_index = torch.stack([head, tail], dim=0)
                eval_edge_type = torch.full_like(tail, fill_value=rel)

                out = model.decode(z, eval_edge_index, eval_edge_type)
                y_pred[_src] = [rev_entities_dict[i] for i in out.topk(K).indices.tolist() if i in rev_entities_dict]
                # print(out)
        return y_pred


    print('Initializing model...')
    model = GAE(
        RGCNEncoder(data.num_nodes, hidden_channels=hidden_channels,
                    num_relations=dataset.num_relations),
        DistMultDecoder(dataset.num_relations // 2, hidden_channels=hidden_channels),
    )
    print('Model Initialized!')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    start_train_t = datetime.datetime.now()
    best_test_mrr = 0
    best_valid_mrr = 0
    best_test_hits = 0
    best_valid_hits = 0
    for epoch in tqdm (range(1, epochs),desc='Training Epochs'):
        # print('Starting training .. ')
        loss = train()
        print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
        if (epoch % val_interval) == 0:
            valid_mrr,valid_hits_10, test_mrr,test_hits_10 = test()
            print(f'Val MRR: {valid_mrr:.4f}, Val Hits@10: {valid_hits_10:.4f}, Test MRR: {test_mrr:.4f}, Test Hits@10: {test_hits_10:.4f}')
            if test_mrr > best_test_mrr:
                best_test_mrr = test_mrr
                best_test_hits = test_hits_10

            if valid_mrr > best_valid_mrr:
                best_valid_mrr = valid_mrr
                best_valid_hits = valid_hits_10
            print(getrusage(RUSAGE_SELF))
    model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results['Results'] = {'Best_MRR' : best_test_mrr.item(),'Best_Hits@10' : best_test_hits}
    print(getrusage(RUSAGE_SELF))
    end_train_t = datetime.datetime.now()
    total_train_t = str((end_train_t - start_train_t).total_seconds())
    total_time = str((start_data_t - end_train_t).total_seconds())
    dict_results['dataset_name'] = dataset_name
    model_name = gen_model_name(dataset_name)
    dict_results['model_name'] = model_name
    dict_results['Train_Time'] = total_train_t
    dict_results['Total_Time'] = total_time
    dict_results["Model_Parameters_Count"] = sum(p.numel() for p in model.parameters())
    dict_results["Model_Trainable_Paramters_Count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dict_results["init_ru_maxrss"] = init_ru_maxrss
    dict_results["model_ru_maxrss"] = model_loaded_ru_maxrss
    dict_results["model_trained_ru_maxrss"] = model_trained_ru_maxrss
    dict_results["Highest_Test_MRR"] = best_test_mrr.item()
    dict_results["Highest_Valid_MRR"] = best_valid_mrr.item()
    dict_results["Highest_Test_Hits@10"] = str(best_test_hits)
    dict_results["Highest_Valid_Hits@10"] = str(best_valid_hits)
    dict_results["Final_Test_MRR"] = test_mrr.item()
    dict_results["Final_Valid_MRR"] = valid_mrr.item()
    dict_results["Final_Test_Hits@10"] = str(test_hits_10)
    dict_results["Final_Valid_Hits@10"] = str(valid_hits_10)
    gnn_hyper_params_dict = { "hidden_channels": hidden_channels,
                               "epochs": epochs,
                              #"runs": runs,
                              #"batch_size": batch_size,
                            #"walk_length": walk_length, "num_steps": num_steps, "emb_size": emb_size
                             }
    dict_results["gnn_hyper_params"] = gnn_hyper_params_dict

    ### DEBUG ###
    for key in dict_results.keys():
        print(f'{key} {type(dict_results[key])}')

    logs_path = os.path.join(root_path, 'logs')
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    model_path = KGNET_Config.trained_model_path
    create_dir([logs_path, model_path])
    with open(os.path.join(logs_path, model_name + '_log.json'), "w") as outfile:
        json.dump(dict_results, outfile)
    torch.save(model.state_dict(), os.path.join(model_path, model_name) + ".model")
    dict_model_param = {}
    dict_model_param['hidden_channels'] = 128
    dict_model_param['data.num_nodes'] = data.num_nodes
    dict_model_param['dataset.num_relations'] = dataset.num_relations
    with open(os.path.join(model_path, model_name+ ".param"),'wb') as f:
        pickle.dump(dict_model_param,f)

    return dict_results
    # print("Total Time Sec=", (end_t - start_t).total_seconds())


if __name__ == '__main__':
    dataset_name = r'mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965'
    root_path = os.path.join(KGNET_Config.datasets_output_path,)
    target_rel = r'http://www.wikidata.org/entity/P101'
    list_src_nodes = ['http://www.wikidata.org/entity/Q5484233',
                      'http://www.wikidata.org/entity/Q16853882',
                      'http://www.wikidata.org/entity/Q777117']
    K = 2
    modelID = r'mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965.model'
    # dataset_name = generate_inference_subgraph(master_ds_name=dataset_name,
    #                             graph_uri='http://wikikg-v2',
    #                             targetNodesList=list_src_nodes,
    #                             target_rel_uri=target_rel,
    #                             )
    # result = rgcn_lp(dataset_name,root_path=KGNET_Config.inference_path,target_rel=target_rel,loadTrainedModel=1,list_src_nodes=list_src_nodes,modelID=modelID,hidden_channels=128)

    result = rgcn_lp(dataset_name,root_path,target_rel=target_rel,loadTrainedModel=1,list_src_nodes=list_src_nodes,modelID=modelID,epochs=21,val_interval=10,hidden_channels=128)
    print(result)
