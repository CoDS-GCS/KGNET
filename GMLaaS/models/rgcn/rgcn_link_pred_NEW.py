""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""
import sys
GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GMLaaS/models/rgcn"
sys.path.insert(0,GMLaaS_models_path)
from Constants import *
#import argparse
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm
from rel_link_pred_dataset import RelLinkPredDataset
#from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import GAE, RGCNConv
from resource import *
import datetime
import os
import json
import pandas as pd
import os.path as osp
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
from utils import uniform,load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
import numpy as np
#from utils import calc_mrr
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'RLPD')
#dataset = RelLinkPredDataset(path, 'FB15k-237')


def gen_model_name(dataset_name='',GNN_Method=''):
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # return dataset_name+'_'+model_name+'_'+timestamp
    return dataset_name
def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)



class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout,dim_size = 100):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, dim_size)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, dim_size))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            dim_size, dim_size, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            dim_size, dim_size, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)

        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)



def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
    return neg_edge_index

def train(train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations, negative_sample)

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding)

    return loss

@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


def valid(valid_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr,hits = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, hits=[1, 3, 10])

    return mrr,hits

def test(test_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr,hits = calc_mrr(entity_embedding, model.relation_embedding, test_triplets, all_triplets, hits=[1, 3, 10])

    return mrr,hits

def rgcn_lp(dataset_name,
            root_path=KGNET_Config.datasets_output_path,
            epochs=3,val_interval=2,
            hidden_channels=10,batch_size=-1,runs=1,
            emb_size=128,walk_length = 2, num_steps=2,
            loadTrainedModel=0,
            target_rel = '',
            list_src_nodes = [],
            K = 1,modelID = ''
            ):

    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results = {}
    print('loading dataset..')

    n_bases = 4 #TODO
    dropout = 0.3 #TODO
    use_cuda = False #TODO
    graph_batch_size = 30000
    negative_sample = 1
    regularization = 1e-2
    grad_norm = 1.0
    graph_split_size = 0.5
    best_mrr = 0


    start_data_t = datetime.datetime.now()
    # dataset = RelLinkPredDataset(root_path, dataset_name)
    # global data,model,optimizer
    # data = dataset[0]
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data(os.path.join(root_path,dataset_name))
    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))
    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triplets)
    valid_triplets = torch.LongTensor(valid_triplets)
    test_triplets = torch.LongTensor(test_triplets)
    load_data_t = str((datetime.datetime.now()-start_data_t).total_seconds())
    dict_results['Sampling_Time'] = load_data_t
    print(f'dataset loaded at {load_data_t}')

    print('Initializing model...')
    # model = GAE(
    #     RGCNEncoder(data.num_nodes, hidden_channels=hidden_channels,
    #                 num_relations=dataset.num_relations),
    #     DistMultDecoder(dataset.num_relations // 2, hidden_channels=hidden_channels),
    # )

    model = RGCN(len(entity2id), len(relation2id), num_bases=n_bases, dropout=dropout,dim_size=hidden_channels)

    print('Model Initialized!')


    if loadTrainedModel == 1:

        ###### LOAD ENTITIES AND REL DICT ##########
        entities_df = pd.read_csv(osp.join(dataset.raw_dir, 'entities.dict'), header=None, sep="\t")
        entities_dict = dict(zip(entities_df[1], entities_df[0]))
        rev_entities_dict = {v:k for k,v in entities_dict.items()}

        relations_df = pd.read_csv(osp.join(dataset.raw_dir, 'relations.dict'), header=None, sep="\t")
        relations_dict = dict(zip(relations_df[1], relations_df[0]))
        if target_rel not in relations_dict:
            return {'error':['Unseen relation']}
        target_rel_ID = relations_dict[target_rel]

        ####### LOAD MODEL STATE ##############
        trained_model_path = os.path.join(KGNET_Config.trained_model_path,modelID)
        model.load_state_dict(torch.load(trained_model_path)); print(f'LOADED PRE-TRAINED MODEL {modelID}')

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

                tail = torch.arange(data.num_nodes)
                tail = torch.cat([torch.tensor([dst]), tail])
                head = torch.full_like(tail, fill_value=src)
                eval_edge_index = torch.stack([head, tail], dim=0)
                eval_edge_type = torch.full_like(tail, fill_value=rel)

                out = model.decode(z, eval_edge_index, eval_edge_type)
                y_pred[_src] = [rev_entities_dict[i] for i in out.topk(K).indices.tolist() if i in rev_entities_dict]
                # print(out)

        return y_pred



    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    start_train_t = datetime.datetime.now()
    best_test_mrr = 0
    best_valid_mrr = 0
    best_test_hits = 0
    best_valid_hits = 0
    for epoch in tqdm (range(1, epochs),desc='Training Epochs'):
        # print('Starting training .. ')
        model.train()
        loss = train(train_triplets, model, use_cuda, batch_size=graph_batch_size,
                     split_size=graph_split_size,
                     negative_sample=negative_sample, reg_ratio=regularization, num_entities=len(entity2id),
                     num_relations=len(relation2id))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')

        if (epoch % val_interval) == 0:
            if use_cuda:
                model.cpu()

            model.eval()
            valid_mrr,hits_valid = valid(valid_triplets, model, test_graph, all_triplets)
            test_mrr,hits_test = test(test_triplets, model, test_graph, all_triplets)
            if valid_mrr > best_valid_mrr:
                best_mrr = valid_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           f'best_valid_mrr_model_{dataset_name}_{target_rel}.pth')

            if test_mrr > best_test_mrr:
                best_test_mrr = test_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           f'best_valid_mrr_model_{dataset_name}_{target_rel}.pth')

            if hits_test[10] > best_test_hits:
                best_test_hits = hits_test[10]

            if hits_valid[10] > best_valid_hits:
                best_valid_hits = hits_valid[10]

            if use_cuda:
                model.cuda()
            print(getrusage(RUSAGE_SELF))
    model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dict_results['Results'] = {'Best_MRR' : best_mrr,'Best_Hits@10' : best_test_hits}
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
    dict_results["Highest_Test_MRR"] = best_test_mrr#.item()
    dict_results["Highest_Valid_MRR"] = best_valid_mrr#.item()
    dict_results["Highest_Test_Hits@10"] = str(best_test_hits)
    dict_results["Highest_Valid_Hits@10"] = str(best_valid_hits)
    dict_results["Final_Test_MRR"] = test_mrr#.item()
    dict_results["Final_Valid_MRR"] = valid_mrr#.item()
    dict_results["Final_Test_Hits@10"] = str(hits_test[10])#str(test_hits_10)
    dict_results["Final_Valid_Hits@10"] =str(hits_valid[10]) #str(valid_hits_10)
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


    return dict_results
    # print("Total Time Sec=", (end_t - start_t).total_seconds())


if __name__ == '__main__':
    dataset_name = r'Yago3-10_isConnectedTo' # mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965
    root_path = os.path.join(KGNET_Config.datasets_output_path,)
    target_rel = r'isConnectedTo' #http://www.wikidata.org/entity/P101
    list_src_nodes = ['http://www.wikidata.org/entity/Q5484233',
                      'http://www.wikidata.org/entity/Q16853882',
                      'http://www.wikidata.org/entity/Q777117']
    K = 2
    modelID = r'mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965.model'
    result = rgcn_lp(dataset_name,root_path,target_rel=target_rel,loadTrainedModel=0,list_src_nodes=list_src_nodes,modelID=modelID,epochs=11,val_interval=5,hidden_channels=100,
                     )
    print(result)
# rgcn_lp(dataset_name='mid-0000100',
#         root_path=os.path.join(KGNET_Config.inference_path,),
#         loadTrainedModel=1,
#         target_rel = "https://dblp.org/rdf/schema#authoredBy",
#         list_src_nodes =  ['https://dblp.org/rec/conf/ctrsa/Rosie22',
#                             'https://dblp.org/rec/conf/ctrsa/WuX22',
#                             'https://dblp.org/rec/conf/padl/2022'],
#         K = 2,
#         modelID = ''
#         )