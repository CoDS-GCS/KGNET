from copy import copy
from Constants import *
import json
import argparse
import shutil
from Constants import *
import pandas as pd
from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import  GraphSAINTRandomWalkSampler
#from KGTOSA_Samplers import GraphSAINTTaskBaisedRandomWalkSampler,GraphSAINTTaskWeightedRandomWalkSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import sys
import os
import psutil
from pathlib import Path
import pandas as pd
import random
import statistics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import traceback
import sys
# sys.path.insert(0, '/media/hussein/UbuntuData/GithubRepos/KGNET/GMLaaS/models/')
GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GMLaaS/models"
sys.path.insert(0,GMLaaS_models_path)
# print("sys.path=",sys.path)
from ogb.nodeproppred import PygNodePropPredDataset
from evaluater import Evaluator
from custome_pyg_dataset import PygNodePropPredDataset_hsh
from resource import *
from logger import Logger
import faulthandler
faulthandler.enable()
from model import Model
import pickle
import Constants


def print_memory_usage():
    # print("max_mem_GB=",psutil.Process().memory_info().rss / (1024 * 1024*1024))
    # print("get_process_memory=",getrusage(RUSAGE_SELF).ru_maxrss/(1024*1024))
    print('used virtual memory GB:', psutil.virtual_memory().used / (1024.0 ** 3), " percent",
          psutil.virtual_memory().percent)

def gen_model_name(dataset_name,GNN_Method):
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # return dataset_name+'_'+model_name+'_'+timestamp
    return dataset_name
    
def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)
    
class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)

class RGCN(torch.nn.Module,Model):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)

        # paper_count=len(x_dict[2])
        # paper_count = len(x_dict[3])
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb
            # print(key," size=",x_dict[int(key)].size())

        # print(key2int)
        # print("x_dict keys=",x_dict.keys())

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)
            # print(key,adj_t_dict[key].size)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                # print("keys=",keys)
                # print("adj_t=",adj_t)
                # print("key2int[src_key]=",key2int[src_key])
                # print("x_dict[key2int[src_key]]=",x_dict[key2int[src_key]].size())
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                # print("out size=",out.size())
                # print("tmp size=",conv.rel_lins[key2int[keys]](tmp).size())
                ################## fill missed rows hsh############################
                tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                out.add_(tmp)

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict
    def load_data(self,GNN_dataset_name,dic_results,root_path,n_classes,GA_Index=0,to_remove_pedicates = [], to_remove_subject_object =[], to_keep_edge_idx_map = [],include_reverse_edge=True):


        gsaint_start_t = datetime.datetime.now()
        dir_path = "/shared_mnt/MAG_subsets/" + GNN_dataset_name
        try:
            shutil.rmtree(dir_path)
            print(f"Folder Deleted: {dir_path}")
        except OSError as e:
            print("Error Deleting : %s : %s" % (dir_path, e.strerror))

        dataset = PygNodePropPredDataset_hsh(name=GNN_dataset_name, root=root_path,numofClasses=str(n_classes))
        dataset_name = GNN_dataset_name + "_GA_" + str(GA_Index)
        print("dataset_name=", dataset_name)
        dic_results["GNN_Model"] = GNN_Methods.Graph_SAINT
        dic_results["GA_Index"] = GA_Index
        dic_results["to_keep_edge_idx_map"] = to_keep_edge_idx_map
        dic_results["usecase"] = dataset_name

        dic_results["gnn_hyper_params"] = str(args)
        print(getrusage(RUSAGE_SELF))
        start_t = datetime.datetime.now()

        data = dataset[0]
        subject_node = list(data.y_dict.keys())[0]
        split_idx = dataset.get_idx_split()
        end_t = datetime.datetime.now()
        print("dataset init time=", end_t - start_t, " sec.")

        data.node_year_dict = None
        data.edge_reltype_dict = None
        to_remove_rels = []
        for keys, (row, col) in data.edge_index_dict.items():
            if (keys[2] in to_remove_subject_object) or (keys[0] in to_remove_subject_object):
                to_remove_rels.append(keys)

        for keys, (row, col) in data.edge_index_dict.items():
            if (keys[1] in to_remove_pedicates):
                to_remove_rels.append(keys)
                to_remove_rels.append((keys[2], '_inv_' + keys[1], keys[0]))

        for elem in to_remove_rels:
            data.edge_index_dict.pop(elem, None)
            data.edge_reltype.pop(elem, None)

        for key in to_remove_subject_object:
            data.num_nodes_dict.pop(key, None)

        dic_results["data_obj"] = str(data)
        ##############add inverse edges ###################


        if include_reverse_edge:
            edge_index_dict = data.edge_index_dict
            key_lst = list(edge_index_dict.keys())
            for key in key_lst:
                r, c = edge_index_dict[(key[0], key[1], key[2])]
                edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

        out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
        edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

        homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                         node_type=node_type, local_node_idx=local_node_idx,
                         num_nodes=node_type.size(0))

        homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
        homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]

        homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
        homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True


        print("dataset.processed_dir", dataset.processed_dir)
        meta_dict = {'processed_dir':dataset.processed_dir,
                     'edge_index':edge_index,
                     'edge_type':edge_type,
                     'node_type':node_type,
                     'local_node_idx':local_node_idx,
                     'local2global':local2global,
                     'key2int':key2int,
                     'subject_node':subject_node,
                     'num_classes':dataset.num_classes,
                     'edge_index_dict':edge_index_dict}
        return data,homo_data,dic_results,meta_dict

    def sampling_method(self,data,sampler_name=GNN_Samplers.RW,sampler_param={}):
        if sampler_name == GNN_Samplers.BRW:
            raise NotImplementedError

        elif sampler_name == GNN_Samplers.RW:
            batch_size = sampler_param['batch_size']
            walk_length = sampler_param['walk_length']
            num_steps = sampler_param['num_steps']
            sample_coverage = 0
            save_dir = sampler_param['processed_dir']
            sampler = GraphSAINTRandomWalkSampler(data,
                                                  batch_size=batch_size,
                                                  walk_length=walk_length,
                                                  num_steps=num_steps,
                                                  sample_coverage=sample_coverage,
                                                  save_dir=save_dir
                                                  )
            return sampler

        elif sampler_name == GNN_Samplers.PPR:
            raise NotImplementedError

        elif sampler_name == GNN_Samplers.WRW:
            raise NotImplementedError

    def train_epoch(self,model,train_loader,epoch_no,num_steps,batch_size,optimizer,x_dict,device=0):
        model.train()
        pbar = tqdm(total=num_steps * batch_size)
        pbar.set_description(f'Epoch {epoch_no:02d}')

        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                        data.local_node_idx)
            out = out[data.train_mask]
            y = data.y[data.train_mask].squeeze()
            loss = F.nll_loss(out, y)
            # print("loss=",loss)
            loss.backward()
            optimizer.step()
            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            pbar.update(batch_size)

        # pbar.refresh()  # force print final state
        pbar.close()
        # pbar.reset()
        return total_loss / total_examples
    #TODO set gnn hyper parameters and initialize logger
    def train_model(self,device=0,num_layers=2,hidden_channels=64,dropout=0.5,lr=0.005,epochs=2,runs=1,batch_size=2000,walk_length=2,num_steps=10,loadTrainedModel=0,dataset_name="DBLP-Springer-Papers",root_path="../../Datasets/",output_path="./",include_reverse_edge=True,n_classes=1000,emb_size=128,sampler_name=GNN_Samplers.RW):
        dic_results = {}
        data,homo_data,dic_results,meta_dict = self.load_data(dataset_name,dic_results,root_path, n_classes, include_reverse_edge)
        train_loader = self.sampling_method(homo_data,sampler_name,{'batch_size':batch_size,
                                                                    'walk_length':walk_length,
                                                                    'num_steps':num_steps,
                                                                    'save_dir':meta_dict['processed_dir']})

        start_t = datetime.datetime.now()
        #######################intialize random features ###############################
        emb_size = emb_size
        feat = torch.Tensor(data.num_nodes_dict[meta_dict['subject_node']], emb_size)
        torch.nn.init.xavier_uniform_(feat)
        feat_dic = {meta_dict['subject_node']: feat}
        ################################################################
        x_dict = {}
        for key, x in feat_dic.items():
            x_dict[meta_dict['key2int'][key]] = x

        num_nodes_dict = {}
        for key, N in data.num_nodes_dict.items():
            num_nodes_dict[meta_dict['key2int'][key]] = N

        end_t = datetime.datetime.now()
        print("model init time CPU=", end_t - start_t, " sec.")

        device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
        model = RGCN(emb_size, hidden_channels, meta_dict['num_classes'], num_layers,
                     dropout, num_nodes_dict, list(x_dict.keys()),
                     len(meta_dict['edge_index_dict'].keys())).to(device)

        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        print("x_dict=", x_dict.keys())
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        # model_name = dataset_name + "_DBLP_conf_GSAINT_QM.model"
        model_name = gen_model_name(dataset_name, dic_results["GNN_Method"])
        # logger = Logger(runs,gnn_hyper_params_dict)

    def test_model(self):
        pass

    def inference_model(self):
        pass

dic_results = {}
def graphSaint(device=0,num_layers=2,hidden_channels=64,dropout=0.5,
               lr=0.005,epochs=2,runs=1,batch_size=2000,walk_length=2,
               num_steps=10,loadTrainedModel=0,dataset_name="DBLP-Springer-Papers",
               root_path="../../Datasets/",output_path="./",include_reverse_edge=True,
               n_classes=1000,emb_size=128,label_mapping={},target_mapping={},modelID=''):
    def train(epoch):
        model.train()
        # tqdm.monitor_interval = 0
        pbar = tqdm(total=num_steps * batch_size)
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                        data.local_node_idx)
            out = out[data.train_mask]
            y = data.y[data.train_mask].squeeze()
            loss = F.nll_loss(out, y)
            # print("loss=",loss)
            loss.backward()
            optimizer.step()
            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            pbar.update(batch_size)

        # pbar.refresh()  # force print final state
        pbar.close()
        # pbar.reset()
        return total_loss / total_examples
    @torch.no_grad()
    def test():
        model.eval()
        out = model.inference(x_dict, edge_index_dict, key2int)
        out = out[key2int[subject_node]]

        y_pred = out.argmax(dim=-1, keepdim=True).cpu()
        y_true = data.y_dict[subject_node]

        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train'][subject_node]],
            'y_pred': y_pred[split_idx['train'][subject_node]],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid'][subject_node]],
            'y_pred': y_pred[split_idx['valid'][subject_node]],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test'][subject_node]],
            'y_pred': y_pred[split_idx['test'][subject_node]],
        })['acc']
        return train_acc, valid_acc, test_acc
    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    dataset_name = dataset_name
    to_remove_pedicates = []
    to_remove_subject_object =[]
    to_keep_edge_idx_map = []
    GNN_datasets=[dataset_name]
    root_path=root_path#"/shared_mnt/DBLP/KGTOSA_DBLP_Datasets/"
    include_reverse_edge=include_reverse_edge
    n_classes=n_classes
    output_path=output_path
    gsaint_Final_Test=0
    for GNN_dataset_name in GNN_datasets:  

        gsaint_start_t = datetime.datetime.now()
        ###################################Delete Folder if exist #############################
        dir_path=root_path+GNN_dataset_name
        try:
            shutil.rmtree(dir_path)
            print("Folder Deleted DEBUGGING")
        except OSError as e:
            print("Error Deleting : %s : %s" % (dir_path, e.strerror))
#         ####################
        print(f'searching for dataset at: {root_path+GNN_dataset_name}')
        dataset = PygNodePropPredDataset_hsh(name=GNN_dataset_name, root=root_path,numofClasses=str(n_classes))
        print("dataset_name=",dataset_name)
        dic_results = {}
        dic_results["GNN_Method"] = GNN_Methods.Graph_SAINT
        dic_results["to_keep_edge_idx_map"] = to_keep_edge_idx_map
        dic_results["dataset_name"] = dataset_name
        gnn_hyper_params_dict={"device":device,"num_layers":num_layers,"hidden_channels":hidden_channels,
            "dropout":dropout,"lr":lr,"epochs":epochs,"runs":runs,"batch_size":batch_size,
            "walk_length":walk_length,"num_steps":num_steps,"emb_size":emb_size}
        dic_results["gnn_hyper_params"] =gnn_hyper_params_dict
        print(getrusage(RUSAGE_SELF))
        start_t = datetime.datetime.now()
        data = dataset[0]
        #global subject_node
        subject_node=list(data.y_dict.keys())[0]
        if loadTrainedModel == 0:
            split_idx = dataset.get_idx_split()
        end_t = datetime.datetime.now()
        print("dataset init time=", end_t - start_t, " sec.")
        dic_results["dataset_load_time"] = (end_t - start_t).total_seconds()
        evaluator = Evaluator(name='ogbn-mag')
        logger = Logger(runs, gnn_hyper_params_dict)
        start_t = datetime.datetime.now()
        # We do not consider those attributes for now.
        data.node_year_dict = None
        data.edge_reltype_dict = None
        to_remove_rels = []
        for keys, (row, col) in data.edge_index_dict.items():
            if (keys[2] in to_remove_subject_object) or (keys[0] in to_remove_subject_object):
                to_remove_rels.append(keys)

        for keys, (row, col) in data.edge_index_dict.items():
            if (keys[1] in to_remove_pedicates):
                to_remove_rels.append(keys)
                to_remove_rels.append((keys[2], '_inv_' + keys[1], keys[0]))

        for elem in to_remove_rels:
            data.edge_index_dict.pop(elem, None)
            data.edge_reltype.pop(elem,None)

        for key in to_remove_subject_object:
            data.num_nodes_dict.pop(key, None)

        # dic_results["data_obj"] = data.to_dict()
        edge_index_dict = data.edge_index_dict
        ##############add inverse edges ###################
        if include_reverse_edge:
            key_lst = list(edge_index_dict.keys())
            for key in key_lst:
                r, c = edge_index_dict[(key[0], key[1], key[2])]
                edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

        # print("data after filter=",str(data))
        # print_memory_usage()
        out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
        edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
        # print_memory_usage()

        homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                         node_type=node_type, local_node_idx=local_node_idx,
                         num_nodes=node_type.size(0))

        homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
        homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]

        if loadTrainedModel == 0:
            homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
            homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True
            start_t = datetime.datetime.now()
            print("dataset.processed_dir",dataset.processed_dir)

            train_loader = GraphSAINTRandomWalkSampler(
            # train_loader = GraphSAINTTaskBaisedRandomWalkSampler(
            #train_loader=GraphSAINTTaskWeightedRandomWalkSampler(
                 homo_data,
                 batch_size=batch_size,
                 walk_length=walk_length,
                 # Subject_indices=local2global[subject_node],
                 # NodesWeightDic=NodesWeightDic,
                 num_steps=num_steps,
                 sample_coverage=0,
                 save_dir=dataset.processed_dir)
        end_t = datetime.datetime.now()
        # print("Sampling time=", end_t - start_t, " sec.")
        # dic_results[dataset_name]["GSaint_Sampling_time"] = (end_t - start_t).total_seconds()
        start_t = datetime.datetime.now()
        # Map informations to their canonical type.
        #######################intialize random features ###############################
        feat = torch.Tensor(data.num_nodes_dict[subject_node], emb_size)
        torch.nn.init.xavier_uniform_(feat)
        feat_dic = {subject_node: feat}
        ################################################################
        x_dict = {}
        for key, x in feat_dic.items():
            x_dict[key2int[key]] = x

        num_nodes_dict = {}
        for key, N in data.num_nodes_dict.items():
            num_nodes_dict[key2int[key]] = N

        end_t = datetime.datetime.now()
        print("model init time CPU=", end_t - start_t, " sec.")
        # dic_results["model init Time"] = (end_t - start_t).total_seconds()
        device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
        model = RGCN(emb_size, hidden_channels, dataset.num_classes, num_layers,
                     dropout, num_nodes_dict, list(x_dict.keys()),
                     len(edge_index_dict.keys())).to(device)

        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        print("x_dict=", x_dict.keys())
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        # model_name = dataset_name + "_DBLP_conf_GSAINT_QM.model"
        model_name = gen_model_name(dataset_name,dic_results["GNN_Method"])
        if loadTrainedModel == 1:
            with torch.no_grad():
                start_t = datetime.datetime.now()
                trained_model_path = Constants.KGNET_Config.trained_model_path + modelID
                # trained_model_path = r'/home/afandi/GitRepos/KGNET/Datasets/trained_models/mid-0000064.model'
                model_params_path = trained_model_path.replace('.model', '.param')

                with open(model_params_path, 'rb') as f:
                    dict_model_param = pickle.load(f)

                model = RGCN(dict_model_param['emb_size'],
                             dict_model_param['hidden_channels'],
                             dict_model_param['dataset.num_classes'],
                             dict_model_param['num_layers'],
                             dict_model_param['dropout'],
                             dict_model_param['num_nodes_dict'],
                             dict_model_param['list_x_dict_keys'],
                             dict_model_param['len_edge_index_dict_keys']
                             )
                label_mapping = dict_model_param['label_mapping']
                model.load_state_dict(torch.load(trained_model_path))
                print('Loaded Graph Saint Model!')
                model.eval()
                out = model.inference(x_dict, edge_index_dict, key2int)
                # out = model(x_dict, edge_index, edge_type, node_type,
                #             local_node_idx)
                out = out[key2int[subject_node]]
                out = out [:,:len(label_mapping)] #TODO
                y_pred = out.argmax(dim=-1, keepdim=True).cpu().flatten().tolist()
                end_t = datetime.datetime.now()
                print(dataset_name, "Infernce Time=", (end_t - start_t).total_seconds())
                print('predictions : ', y_pred)
                dict_pred = {}
                for i, pred in enumerate(y_pred):
                    dict_pred[target_mapping[i]] = label_mapping[pred]

                dic_results["InferenceTime"] = (end_t - start_t).total_seconds()

                # for pred in y_pred.flatten
                # dic_results['y_pred'] = pd.DataFrame({int(pred) : label_mapping[pred] for pred in y_pred.flatten().tolist()})
                # dic_results['y_pred'] = pd.DataFrame({'ent id':y_pred.flatten().tolist(),
                # 'ent name': [label_mapping[pred] for pred in y_pred.flatten().tolist()]})
                dic_results['y_pred'] = dict_pred
                print(dic_results['y_pred'])

            return dic_results
        else:
            print("start test")
            test()  # Test if inference on GPU succeeds.
            total_run_t = 0
            for run in range(runs):
                start_t = datetime.datetime.now()
                model.reset_parameters()
                for epoch in range(1, 1 + epochs):
                    loss = train(epoch)
                    ##############
                    if loss==-1:
                        return 0.001
                   ##############

                    torch.cuda.empty_cache()
                    result = test()
                    logger.add_result(run, result)
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}%, '
                          f'Test: {100 * test_acc:.2f}%')
                    # print("model # Total  parameters ", sum(p.numel() for p in model.parameters()))
                    # print("model # trainable paramters ", sum(p.numel() for p in model.parameters() if p.requires_grad))
                logger.print_statistics(run)
                end_t = datetime.datetime.now()
                total_run_t = total_run_t + (end_t - start_t).total_seconds()
                print("model run ", run, " train time CPU=", end_t - start_t, " sec.")
                print(getrusage(RUSAGE_SELF))
            total_run_t = (total_run_t + 0.00001) / runs
            gsaint_end_t = datetime.datetime.now()
            Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
            model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            dic_results['model_name'] = model_name
            dic_results["init_ru_maxrss"] = init_ru_maxrss
            dic_results["model_ru_maxrss"] = model_loaded_ru_maxrss
            dic_results["model_trained_ru_maxrss"] = model_trained_ru_maxrss
            dic_results["Highest_Train_Acc"] = Highest_Train.item()
            dic_results["Highest_Valid_Acc"] = Highest_Valid.item()
            dic_results["Final_Train_Acc"] = Final_Train.item()
            gsaint_Final_Test= Final_Test.item()
            dic_results["Final_Test_Acc"] = Final_Test.item()
            dic_results["Train_Runs_Count"] = runs
            dic_results["Train_Time"] = total_run_t
            dic_results["Total_Time"] = (gsaint_end_t - gsaint_start_t).total_seconds()
            dic_results["Model_Parameters_Count"]= sum(p.numel() for p in model.parameters())
            dic_results["Model_Trainable_Paramters_Count"]=sum(p.numel() for p in model.parameters() if p.requires_grad)

            ############### Model Hyper Parameters ###############
            dict_model_param = {}
            dict_model_param['emb_size'] = emb_size
            dict_model_param['hidden_channels'] = hidden_channels  # dataset.num_classes
            dict_model_param['dataset.num_classes'] = dataset.num_classes
            dict_model_param['num_layers'] = num_layers
            dict_model_param['dropout'] = dropout
            dict_model_param['num_nodes_dict'] = num_nodes_dict
            dict_model_param['list_x_dict_keys'] = list(x_dict.keys())
            dict_model_param['len_edge_index_dict_keys'] = len(edge_index_dict.keys())
            if len(label_mapping) > 0:
                dict_model_param['label_mapping'] = label_mapping
            logs_path = os.path.join(output_path,'logs')
            model_path = os.path.join(output_path,'trained_models')
            create_dir([logs_path,model_path]) # args: list of paths
            # pd.DataFrame(dic_results).transpose().to_json(os.path.join(logs_path,model_name+'.json') )
            with open(os.path.join(logs_path, model_name +'_log.metadata'), "w") as outfile:
                json.dump(dic_results, outfile)
            torch.save(model.state_dict(), os.path.join(model_path , model_name)+".model")
            dic_results["data_obj"] = data.to_dict()
            with open(os.path.join(model_path, model_name) + ".param", 'wb') as f:
                pickle.dump(dict_model_param, f)

        del train_loader
    return dic_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--loadTrainedModel', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="inference")
    parser.add_argument('--root_path', type=str, default="../../Datasets/")
    parser.add_argument('--output_path', type=str, default="./")
    parser.add_argument('--include_reverse_edge', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=440)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--modelID', type=str, default='mid-0000092.model')

    args = parser.parse_args()
    print(args)
    print(graphSaint(args.device,args.num_layers,args.hidden_channels,args.dropout,args.lr,args.epochs,args.runs,args.batch_size,args.walk_length,args.num_steps,args.loadTrainedModel,args.dataset_name,args.root_path,args.output_path,args.include_reverse_edge,args.n_classes,args.emb_size,modelID=args.modelID))






