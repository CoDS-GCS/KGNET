import sys
import os
GMLaaS_models_path = sys.path[0].split("KGNET")[0] + "/KGNET/GMLaaS/models"
sys.path.insert(0, GMLaaS_models_path)
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from copy import copy
import json
import argparse
# import copy
import datetime
import psutil
import shutil
from Constants import *

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter, ModuleDict
import tracemalloc
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
from resource import *
from logger import Logger
import logging
import faulthandler
faulthandler.enable()

from evaluater import Evaluator
from custome_pyg_dataset import PygNodePropPredDataset_hsh
from resource import *
from logger import Logger
import faulthandler
faulthandler.enable()
from model import Model
# subject_node='Paper'
# print(subject_node)

def print_memory_usage():
    # print("max_mem_GB=",psutil.Process().memory_info().rss / (1024 * 1024*1024))
    # print("get_process_memory=",getrusage(RUSAGE_SELF).ru_maxrss/(1024*1024))
    print('virtual memory GB:', psutil.virtual_memory().active / (1024.0 ** 3), " percent",
          psutil.virtual_memory().percent)

def gen_model_name(dataset_name,GNN_Method):
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # return dataset_name+'_'+model_name+'_'+timestamp
    return dataset_name

def create_dir(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.mkdir(path)

class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(RGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # print('in_channels=', in_channels, 'out_channels =', out_channels)

        # `ModuleDict` does not allow tuples :( ## create linear layer for each predicate type i.e writes, affaliated with
        # print_memory_usage()
        # print("alloc Linear rel_lins")
        edge_types_dic = {}
        for key in edge_types:
            edge_types_dic[f'{key[0].replace(".","_")}_{key[1].replace(".","_")}_{key[2].replace(".","_")}'] = Linear(in_channels, out_channels, bias=False)
        self.rel_lins = ModuleDict(edge_types_dic)
        # print_memory_usage()
        # print("alloc Linear root_lins")
        self.root_lins = ModuleDict({  ## create linear layer for each node type (distinct veriex i.e author,paper,...)
            key.replace(".","_"): Linear(in_channels, out_channels, bias=True)
            for key in node_types
        })
        # print_memory_usage()
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):  ## aggregate updates
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)

        for key, adj_t in adj_t_dict.items():
            key_str = f'{key[0]}_{key[1]}_{key[2]}'
            x = x_dict[key[0]]
            out = self.rel_lins[key_str](adj_t.matmul(x, reduce='mean'))
            out_dict[key[2]].add_(out)

        return out_dict


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, edge_types):
        super(RGCN, self).__init__()

        node_types = list(num_nodes_dict.keys())
        set_diff = set(node_types).difference(set([x_types]))
        param_dic = {}
        # print('alloc paramters')
        for key in set_diff:
            if key not in ['type']:
                # print(key, " is tensor of", num_nodes_dict[key], " x ", in_channels)
                # print("alloc RGCN Parameter for key "+key)
                param_dic[key.replace(".","_")] = Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
                # print(key)
                # print_memory_usage()
        self.x_dict = None
        # self.embs = ParameterDict({ ## set node embedding features for all types except paper
        #     key: Parameter(torch.Tensor(num_nodes_dict[key], in_channels)) ## vertixcount*embedding size
        #     for key in set(node_types).difference(set([x_types]))
        # })
        # print("param_dic=",param_dic)
        self.embs = ParameterDict(param_dic)
        self.convs = ModuleList()
        # print('alloc conv layers')
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, node_types, edge_types))  ## Start layer
        for _ in range(num_layers - 2):  ## hidden Layers
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, node_types, edge_types))
        self.convs.append(RGCNConv(hidden_channels, out_channels, node_types, edge_types))  ## output layer
        # print_memory_usage()
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embs.values():
            torch.nn.init.xavier_uniform_(emb)  ## intialize embeddinga with Xavier uniform dist
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        x_dict = copy(x_dict)  ## copy x_dict features
        for key, emb in self.embs.items():
            x_dict[key] = emb

        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, adj_t_dict)  ## update features from by convolution layer forward (mean)
            for key, x in x_dict.items():
                x_dict[key] = F.relu(x)  ## relu
                x_dict[key] = F.dropout(x, p=self.dropout, training=self.training)  ## dropout some updated features
        return self.convs[-1](x_dict, adj_t_dict)


def train(model, x_dict, adj_t_dict, y_true, train_idx, optimizer):
    # print_memory_usage()
    model.train()
    # print_memory_usage()
    optimizer.zero_grad()
    out = model(x_dict, adj_t_dict)[subject_node].log_softmax(dim=-1)
    # print("out=",out)
    # print(out.shape)
    # print(y_true.shape)
    loss = F.nll_loss(out[train_idx], y_true[train_idx].squeeze())
    
            
    # print("max(train_idx)=",max(train_idx))
    # print("loss=",loss)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x_dict, adj_t_dict, y_true, split_idx, evaluator):
    model.eval()

    out = model(x_dict, adj_t_dict)[subject_node]
    y_pred = out.argmax(dim=-1, keepdim=True)

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
def rgcn(device=0,num_layers=2,hidden_channels=64,dropout=0.5,lr=0.005,epochs=2,runs=1,batch_size=2000,walk_length=2,num_steps=10,loadTrainedModel=0,dataset_name="DBLP-Springer-Papers",root_path="../../Datasets/",output_path="./",include_reverse_edge=True,n_classes=1000,emb_size=128):

    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss

    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
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
    # device = torch.device(device)
    dic_results = {}
    rgcn_start_t = datetime.datetime.now()
    start_t = datetime.datetime.now()
    for dataset_name in GNN_datasets:    
        # dataset = PygNodePropPredDataset_hsh(name=dataset_name,root='/shared_mnt/KGTOSA_MAG/',numofClasses='350')
        dataset = PygNodePropPredDataset_hsh(name=dataset_name,root=root_path,numofClasses=str(n_classes))       
        print("dataset_name=", dataset_name)
        dic_results = {}
        dic_results["dataset_name"] = dataset_name
        dic_results["GNN_Method"] = GNN_Methods.RGCN
        gnn_hyper_params_dict={"device":device,"num_layers":num_layers,"hidden_channels":hidden_channels,
                "dropout":dropout,"lr":lr,"epochs":epochs,"runs":runs,"batch_size":batch_size,
                "walk_length":walk_length,"num_steps":num_steps,"emb_size":emb_size}
        dic_results["gnn_hyper_params"] = gnn_hyper_params_dict
        data = dataset[0]

        print(getrusage(RUSAGE_SELF))
        start_t = datetime.datetime.now()
        global subject_node
        subject_node=list(data.y_dict.keys())[0]

        split_idx = dataset.get_idx_split()
        end_t = datetime.datetime.now()
        print("data load time=", end_t - start_t, " sec.")
        dic_results["rgcn_data_init_time"] = (end_t - start_t).total_seconds()
        # We do not consider those attributes for now.
        data.node_year_dict = None        
        print(list(data.y_dict.keys()))
        # global subject_node
            
        data.edge_reltype_dict = None
        # print("data.edge_index_dict=", data.edge_index_dict)
        # 'authoredBy',
        remove_subject_object = []
        remove_pedicates = []
        to_remove_rels = []
        for keys, (row, col) in data.edge_index_dict.items():
            if (keys[2] in remove_subject_object) or (keys[0] in remove_subject_object):
                to_remove_rels.append(keys)

        for keys, (row, col) in data.edge_index_dict.items():
            if (keys[1] in remove_pedicates):
                to_remove_rels.append(keys)
                to_remove_rels.append((keys[2], '_inv_'+keys[1], keys[0]))

        for elem in to_remove_rels:
            data.edge_index_dict.pop(elem, None)

        for key in remove_subject_object:
            data.num_nodes_dict.pop(key, None)


        # Convert to new transposed `SparseTensor` format and add reverse edges.
        data.adj_t_dict = {}
        total_size = 0
        for keys, (row, col) in data.edge_index_dict.items():
            # print("keys=", keys)
            sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
            # print("sizes=", sizes)
            total_size += (data.num_nodes_dict[keys[0]] * data.num_nodes_dict[keys[2]])
            # print("total_size=",total_size/(1024*1024*1024))
            # print_memory_usage()
            adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
            # total_size+=adj.size(0)
            # adj = SparseTensor(row=row, col=col)[:sizes[0], :sizes[1]] # TEST
            # if keys[0] != keys[2]:  ## subject and object are diffrent
            data.adj_t_dict[(keys[0].replace(".","_"), keys[1].replace(".","_"), keys[2].replace(".","_"))] = adj.t()
            data.adj_t_dict[(keys[2].replace(".","_"), '_inv_'+keys[1].replace(".","_"), keys[0].replace(".","_"))] = adj
            # else:
                # data.adj_t_dict[keys] = adj.to_symmetric()
        data.edge_index_dict = None

        print(data)
        edge_types = list(data.adj_t_dict.keys())
        start_t = datetime.datetime.now()
        # x_types = list(data.x_dict.keys())
        x_types = subject_node
        ############init papers with random embeddings #######################
        # len(data.x_dict[subject_node][0])
        feat = torch.Tensor(data.num_nodes_dict[subject_node], emb_size)
        torch.nn.init.xavier_uniform_(feat)
        feat_dic = {subject_node: feat}
        # print("feat_dic=",feat_dic)
        #####################################
        # data.x_dict[subject_node].size(-1)

        # data.num_nodes_dict.pop('type', None)
        print("dataset.num_classes=", dataset.num_classes)
        model = RGCN(feat.size(-1), hidden_channels,
                     dataset.num_classes, num_layers, dropout,
                     data.num_nodes_dict, x_types, edge_types)
        train_idx = split_idx['train'][subject_node].to(device)
        print(model)

        evaluator = Evaluator(name='ogbn-mag')
        ####################
        logger = Logger(runs, gnn_hyper_params_dict)
        ####################        
        end_t = datetime.datetime.now()
        dic_results["model init Time"] = (end_t - start_t).total_seconds()
        start_t = datetime.datetime.now()
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        model_name = gen_model_name(dataset_name,dic_results["GNN_Method"])
        data = data.to(device)
        model = model.to(device)

        print("model init time CPU=", end_t - start_t, " sec.")
        total_run_t = 0

        if args.loadTrainedModel:
            model.load_state_dict(torch.load(os.path.join(KGNET_Config.trained_model_path,args.model_id)))
            list_infTime = []
            for i in range(5):
                time_infStart = datetime.datetime.now()
                result = test(model, feat_dic, data.adj_t_dict,
                              data.y_dict[subject_node], split_idx, evaluator)
                time_infEnd = (datetime.datetime.now() - time_infStart).total_seconds()
                print(result,'\n', f'Time taken = {time_infEnd}')
                list_infTime.append(time_infEnd)
            print(f'\naverage inf time {sum(list_infTime)/len(list_infTime)} seconds')
            return

        for run in range(runs):
            start_t = datetime.datetime.now()
            model.reset_parameters()
            print_memory_usage()

            for epoch in range(1, 1 + epochs):
                # print_memory_usage()
                # print("data.y_dict[subject_node]=",max(data.y_dict[subject_node]),min(data.y_dict[subject_node]),len(data.y_dict[subject_node]))
                loss = train(model, feat_dic, data.adj_t_dict,
                             data.y_dict[subject_node], train_idx, optimizer)
                # print_memory_usage()
                result = test(model, feat_dic, data.adj_t_dict,
                              data.y_dict[subject_node], split_idx, evaluator)
                # print_memory_usage()
                logger.add_result(run, result)

                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')
            end_t = datetime.datetime.now()
            logger.print_statistics(run)
            total_run_t = total_run_t + (end_t - start_t).total_seconds()
            print("model run ", run, " train time CPU=", end_t - start_t, " sec.")
            print(getrusage(RUSAGE_SELF))
        logger.print_statistics()
        total_run_t = (total_run_t + 0.00001) / runs
        rgcn_end_t = datetime.datetime.now()
        Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
        model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        dic_results["Inference_Time"] = 0
        dic_results["model_name"] = model_name
        dic_results["init_ru_maxrss"] = init_ru_maxrss
        dic_results["model_ru_maxrss"] = model_loaded_ru_maxrss
        dic_results["model_trained_ru_maxrss"] = model_trained_ru_maxrss
        dic_results["Highest_Train_Acc"] = Highest_Train.item()
        dic_results["Highest_Valid_Acc"] = Highest_Valid.item()
        dic_results["Final_Train_Acc"] = Final_Train.item()
        dic_results["Final_Test_Acc"] = Final_Test.item()
        dic_results["Train_Runs_Count"] = runs
        dic_results["Train_Time"] = total_run_t
        dic_results["Total_Time"] = (rgcn_end_t - rgcn_start_t).total_seconds()
        dic_results["Model_Parameters_Count"]= sum(p.numel() for p in model.parameters())
        dic_results["Model_Trainable_Paramters_Count"]=sum(p.numel() for p in model.parameters() if p.requires_grad)

        logs_path = os.path.join(output_path,'logs')
        model_path = os.path.join(output_path,'trained_models')        
        create_dir([logs_path,model_path]) 
        with open(os.path.join(logs_path, model_name +'_log.metadata'), "w") as outfile:
            json.dump(dic_results, outfile)
        torch.save(model.state_dict(), os.path.join(model_path , model_name)+".model")
        dic_results["data_obj"] = data.to_dict()
    return dic_results

# def main():

#     parser = argparse.ArgumentParser(description='OGBN-MAG (Full-Batch)')
#     parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--log_steps', type=int, default=1)
#     parser.add_argument('--num_layers', type=int, default=2)  # 5
#     parser.add_argument('--hidden_channels', type=int, default=64)
#     parser.add_argument('--dropout', type=float, default=0.5)
#     parser.add_argument('--lr', type=float, default=0.01)
#     # parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--runs', type=int, default=3)
#     parser.add_argument('--loadTrainedModel', type=int, default=0)
#     parser.add_argument('--trainFM', type=int, default=1)
#     parser.add_argument('--batch_size', type=int, default=2000)
#     parser.add_argument('--walk_length', type=int, default=2)
#     parser.add_argument('--num_steps', type=int, default=10)
#     parser.add_argument('--loadTrainedModel', type=int, default=0)
#     parser.add_argument('--dataset_name', type=str, default="DBLP-Springer-Papers")
#     parser.add_argument('--root_path', type=str, default="../../Datasets/")
#     parser.add_argument('--output_path', type=str, default="./")
#     parser.add_argument('--include_reverse_edge', type=bool, default=True)
#     parser.add_argument('--n_classes', type=int, default=440)
#     parser.add_argument('--emb_size', type=int, default=128)
#     args = parser.parse_args()
#     print(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OGBN-MAG (Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)  # 5
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--loadTrainedModel', type=int, default=0)
    parser.add_argument('--model_id', type=str, default='ogbn_mag.model')
    parser.add_argument('--trainFM', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default="YAGO310_Person-Aff_50_FG") # ogbn_mag
    parser.add_argument('--root_path', type=str, default=KGNET_Config.datasets_output_path)
    parser.add_argument('--output_path', type=str, default="./")
    parser.add_argument('--include_reverse_edge', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=50) # 349-MAG
    parser.add_argument('--emb_size', type=int, default=128)
    args = parser.parse_args()
    print(args)
    print(rgcn(args.device,args.num_layers,args.hidden_channels,args.dropout,args.lr,args.epochs,args.runs,args.batch_size,args.walk_length,args.num_steps,args.loadTrainedModel,args.dataset_name,args.root_path,args.output_path,args.include_reverse_edge,args.n_classes,args.emb_size))
