import argparse
import copy
# !conda install pyg -c pygmport argparse# !conda install pyg -c pyg
import datetime
import shutil
import psutil
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Parameter, ModuleDict, ModuleList, Linear, ParameterDict
import tracemalloc
## conda install pyg -c pyg
from torch_sparse import SparseTensor
# pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# pip install torch-geometric
import sys
import traceback
import sys
from resource import *
from logger import Logger
import logging
import faulthandler
faulthandler.enable()
import traceback
# sys.path.insert(0, '/shared_mnt/github_repos/ogb/')
# sys.path.insert(0, '/ibex/scratch/omarro/KGNET/KGTOSA_NC/ogb/')
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# sys.path.insert(0, '/shared_mnt/github_repos/ogb/')
from evaluate import Evaluator
from dataset_pyg_hsh import PygNodePropPredDataset_hsh
subject_node='Paper'
print(subject_node)

def print_memory_usage():
    # print("max_mem_GB=",psutil.Process().memory_info().rss / (1024 * 1024*1024))
    # print("get_process_memory=",getrusage(RUSAGE_SELF).ru_maxrss/(1024*1024))
    print('virtual memory GB:', psutil.virtual_memory().active / (1024.0 ** 3), " percent",
          psutil.virtual_memory().percent)


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
            # print("key=", key)
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
        x_dict = copy.copy(x_dict)  ## copy x_dict features
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


def main():
    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    print(getrusage(RUSAGE_SELF))
    parser = argparse.ArgumentParser(description='OGBN-MAG (Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)  # 5
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--loadTrainedModel', type=int, default=0)
    parser.add_argument('--trainFM', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dic_results = {}
    rgcn_start_t = datetime.datetime.now()
    start_t = datetime.datetime.now()
    MAG_datasets=[
                  # "ogbn-mag-ReleventSubGraph_Venue",
                  # "ogbn-mag-ReleventSubGraph_Discipline", 
                  # "KGTOSA_MAG_Paper_Discipline_StarQuery",
                  # "KGTOSA_MAG_Paper_Venue_StarQuery",
                # "KGTOSA_MAG_Paper_Venue_BStarQuery",
                # "KGTOSA_MAG_Paper_Discipline_BStarQuery",
                # "KGTOSA_MAG_Paper_Venue_2HOPS_Literals",
                # "OGBN-MAG_PV_FM",
                # "OGBN-MAG_PV_StarQuery",
                # "KGTOSA_MAG_Paper_Discipline_2HOPS_WO_Literals"
                # "OGBN-MAG_StarQuery_PD",
                # "OGBN-MAG_FM_PD",
                "OGBN-MAG_PV_BStarQuery",
                "OGBN-MAG_PV_PathQuery",
                "OGBN-MAG_PV_BPathQuery",
    ]       
    for dataset_name in MAG_datasets:    
        # dataset = PygNodePropPredDataset_hsh(name=dataset_name,root='/shared_mnt/KGTOSA_MAG/',numofClasses='350')
        dataset = PygNodePropPredDataset_hsh(name=dataset_name,root='/shared_mnt/MAG_subsets/',numofClasses='350')       
        print("dataset_name=", dataset_name)
        dic_results[dataset_name] = {}
        dic_results[dataset_name]["usecase"] = dataset_name
        dic_results[dataset_name]["GNN_Model"] = "RGCN"
        dic_results[dataset_name]["rgcn_hyper_params"] = str(args)

        split_idx = dataset.get_idx_split()
        data = dataset[0]
        end_t = datetime.datetime.now()
        print("data load time=", end_t - start_t, " sec.")
        dic_results[dataset_name]["rgcn_data_init_time"] = (end_t - start_t).total_seconds()
        # We do not consider those attributes for now.
        data.node_year_dict = None        
        print(list(data.y_dict.keys()))
        global subject_node
        subject_node=list(data.y_dict.keys())[0]
            
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
        dic_results[dataset_name]["data"] = str(data)
        edge_types = list(data.adj_t_dict.keys())
        start_t = datetime.datetime.now()
        # x_types = list(data.x_dict.keys())
        x_types = subject_node
        ############init papers with random embeddings #######################
        # len(data.x_dict[subject_node][0])
        feat = torch.Tensor(data.num_nodes_dict[subject_node], 128)
        torch.nn.init.xavier_uniform_(feat)
        feat_dic = {subject_node: feat}
        # print("feat_dic=",feat_dic)
        #####################################
        # data.x_dict[subject_node].size(-1)

        # data.num_nodes_dict.pop('type', None)
        print("dataset.num_classes=", dataset.num_classes)
        model = RGCN(feat.size(-1), args.hidden_channels,
                     dataset.num_classes, args.num_layers, args.dropout,
                     data.num_nodes_dict, x_types, edge_types)
        train_idx = split_idx['train'][subject_node].to(device)

        evaluator = Evaluator(name='ogbn-mag')
        ####################
        logger = Logger(args.runs, args)
        ####################        
        end_t = datetime.datetime.now()
        dic_results[dataset_name]["model init Time"] = (end_t - start_t).total_seconds()
        start_t = datetime.datetime.now()
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        data = data.to(device)
        model = model.to(device)

        print("model init time CPU=", end_t - start_t, " sec.")
        total_run_t = 0
        for run in range(args.runs):
            start_t = datetime.datetime.now()
            model.reset_parameters()
            print_memory_usage()

            for epoch in range(1, 1 + args.epochs):
                # print_memory_usage()
                # print("data.y_dict[subject_node]=",max(data.y_dict[subject_node]),min(data.y_dict[subject_node]),len(data.y_dict[subject_node]))
                loss = train(model, feat_dic, data.adj_t_dict,
                             data.y_dict[subject_node], train_idx, optimizer)
                # print_memory_usage()
                result = test(model, feat_dic, data.adj_t_dict,
                              data.y_dict[subject_node], split_idx, evaluator)
                # print_memory_usage()
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
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
        total_run_t = (total_run_t + 0.00001) / args.runs
        rgcn_end_t = datetime.datetime.now()
        Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
        model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        dic_results[dataset_name]["init_ru_maxrss"] = init_ru_maxrss
        dic_results[dataset_name]["model_ru_maxrss"] = model_loaded_ru_maxrss
        dic_results[dataset_name]["model_trained_ru_maxrss"] = model_trained_ru_maxrss
        dic_results[dataset_name]["Highest_Train"] = Highest_Train.item()
        dic_results[dataset_name]["Highest_Valid"] = Highest_Valid.item()
        dic_results[dataset_name]["Final_Train"] = Final_Train.item()
        dic_results[dataset_name]["Final_Test"] = Final_Test.item()
        dic_results[dataset_name]["runs_count"] = args.runs
        dic_results[dataset_name]["avg_train_time"] = total_run_t
        dic_results[dataset_name]["rgcn_total_time"] = (rgcn_end_t - rgcn_start_t).total_seconds()
        pd.DataFrame(dic_results).transpose().to_csv(
            "/shared_mnt/KGTOSA_MAG/OGBN_"+dataset_name+"_RGCN_times" + ".csv",
            index=False)
        # shutil.rmtree("/shared_mnt/DBLP_GNN_Usecases/OGBN_DBLP_QM_ZIP/" + dataset_name)
        # torch.save(model.state_dict(),
                   # "/shared_mnt/DBLP_GNN_Usecases/OGBN_DBLP_QM_ZIP/" + dataset_name + "_QM_conf_RGCN.model")

if __name__ == "__main__":
    tracemalloc.start()
    print("main_starts")
    main()
