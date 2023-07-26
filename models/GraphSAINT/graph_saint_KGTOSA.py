from copy import copy
import argparse
import shutil
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
from KGTOSA_Samplers import GraphSAINTTaskBaisedRandomWalkSampler,GraphSAINTTaskWeightedRandomWalkSampler
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
#sys.path.insert(0, '/shared_mnt/github_repos/ogb/')
from ogb.nodeproppred import PygNodePropPredDataset
from evaluater import Evaluator
from custome_pyg_dataset import PygNodePropPredDataset_hsh
from resource import *
from logger import Logger
import faulthandler
faulthandler.enable()
import pickle
subject_node='Paper'
def print_memory_usage():
    # print("max_mem_GB=",psutil.Process().memory_info().rss / (1024 * 1024*1024))
    # print("get_process_memory=",getrusage(RUSAGE_SELF).ru_maxrss/(1024*1024))
    print('used virtual memory GB:', psutil.virtual_memory().used / (1024.0 ** 3), " percent",
          psutil.virtual_memory().percent)


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


class RGCN(torch.nn.Module):
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
dic_results = {}
def graphSaint():
    def train(epoch):
        model.train()
        # tqdm.monitor_interval = 0
        pbar = tqdm(total=args.num_steps * args.batch_size)
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
            pbar.update(args.batch_size)

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


    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=30)
    # parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--loadTrainedModel', type=int, default=0)
    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    args = parser.parse_args()
    to_remove_pedicates = []
    to_remove_subject_object =[]
    to_keep_edge_idx_map = []
    GA_Index = 0
    MAG_datasets=["DBLP_Paper_Venue_FM_Literals2Nodes_SY1900_EY2021_50Class"]
    root_path="/shared_mnt/DBLP/KGTOSA_DBLP_Datasets/"
    include_reverse_edge=True
    n_classes=50
    print(args)
    gsaint_Final_Test=0
    for GA_dataset_name in MAG_datasets:  
        try:
            gsaint_start_t = datetime.datetime.now()
            ###################################Delete Folder if exist #############################
            dir_path=root_path+GA_dataset_name
            try:
                shutil.rmtree(dir_path)
                print("Folder Deleted")
            except OSError as e:
                print("Error Deleting : %s : %s" % (dir_path, e.strerror))
    #         ####################
            dataset = PygNodePropPredDataset_hsh(name=GA_dataset_name, root=root_path,numofClasses=str(n_classes))
            dataset_name=GA_dataset_name+"_GA_"+str(GA_Index)
            print("dataset_name=",dataset_name)
            dic_results[dataset_name] = {}
            dic_results[dataset_name]["GNN_Model"] = "GSaint"
            dic_results[dataset_name]["GA_Index"] = GA_Index
            dic_results[dataset_name]["to_keep_edge_idx_map"] = to_keep_edge_idx_map        
            dic_results[dataset_name]["usecase"] = dataset_name
            dic_results[dataset_name]["gnn_hyper_params"] = str(args)

            print(getrusage(RUSAGE_SELF))
            start_t = datetime.datetime.now()
            data = dataset[0]
            global subject_node
            subject_node=list(data.y_dict.keys())[0]
            split_idx = dataset.get_idx_split()
            end_t = datetime.datetime.now()
            print("dataset init time=", end_t - start_t, " sec.")
            dic_results[dataset_name]["GSaint_data_init_time"] = (end_t - start_t).total_seconds()
            evaluator = Evaluator(name='ogbn-mag')
            logger = Logger(args.runs, args)

            start_t = datetime.datetime.now()
            # We do not consider those attributes for now.
            data.node_year_dict = None
            data.edge_reltype_dict = None

            # remove_subject_object = [
            #     # 'doi',
            #     # 'sameAs',
            #     # 'rdf',
            #     # 'net',
            #     # 'differentFrom',
            #     # 'otherHomepage',
            #     # 'primaryElectronicEdition',
            #     # 'otherElectronicEdition',
            #     # 'archivedElectronicEdition',
            #     # 'entity',
            #     # 'db'
            # ]
            # remove_pedicates = [
            #     # 'schema#awardWebpage',
            #     # 'schema#webpage',
            #     # 'schema#archivedWebpage',
            #     # 'schema#primaryHomepage',
            #     # 'schema#wikipedia',
            #     # 'schema#orcid',
            #     # 'schema#publishedAsPartOf'
            # ]
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


            dic_results[dataset_name]["data"] = str(data)
            ##############add inverse edges ###################
            if include_reverse_edge:
                edge_index_dict = data.edge_index_dict
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

            homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
            homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True
            # print(homo_data)
            start_t = datetime.datetime.now()
            print("dataset.processed_dir",dataset.processed_dir)
            # weights_dic={'rec':0.5,'pid':0.3,'Object_schema#publishedInJournal':0.1,'Object_schema#numberOfCreators':0.1} # non target based
            # weights_dic={'rec':0.5,'net':0.087,'Object_schema#numberOfCreators':0.086,'db':0.046,'Object_schema#yearOfEvent':0.046,'Object_schema#title':0.04} # target based
            # print("weights_dic=",weights_dic)
            # weights_dic={'CreativeWork': 0.3, 'datePublished': 0.1, 'actor': 0.1,
            #  'countryOfOrigin': 0.1, 'producer': 0.1, 'duration': 0.1, 'contentLocation': 0.1,
            #  'inLanguage': 0.1}
            # NodesWeightDic={}
            # for key in weights_dic.keys():
            #     NodesWeightDic[(min(local2global[key]).item(),max(local2global[key]).item())]=weights_dic[key]
            train_loader = GraphSAINTRandomWalkSampler(
            # train_loader = GraphSAINTTaskBaisedRandomWalkSampler(
            #train_loader=GraphSAINTTaskWeightedRandomWalkSampler(
                 homo_data,
                 batch_size=args.batch_size,
                 walk_length=args.num_layers,
                 # Subject_indices=local2global[subject_node],
                 # NodesWeightDic=NodesWeightDic,
                 num_steps=args.num_steps,
                 sample_coverage=0,
                 save_dir=dataset.processed_dir)
            end_t = datetime.datetime.now()
            print("Sampling time=", end_t - start_t, " sec.")
            dic_results[dataset_name]["GSaint_Sampling_time"] = (end_t - start_t).total_seconds()
            start_t = datetime.datetime.now()
            # Map informations to their canonical type.
            #######################intialize random features ###############################
            emb_size=128
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
            dic_results[dataset_name]["model init Time"] = (end_t - start_t).total_seconds()
            device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
            model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
                         args.dropout, num_nodes_dict, list(x_dict.keys()),
                         len(edge_index_dict.keys())).to(device)

            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            print("x_dict=", x_dict.keys())
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            output_path = "/shared_mnt/KGTOSA_MAG/"
            model_name = dataset_name + "_DBLP_conf_GSAINT_QM.model"
            if args.loadTrainedModel == 1:
                with torch.no_grad():
                    start_t = datetime.datetime.now()
                    model.load_state_dict(torch.load(output_path +model_name))
                    model.eval()
                    out = model.inference(x_dict, edge_index_dict, key2int)
                    out = out[key2int[subject_node]]
                    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
                    end_t = datetime.datetime.now()
                    print(dataset_name, "Infernce Time=", (end_t - start_t).total_seconds())
                    dic_results[dataset_name]["InfernceTime="] = (end_t - start_t).total_seconds()
            else:
                print("start test")
                test()  # Test if inference on GPU succeeds.
                total_run_t = 0
                for run in range(args.runs):
                    start_t = datetime.datetime.now()
                    model.reset_parameters()
                    for epoch in range(1, 1 + args.epochs):
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
                        print("model # Total  parameters ", sum(p.numel() for p in model.parameters()))
                        print("model # trainable paramters ", sum(p.numel() for p in model.parameters() if p.requires_grad))
                    logger.print_statistics(run)
                    end_t = datetime.datetime.now()
                    total_run_t = total_run_t + (end_t - start_t).total_seconds()
                    print("model run ", run, " train time CPU=", end_t - start_t, " sec.")
                    print(getrusage(RUSAGE_SELF))
                total_run_t = (total_run_t + 0.00001) / args.runs
                gsaint_end_t = datetime.datetime.now()
                Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
                model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
                dic_results[dataset_name]["init_ru_maxrss"] = init_ru_maxrss
                dic_results[dataset_name]["model_ru_maxrss"] = model_loaded_ru_maxrss
                dic_results[dataset_name]["model_trained_ru_maxrss"] = model_trained_ru_maxrss
                dic_results[dataset_name]["Highest_Train"] = Highest_Train.item()
                dic_results[dataset_name]["Highest_Valid"] = Highest_Valid.item()
                dic_results[dataset_name]["Final_Train"] = Final_Train.item()
                gsaint_Final_Test= Final_Test.item()
                dic_results[dataset_name]["Final_Test"] = Final_Test.item()
                dic_results[dataset_name]["runs_count"] = args.runs
                dic_results[dataset_name]["avg_train_time"] = total_run_t
                dic_results[dataset_name]["rgcn_total_time"] = (gsaint_end_t - gsaint_start_t).total_seconds()
                dic_results[dataset_name]["model_parameters_count"]= sum(p.numel() for p in model.parameters())
                dic_results[dataset_name]["model_trainable_paramters_count"]=sum(p.numel() for p in model.parameters() if p.requires_grad)

                pd.DataFrame(dic_results).transpose().to_csv(output_path+"GSAINT_" + GA_dataset_name + "_Times.csv", index=False)
                # shutil.rmtree("/shared_mnt/DBLP/" + dataset_name)
                torch.save(model.state_dict(), output_path +model_name )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("dataset_name Exception")
        del train_loader
    return gsaint_Final_Test

if __name__ == '__main__':
    print(graphSaint())






