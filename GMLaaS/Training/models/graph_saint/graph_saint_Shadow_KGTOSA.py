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
from torch_geometric.loader import  GraphSAINTRandomWalkSampler , ShaDowKHopSampler
#GraphSAINTTaskBaisedRandomWalkSample
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

sys.path.insert(0, '/shared_mnt/github_repos/ogb/')
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator, PygNodePropPredDataset_hsh
from resource import *
from logger import Logger
import faulthandler

faulthandler.enable()
import pickle

torch.multiprocessing.set_sharing_strategy('file_system')
subject_node = 'Paper'
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
        # pbar = tqdm(total=args.num_steps * args.batch_size)
        print("len(train_loader)",len(train_loader))
        pbar = tqdm(total=len(train_loader))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for data in train_loader:
            # print("type(data)=",type(data))
            # print("homo_data.y=",homo_data.y)
            data = data.to(device)
            optimizer.zero_grad()
            out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                        data.local_node_idx)
            # out = out[data.train_mask]
            out = torch.index_select(out, 0, data.root_n_id)
            ###################### for graph saint ######################
            # y = data.y[data.train_mask].squeeze()
            #####################for Shadow GNN ##################
            # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/shadow.py
            # train_mask_true_idx=torch.index_select(data.local_node_idx, 0, (data.train_mask == True).nonzero(as_tuple=True)[0] )
            # y=torch.index_select(homo_data.y[local2global[subject_node]], 0, train_mask_true_idx).squeeze()
            y = data.y.squeeze(1)
            # print(data.y)
            ######################################################
            # print("data.train_mask=", data.train_mask)
            # print(out.shape)
            # print(y.shape)
            # print(len(data.train_mask))
            loss = F.nll_loss(out, y)
            # print("loss=",loss)
            loss.backward()
            optimizer.step()
            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            # pbar.update(args.batch_size)
            pbar.update(1)

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
    # parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--loadTrainedModel', type=int, default=0)
    parser.add_argument('--graphsaint_dic_path', type=str, default='none')
    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    args = parser.parse_args()
    # graphsaint_dic_path='args.graphsaint_dic_path'
    GSAINT_Dic = {}
    # with open(graphsaint_dic_path, 'rb') as handle:
    #     GSAINT_Dic = pickle.load(handle)

    # to_remove_pedicates=GSAINT_Dic['to_remove_pedicates']
    # to_remove_subject_object=GSAINT_Dic['to_remove_subject_object']
    # to_keep_edge_idx_map=GSAINT_Dic['to_keep_edge_idx_map']
    # GA_Index=GSAINT_Dic['GA_Index']
    to_remove_pedicates = []
    to_remove_subject_object = []
    to_keep_edge_idx_map = []
    GA_Index = 0
    # GA_dataset_name="KGTOSA_MAG_StarQuery_10M"
    # GA_dataset_name="KGTOSA_MAG_StarQuery"
    # GA_dataset_name = "KGTOSA_MAG_Paper_Discipline_StarQuery"
    MAG_datasets=[
                  # "ogbn-mag-ReleventSubGraph_Venue",
                  # "ogbn-mag-ReleventSubGraph_Discipline", 
                  # "KGTOSA_MAG_Paper_Discipline_StarQuery",
                  # "KGTOSA_MAG_Paper_Venue_StarQuery",
                  # "KGTOSA_MAG_Paper_Venue_BStarQuery",
                  # "KGTOSA_MAG_Paper_Discipline_BStarQuery",
                  # "KGTOSA_MAG_Paper_Venue_2HOPS_WO_Literals",
                  # "KGTOSA_MAG_Paper_Discipline_2HOPS_WO_Literals"
                  # "KGTOSA_DBLP_26K",
                  # "IMDB12K",
                  # "OGBN-MAG_StarQuery_PD",
                  # "OGBN-MAG_FM_PD",
                  "OGBN-MAG_PV_StarQuery",
                  "OGBN-MAG_PV_PathQuery",
                  "OGBN-MAG_PV_BStarQuery",
                  "OGBN-MAG_PV_BPathQuery",
                  "OGBN-MAG_PV_FM",
                ]
    print(args)
    gsaint_Final_Test = 0
    for GA_dataset_name in MAG_datasets:
        try:
            gsaint_start_t = datetime.datetime.now()
            ###################################Delete Folder if exist #############################
            # dir_path = "/shared_mnt/KGTOSA_MAG/" + GA_dataset_name
            # dir_path = "/MAG/MAG_2Hops/" + GA_dataset_name            
            dir_path="/shared_mnt/MAG_subsets/"+GA_dataset_name
            try:
                shutil.rmtree(dir_path)
                print("Folder Deleted")
            except OSError as e:
                print("Error Deleting : %s : %s" % (dir_path, e.strerror))
            #         ####################
            # dataset = PygNodePropPredDataset_hsh(name=GA_dataset_name, root='/shared_mnt/KGTOSA_MAG/',
            #                                      numofClasses=str(350))
            dataset = PygNodePropPredDataset_hsh(name=GA_dataset_name, root='/shared_mnt/MAG_subsets/', numofClasses=str(350))
            # dataset = PygNodePropPredDataset_hsh(name=GA_dataset_name, root='/shared_mnt/github_repos/KG-EaaS/DBLP_Usecase/',numofClasses=str(4))
            # dataset = PygNodePropPredDataset_hsh(name=GA_dataset_name, root='/shared_mnt/github_repos/KG-EaaS/DBLP_Usecase/',numofClasses=str(3))
            dataset_name = GA_dataset_name + "_GA_" + str(GA_Index)
            print("dataset_name=", dataset_name)
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
            subject_node = list(data.y_dict.keys())[0]
            split_idx = dataset.get_idx_split()
            # split_idx = dataset.get_idx_split('random')
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
                    # print("to remove keys=",keys)
                    to_remove_rels.append(keys)

            for keys, (row, col) in data.edge_index_dict.items():
                if (keys[1] in to_remove_pedicates):
                    # print("to remove keys=",keys)
                    to_remove_rels.append(keys)
                    to_remove_rels.append((keys[2], '_inv_' + keys[1], keys[0]))

            for elem in to_remove_rels:
                data.edge_index_dict.pop(elem, None)
                data.edge_reltype.pop(elem, None)

            for key in to_remove_subject_object:
                data.num_nodes_dict.pop(key, None)

            dic_results[dataset_name]["data"] = str(data)
            ##############add inverse edges ###################
            edge_index_dict = data.edge_index_dict
            key_lst = list(edge_index_dict.keys())
            for key in key_lst:
                r, c = edge_index_dict[(key[0], key[1], key[2])]
                edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

            # print("data after filter=",str(data))
            # print_memory_usage()
            print("data=",data)
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
            print(homo_data)
            start_t = datetime.datetime.now()
            print("dataset.processed_dir", dataset.processed_dir)
            kwargs = {'batch_size': 16384, 'num_workers': 64, 'persistent_workers': True}
            print("homo_data.train_mask=",len(homo_data.train_mask==False))
            train_loader = ShaDowKHopSampler(homo_data, depth=2, num_neighbors=3,
                                             node_idx=homo_data.train_mask,
                                             # node_idx = None,
                                             # node_idx=local2global[subject_node]
                                              **kwargs)
            end_t = datetime.datetime.now()
            print("Sampling time=", end_t - start_t, " sec.")
            dic_results[dataset_name]["GSaint_Sampling_time"] = (end_t - start_t).total_seconds()
            start_t = datetime.datetime.now()
            # Map informations to their canonical type.
            #######################intialize random features ###############################
            feat = torch.Tensor(data.num_nodes_dict[subject_node], 128)
            torch.nn.init.xavier_uniform_(feat)
            feat_dic = {subject_node: feat}
            ################################################################
            x_dict = {}
            # for key, x in data.x_dict.items():
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
            y_true = data.y_dict[subject_node]
            print("x_dict=", x_dict.keys())
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            if args.loadTrainedModel == 1:
                model.load_state_dict(torch.load("ogbn-DBLP-FM-GSaint.model"))
                model.eval()
                out = model.inference(x_dict, edge_index_dict, key2int)
                out = out[key2int[subject_node]]
                y_pred = out.argmax(dim=-1, keepdim=True).cpu()
                y_true = data.y_dict[subject_node]

                out_lst = torch.flatten(y_true).tolist()
                pred_lst = torch.flatten(y_pred).tolist()
                out_df = pd.DataFrame({"y_pred": pred_lst, "y_true": out_lst})
                # print(y_pred, data.y_dict[subject_node])
                # print(out_df)
                out_df.to_csv("GSaint_DBLP_conf_output.csv", index=None)
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
                        if loss == -1:
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
                gsaint_Final_Test = Final_Test.item()
                dic_results[dataset_name]["Final_Test"] = Final_Test.item()
                dic_results[dataset_name]["runs_count"] = args.runs
                dic_results[dataset_name]["avg_train_time"] = total_run_t
                dic_results[dataset_name]["rgcn_total_time"] = (gsaint_end_t - gsaint_start_t).total_seconds()
                pd.DataFrame(dic_results).transpose().to_csv(
                    "/shared_mnt/KGTOSA_MAG/ShadowSAINT_" + GA_dataset_name + "_Times.csv", index=False)
                # shutil.rmtree("/shared_mnt/DBLP/" + dataset_name)
                # torch.save(model.state_dict(), "/shared_mnt/DBLP/" + dataset_name + "_DBLP_conf__GSAINT_QM.model")
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("dataset_name Exception")
        del train_loader
    return gsaint_Final_Test


if __name__ == '__main__':
    print(graphSaint())






