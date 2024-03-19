from copy import copy
import json
import argparse
import os

import shutil
from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
import sys
# sys.path.insert(0, '/media/hussein/UbuntuData/GithubRepos/KGNET/GMLaaS/models/')
GMLaaS_models_path=sys.path[0].split("KGNET")[0]+"/KGNET/GMLaaS/models"
sys.path.insert(0,GMLaaS_models_path)
sys.path.append(os.path.join(os.path.abspath(__file__).split("KGNET")[0],'KGNET'))
from Constants import *

from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import  GraphSAINTRandomWalkSampler,ShaDowKHopSampler

# from GMLaaS.DataTransform.TSV_TO_PYG_dataset import inference_transform_tsv_to_PYG
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import psutil
import numpy
from collections import OrderedDict
from KGNET import KGNET
import pandas as pd
from evaluater import Evaluator
from custome_pyg_dataset import PygNodePropPredDataset_hsh
from resource import *
from logger import Logger
import faulthandler
faulthandler.enable()
import pickle
import warnings
import zarr
from kgwise_utils import generate_inference_subgraph
from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor

torch.multiprocessing.set_sharing_strategy('file_system')


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
""" Functions for Loading Mappings in Generate_inference_subgraph ()"""


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
                 dropout, num_nodes_dict, x_types, num_edge_types , state_dict = None):
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

        ### Initialize Flat L2 ###
        # self.index = faiss.IndexFlatL2(self.in_channels)
        # self.emb_mapping = {}

        # Create embeddings for all node types that do not come with features.
        if state_dict: #TODO This commented code works fine with demo DBLP TOSA sampled inference
            self.emb_dict = ParameterDict()
            for key in state_dict.keys():
                if num_nodes_dict[key] == state_dict[key].size()[0]:
                    self.emb_dict[key] = state_dict[key]
                else:
                    raise NotImplementedError

        else:
            self.emb_dict = {}

            # self.emb_dict = ParameterDict({
            #     f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            #     for key in set(node_types).difference(set(x_types))
            # })

        # self.emb_dict = ParameterDict({
        #     f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
        #     for key in set(node_types).difference(set(x_types))
        # })
        """ Comment ^ for better efficiency at inference"""

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        # self.reset_parameters()
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
            if x.is_sparse:
                h[mask] = x.index_select(0,local_node_idx[mask]).to_dense()
                continue
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            if emb.is_sparse:
                h[mask] = emb.index_select(0,local_node_idx[mask]).to_dense()
                continue
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



    def store_emb(self,model_name,root_path=os.path.join(KGNET_Config.trained_model_path,'emb_store')):
        path = os.path.join(root_path,model_name)
        if not os.path.exists(root_path):
            os.mkdir(root_path)

        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        emb_store = zarr.DirectoryStore(path)
        root = zarr.group(store=emb_store)

        emb_mapping = {}
        # Iterate over node types in emb_dict
        ptr = 0
        for key, val in self.emb_dict.items():
            # Convert PyTorch tensor to a NumPy array with a compatible data type
            val_np = val.cpu().detach().numpy()

            # Create a Zarr array for each node type
            emb_array = root.create(key, shape=val_np.shape, dtype=val_np.dtype, chunks=(128, -1)) #chunks=(val_np.shape[0], -1)
            emb_array[:] = val_np  # Assign the embeddings to the Zarr array

            # Update the mapping information
            # emb_mapping[key] = {'start': 0, 'end': val_np.shape[0]}
            emb_mapping[key] = (ptr,ptr+val_np.shape[0]-1)
            ptr+=val_np.shape[0]#+1
        # Save the mapping information
        emb_mapping_path = os.path.join(path, 'index.map')
        with open(emb_mapping_path, 'wb') as f:
            pickle.dump(emb_mapping, f)


    def Zarr_inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)

        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            if torch.all(row == torch.tensor([-1])) and torch.all(col == torch.tensor([-1])):
                continue
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            try:
                for keys, adj_t in adj_t_dict.items():
                    try:
                        src_key, target_key = keys[0], keys[-1]
                        out = out_dict[key2int[target_key]]
                        tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                        tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                        out.add_(tmp)
                    except Exception as e:
                        print(f'{e} , rel:{keys}')
                        adj_t = adj_t.to_torch_sparse_coo_tensor()
                        adj_t = torch.sparse.FloatTensor(
                            indices=adj_t._indices(),  # Reuse original indices
                            values=adj_t._values(),  # Reuse original values
                            size=(adj_t.size(dim=0), x_dict[key2int[src_key]].size(dim=0))  # Specify the new size
                        )
                        tmp = adj_t.mm(x_dict[key2int[src_key]])
                        tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]]).to_sparse()
                        out.add_(tmp)


            except Exception as e:
                print(e)
                raise Exception

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    if j in out_dict:
                        F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict

    def RW_sampling_inference(self, model, homo_data, x_dict, local2global, subject_node, node_type, target_mask, device='cpu'):
        model.eval()
        inference_nodes = local2global[subject_node]  # [-100:]
        inference_nodes = torch.zeros_like(inference_nodes, dtype=torch.bool)
        inference_nodes[target_mask] = True
        homo_data.inference_mask = torch.zeros(node_type.size(0), dtype=torch.bool)
        homo_data.inference_mask[local2global[subject_node][inference_nodes]] = True

        batch_size = 1024 * 2  # len_target #// 100 # inference_nodes.shape[0]#x_dict[2].shape[0]
        inference_loader = GraphSAINTRandomWalkSampler(homo_data,
                                              batch_size=batch_size,
                                              walk_length=6,
                                              num_steps=30,
                                              sample_coverage=0,
                                              save_dir="./"
                                              )
        """ target node """
        tensor_target_nodes = torch.empty(local2global[subject_node].size(0),dtype=torch.int64).fill_(-2)
        pbar = tqdm(total=len(inference_loader))
        for data in inference_loader:

            data = data.to(device)
            out = model(x_dict,
                        data.edge_index,
                        data.edge_attr,
                        data.node_type,
                        data.local_node_idx)
            # for key, value in zip(local2global[subject_node][data.local_node_idx[data.inference_mask]].tolist(),
            #                       out[data.inference_mask]):
            #     dict_target_nodes[key] = value.argmax(dim=-1, keepdim=True).cpu()
            tensor_target_nodes[[data.local_node_idx[data.inference_mask]]] = out[data.inference_mask].argmax(dim=-1,
                                                                                                              keepdim=True).squeeze()
            pbar.update(1)
        pbar.close()
        return tensor_target_nodes[target_mask]



    def sampled_inference(self, model, homo_data, x_dict, local2global, subject_node, node_type, target_mask, device='cpu'):
        model.eval()
        #x_dict[2] = x_dict[2][-100:]
        inference_nodes = local2global[subject_node]#[-100:]
        inference_nodes = torch.zeros_like(inference_nodes,dtype=torch.bool)
        # inference_nodes[-len_target:] = True
        # global target_masks
        inference_nodes[target_mask] = True
        homo_data.inference_mask = torch.zeros(node_type.size(0),dtype=torch.bool)
        homo_data.inference_mask[local2global[subject_node][inference_nodes]] = True
        homo_data.inference_mask
        batch_size = 1024*2#len_target #// 100 # inference_nodes.shape[0]#x_dict[2].shape[0]
        kwargs = {'batch_size': batch_size, 'num_workers': 0,}
        inference_loader = ShaDowKHopSampler(homo_data,depth=2,num_neighbors=20,
                                             node_idx=homo_data.inference_mask,
                                             batch_size=batch_size,
                                             num_workers=12,
                                             # **kwargs
                                             )


        pbar = tqdm(total=len(inference_loader))
        all_y_preds = []



        for data in inference_loader:

            data = data.to(device)
            out = model(x_dict,
                        data.edge_index,
                        data.edge_attr,
                        data.node_type,
                        data.local_node_idx)
            out = torch.index_select(out, 0, data.root_n_id)
            y_pred = out.argmax(dim=-1, keepdim=True).cpu()
            all_y_preds.append(y_pred)
            pbar.update(1)

        pbar.close()
        return torch.cat(all_y_preds,dim=0).squeeze(1)#y_pred.squeeze(1)
        # return y_pred.squeeze(1)


    def load_Zarr_emb(self,edge_index_dict,key2int,model_name,target,root_path=os.path.join(KGNET_Config.trained_model_path,'emb_store'),**kwargs):
        def create_indices(v, emb_size):
            repeated_v = torch.tensor(v).repeat_interleave(emb_size)   # Repeat each element in v emb_size times
            repeated_range = torch.arange(emb_size).repeat(len(v))     # Create a tensor containing the range [0, 1, 2, ..., emb_size-1] repeated len(v) times
            indices = torch.stack([repeated_v, repeated_range])        # Combine the repeated_v and repeated_range tensors to form the final 2D tensor
            return indices


        def get_to_load(edge_index_dict,):
            to_load = {}
            for k, v in edge_index_dict.items():
                if v.equal(torch.tensor([[-1], [-1]])):
                    continue
                src, _, dst = k
                if key2int[src] != target:
                    to_load.setdefault(key2int[src], set()).update(v[0].numpy())

                if key2int[dst] != target:
                    to_load.setdefault(key2int[dst], set()).update(v[1].numpy())
            return to_load

        path=os.path.join(root_path,model_name)

        global time_map_end,time_load_end
        time_map_start = datetime.datetime.now()
        # to_load = global_to_local(edge_index,local2global)
        to_load = get_to_load(edge_index_dict)

        time_map_end = (datetime.datetime.now() - time_map_start).total_seconds()
        emb_store = zarr.DirectoryStore(path)
        root = zarr.group(store=emb_store)

        time_load_start = datetime.datetime.now()
        for k,v in to_load.items():
            v = list(v)
            root_k_shape = root[k].shape
            emb_array = torch.tensor(root[k][v].astype(float))
            v = create_indices(v,root_k_shape[1])
            sparse_tensor = torch.sparse.FloatTensor(v, emb_array.view(-1), torch.Size(root_k_shape)).to(torch.float32)
            self.emb_dict[str(k)] = sparse_tensor
        time_load_end = (datetime.datetime.now()-time_load_start).total_seconds()
        warnings.warn('total time for mapping : {}'.format(time_load_end))

    def inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)

        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)


            for keys, adj_t in adj_t_dict.items():
                    src_key, target_key = keys[0], keys[-1]
                    out = out_dict[key2int[target_key]]
                    tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                    tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                    out.add_(tmp)

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict

dic_results = {}

def wise_SHsaint(device=0, num_layers=2, hidden_channels=64, dropout=0.5,
                 lr=0.005, epochs=2, runs=1, batch_size=1024*2, walk_length=2,
                 num_steps=10, loadTrainedModel=0, dataset_name="DBLP-Springer-Papers",
                 root_path="../../Datasets/", output_path="./", include_reverse_edge=True,
                 n_classes=50, emb_size=128, label_mapping={}, target_mapping={},target_rel='', modelID='',
                 target_masks=None,target_masks_inf = None):
    def train(epoch):
        model.train()
        pbar = tqdm(total=len(train_loader))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,data.local_node_idx)
            # out = out[data.train_mask]
            out = torch.index_select(out, 0, data.root_n_id)
            y = data.y.squeeze(1)
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

    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    process_start = datetime.datetime.now()
    dataset_name = dataset_name

    GNN_datasets=[dataset_name]


    for GNN_dataset_name in GNN_datasets:
        # try:
        gsaint_start_t = datetime.datetime.now()
        ###################################Delete Folder if exist #############################
        dir_path=root_path+GNN_dataset_name

        """ TOSA Subgraph Inference"""
        # if loadTrainedModel == 1:
        #     # dir_path = generate_inference_subgraph
        #     output = generate_inference_subgraph(master_ds_name=GNN_dataset_name,
        #                                            target_rel_uri=target_rel) # TODO: parameterize for generality
        #     if len(output) == 3:
        #         dir_path, target_masks,target_masks_inf = output
        #     else:
        #         dir_path, target_masks = output
        #         target_masks_inf = None
        #
        #     GNN_dataset_name = dir_path.split('/')[-1]
        #     root_path = KGNET_Config.inference_path
        """ ************************* """

        try:
            shutil.rmtree(dir_path)
            print("Folder Deleted")
        except OSError as e:
            print("Error Deleting : %s : %s" % (dir_path, e.strerror))
        #         ####################
        """ BASELINE COMPLETE Graph TSV Transformation"""
        # if loadTrainedModel == 1 :#and not os.path.exists(os.path.join(KGNET_Config.datasets_output_path,GNN_dataset_name+'.zip')):
        #     transform_start = datetime.datetime.now()
        #     if os.path.exists(os.path.join(KGNET_Config.datasets_output_path,GNN_dataset_name+'.tsv')):
        #         transform_tsv_to_PYG(dataset_name=dataset_name,
        #                                        dataset_name_csv=dataset_name,
        #                                        dataset_types=r'/home/afandi/GitRepos/KGNET/Datasets/dblp2022_Types (rec).csv',
        #                                        # TODO: Replace with arg based file
        #                                        target_rel='publishedIn', # TODO: Replace with dynamic target
        #                                        split_rel="yearOfPublication",
        #                                        similar_target_rels = [],
        #                                        split_rel_train_value=2019,
        #                                        split_rel_valid_value=2020,
        #                                        Header_row = True,
        #                                        output_root_path=KGNET_Config.datasets_output_path)
        #         transform_end = (datetime.datetime.now() - transform_start).total_seconds()
        #         print(f'**** TRANSFORMATION TIME : {transform_end} s **********')
        #     else:
        #         raise Exception ('TSV file not found')


        dataset = PygNodePropPredDataset_hsh(name=GNN_dataset_name, root=root_path, numofClasses=str(n_classes))

        print("dataset_name=", dataset_name)
        dic_results = {}
        dic_results["GNN_Method"] = GNN_Methods.Graph_SAINT
        dic_results["to_keep_edge_idx_map"] = []
        dic_results["dataset_name"] = dataset_name


        print(getrusage(RUSAGE_SELF))
        start_t = datetime.datetime.now()
        data = dataset[0]
        # global subject_node
        subject_node = list(data.y_dict.keys())[0]
        if loadTrainedModel == 0:
            split_idx = dataset.get_idx_split()
        end_t = datetime.datetime.now()
        print("dataset init time=", end_t - start_t, " sec.")
        dic_results["dataset_load_time"] = (end_t - start_t).total_seconds()
        evaluator = Evaluator(name='ogbn-mag')

        start_t = datetime.datetime.now()
        # We do not consider those attributes for now.
        data.node_year_dict = None
        data.edge_reltype_dict = None

        to_remove_rels = []


        for elem in to_remove_rels:
            data.edge_index_dict.pop(elem, None)
            data.edge_reltype.pop(elem, None)

        edge_index_dict = data.edge_index_dict
        ##############add inverse edges ###################
        if include_reverse_edge:
            key_lst = list(edge_index_dict.keys())
            for key in key_lst:
                r, c = edge_index_dict[(key[0], key[1], key[2])]
                edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])


        out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
        edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
        """ Removing -1 -1 masks from the edge index and edge type"""
        mask = ~(edge_index == -1).any(dim=0)
        edge_index = edge_index[:,mask]
        edge_type = edge_type[mask]
        """ ******************************** """
        homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                         node_type=node_type, local_node_idx=local_node_idx,
                         num_nodes=node_type.size(0))

        homo_data.y = node_type.new_full((node_type.size(0), 1), -2)
        try:
            homo_data.y[local2global[subject_node][target_masks]] = data.y_dict[subject_node]
        except:
            warnings.warn('Warning! Mismatch in homo_data.y')

        if loadTrainedModel == 0:
            homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
            homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True
            start_t = datetime.datetime.now()
            print("dataset.processed_dir", dataset.processed_dir)
            kwargs = {'batch_size': batch_size, 'num_workers': 64, 'persistent_workers': True}
            train_loader = ShaDowKHopSampler(homo_data, depth=2, num_neighbors=3,
                                             node_idx=homo_data.train_mask,
                                              **kwargs)

        start_t = datetime.datetime.now()
        # Map informations to their canonical type.
        #######################intialize random features ###############################
        # global target_masks

        # if 'target_masks' not in globals():
        #     target_csv = os.path.join(KGNET_Config.datasets_output_path, 'TARGET_18000d.csv')
        #     original_map = os.path.join(KGNET_Config.datasets_output_path, GNN_dataset_name, 'mapping',
        #                                 subject_node + '_entidx2name.csv.gz')
        #     # target_csv = os.path.join(KGNET_Config.datasets_output_path, GNN_dataset_name, 'mapping',
        #     #                             subject_node + '_entidx2name.csv')
        #     # original_map = '/home/afandi/GitRepos/KGNET/Datasets/DBLP_Paper_Venue_FM_FTD_d1h1/mapping/rec_entidx2name.csv'
        #     target_df = pd.read_csv(target_csv, header=None, names=['ent name'])
        #     original_df = pd.read_csv(original_map)
        #     target_masks = pd.merge(original_df, target_df, on='ent name', how='inner', suffixes=('_orig', '_inf'))[
        #         'ent idx'].to_list()
        #     del target_csv, original_map, target_df, original_df

        if loadTrainedModel == 1 and target_masks is not None:
            feat = torch.sparse.FloatTensor(size=(data.num_nodes_dict[subject_node],emb_size))
            # feat = torch.Tensor(data.num_nodes_dict[subject_node], emb_size)
        #
        else:
            feat = torch.Tensor(data.num_nodes_dict[subject_node], emb_size)
        # torch.nn.init.xavier_uniform_(feat)
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
        device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'


        process_end = (datetime.datetime.now() - process_start).total_seconds()
        process_ram = getrusage(RUSAGE_SELF).ru_maxrss/ (1024 * 1024)
        if loadTrainedModel == 1:
            with torch.no_grad():
                y_true = data.y_dict[subject_node]
                global download_end_time,time_map_end, time_load_end
                start_t = datetime.datetime.now()
                trained_model_path = KGNET_Config.trained_model_path + modelID
                model_name = gen_model_name(dataset_name, dic_results["GNN_Method"])
                model_params_path = trained_model_path.replace('.model', '.param')

                with open(model_params_path, 'rb') as f:
                    dict_model_param = pickle.load(f)

                if not target_mapping:
                    target_mapping = pd.read_csv(os.path.join(dir_path, 'mapping', f'{subject_node}_entidx2name.csv.gz'),compression='gzip')
                    target_mapping = target_mapping.set_index('ent idx')['ent name'].to_dict()
                if not label_mapping:
                    label_mapping = pd.read_csv(os.path.join(dir_path, 'mapping', 'labelidx2labelname.csv.gz'))
                    label_mapping = label_mapping.set_index('label idx')['label name'].to_dict()

                model = RGCN(dict_model_param['emb_size'],
                             dict_model_param['hidden_channels'],
                             dict_model_param['dataset.num_classes'],
                             dict_model_param['num_layers'],
                             dict_model_param['dropout'],
                             #num_nodes_dict,
                             dict_model_param['num_nodes_dict'],
                             dict_model_param['list_x_dict_keys'],
                             dict_model_param['len_edge_index_dict_keys'])

                SAMPLED_INFERENCE = False
                RW_SAMPLER = True

                """ ************ RANDOM WALK WISE Inference ****************"""

                if RW_SAMPLER:
                    model.load_state_dict(torch.load(trained_model_path), strict=False)
                    model.load_Zarr_emb(edge_index_dict, key2int, modelID.split('.model')[0],
                                        target=key2int[subject_node],
                                        num_target_nodes=num_nodes_dict[key2int[subject_node]],
                                        target_masks=target_masks)
                    y_pred = model.RW_sampling_inference(model, homo_data, x_dict, local2global, subject_node, node_type,
                                                     target_mask=target_masks)
                    y_pred = y_pred.unsqueeze(1)
                    dic_results["InferenceTime"] = (end_t - start_t).total_seconds()
                    print(f'y_true: {y_true.size()}\ny_pred: {y_pred.size()}\nTarget masks: {len(target_masks)}')

                    if y_true.size()[0] == y_pred.size()[0] and target_masks_inf:
                        y_true=y_true[target_masks_inf]
                        y_pred = y_pred[target_masks_inf]
                    elif y_true.size()[0] > y_pred.size()[0]:
                        y_true = y_true[target_masks]
                    elif y_true.size()[0] < y_pred.size()[0]:
                        y_pred = y_pred[target_masks]

                    test_acc = evaluator.eval({
                        'y_true': y_true,
                        'y_pred': y_pred,
                    })['acc']

                    total_time = process_end+dic_results["InferenceTime"]
                    if 'download_end_time' in locals() or 'download_end_time' in globals():
                        process_end = process_end-download_end_time
                    else:
                        download_end_time = 0

                    print('*' * 8, '\tRAM USAGE BEFORE Model/Inference:\t', process_ram, ' GB')
                    print('*'*8, '\tDOWNLOAD TIME: ', download_end_time, 's')
                    print('*'*8, '\tPROCESSING TIME: ', process_end, 's')
                    # print('*' * 8, '\tZARR MAPPING:\t\t',time_map_end, 's')
                    # print('*' * 8, '\tZARR LOADING:\t\t',time_load_end, 's')
                    print('*'*8,'\tInference TIME: ',dic_results["InferenceTime"],'s')
                    print('*'*8,'\tTotal TIME: ', total_time ,'s',)
                    print('*'*8,'\tMax RAM Usage: ',getrusage(RUSAGE_SELF).ru_maxrss/ (1024 * 1024),)
                    print('*'*8,'\tAccracy: ',test_acc,)
                    print('*'*8,'\tClasses in DS: ',len(y_true.unique()))#,'\n',y_true.value_counts())
                    ### For KGNET inference, return labels ###

                    dict_pred = {}
                    for i, pred in enumerate(y_pred.flatten()):
                        dict_pred[target_mapping[i]] = label_mapping[pred.item()]
                    dic_results['y_pred'] = dict_pred
                    dic_results['totalTime'] = total_time
                    return dic_results

                """ ************ WISE Sampled Inference ****************"""
                if SAMPLED_INFERENCE:
                    model.load_state_dict(torch.load(trained_model_path),strict=False)
                    model.load_Zarr_emb(edge_index_dict,key2int,modelID.split('.model')[0],target=key2int[subject_node],num_target_nodes = num_nodes_dict[key2int[subject_node]],target_masks = target_masks )

                    LEN_TARGET = y_true.size()[0]
                    print(' Sampled Inference ')
                    y_pred = model.sampled_inference(model, homo_data, x_dict, local2global, subject_node, node_type,
                                                     target_mask=target_masks)#y_true.size()[0])
                    y_pred = y_pred.unsqueeze(1)
                    end_t = datetime.datetime.now()
                    dic_results["InferenceTime"] = (end_t - start_t).total_seconds()
                    print(f'y_true: {y_true.size()}\ny_pred: {y_pred.size()}\nTarget masks: {len(target_masks)}')

                    if y_true.size()[0] == y_pred.size()[0] and target_masks_inf:
                        y_true=y_true[target_masks_inf]
                        y_pred = y_pred[target_masks_inf]
                    elif y_true.size()[0] > y_pred.size()[0]:
                        y_true = y_true[target_masks]
                    elif y_true.size()[0] < y_pred.size()[0]:
                        y_pred = y_pred[target_masks]

                    test_acc = evaluator.eval({
                        'y_true': y_true,
                        'y_pred': y_pred,
                    })['acc']

                    total_time = process_end+dic_results["InferenceTime"]
                    if 'download_end_time' in locals() or 'download_end_time' in globals():
                        process_end = process_end-download_end_time
                    else:
                        download_end_time = 0

                    print('*' * 8, '\tRAM USAGE BEFORE Model/Inference:\t', process_ram, ' GB')
                    print('*'*8, '\tDOWNLOAD TIME: ', download_end_time, 's')
                    print('*'*8, '\tPROCESSING TIME: ', process_end, 's')
                    # print('*' * 8, '\tZARR MAPPING:\t\t',time_map_end, 's')
                    # print('*' * 8, '\tZARR LOADING:\t\t',time_load_end, 's')
                    print('*'*8,'\tInference TIME: ',dic_results["InferenceTime"],'s')
                    print('*'*8,'\tTotal TIME: ', total_time ,'s',)
                    print('*'*8,'\tMax RAM Usage: ',getrusage(RUSAGE_SELF).ru_maxrss/ (1024 * 1024),)
                    print('*'*8,'\tAccracy: ',test_acc,)
                    print('*'*8,'\tClasses in DS: ',len(y_true.unique()))#,'\n',y_true.value_counts())
                    ### For KGNET inference, return labels ###

                    dict_pred = {}
                    for i, pred in enumerate(y_pred.flatten()):
                        dict_pred[target_mapping[i]] = label_mapping[pred.item()]
                    dic_results['y_pred'] = dict_pred
                    dic_results['totalTime'] = total_time
                    return dic_results
                """ ************ ********************* ****************"""


                """ FULL BATCH INFERENCE """
                """ Load Zarr Embeddings (load_min_emb)"""
                load_start = datetime.datetime.now()
                model.load_state_dict(torch.load(trained_model_path),strict=False)
                model.load_Zarr_emb(edge_index_dict,key2int,modelID.split('.model')[0],target=key2int[subject_node],num_target_nodes = num_nodes_dict[key2int[subject_node]],target_masks = target_masks )
                load_end = (datetime.datetime.now() - load_start).total_seconds()
                print('Loaded Graph Saint Model!')
                model.eval()
                """ Inference type"""
                # out = model.inference(x_dict, edge_index_dict, key2int)
                out = model.Zarr_inference(x_dict, edge_index_dict, key2int)
                """ ************** """

                out = out[key2int[subject_node]]
                y_pred = out.argmax(dim=-1, keepdim=True).cpu()#.flatten().tolist()
                end_t = datetime.datetime.now()
                dic_results["InferenceTime"] = (end_t - start_t).total_seconds()
                init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss

                # y_true = data.y_dict[subject_node]
                print(f'y_true: {y_true.size()}\ny_pred: {y_pred.size()}\nTarget masks: {len(target_masks)}')
                if y_true.size()[0] > y_pred.size()[0]:
                    y_true=y_true[target_masks]
                elif y_true.size()[0] < y_pred.size()[0]:
                    y_pred = y_pred[target_masks]

                test_acc = evaluator.eval({
                    'y_true': y_true,#[target_masks],
                    'y_pred': y_pred,#[target_masks],
                })['acc']
                print('*' * 8, '\tRAM USAGE BEFORE Model/Inference:\t', process_ram, ' GB')
                print('*'*8, '\tPROCESSING TIME:\t',process_end, 's')
                # print('*' * 8, '\tZARR MAPPING:\t\t',time_map_end, 's')
                # print('*' * 8, '\tZARR LOADING:\t\t',time_load_end, 's')
                # print('*' * 8, '\tTOTAL MODEL LOAD:\t',load_end, 's')
                print('*'*8,'\tInference TIME:\t\t',dic_results["InferenceTime"]-load_end,'s')
                print('*'*8,'\tTotal TIME:\t\t',process_end+dic_results["InferenceTime"],'s',)
                print('*'*8,'\tMax RAM Usage:\t\t',getrusage(RUSAGE_SELF).ru_maxrss/ (1024 * 1024),' GB')
                print('*'*8,'\tAccuracy:\t\t',test_acc,)
                print('*'*8,'\tClasses in DS:\t\t',len(y_true.unique()),)
                total_time = process_end + dic_results["InferenceTime"]
                ### For KGNET inference, return labels ###
                dict_pred = {}
                for i, pred in enumerate(y_pred.flatten()):
                    dict_pred[target_mapping[i]] = label_mapping[pred.item()]
                dic_results['y_pred'] = dict_pred
                dic_results['totalTime'] = total_time
                return dic_results

            return dic_results
        else:
            model = RGCN(emb_size, hidden_channels, dataset.num_classes, num_layers,
                         dropout, num_nodes_dict, list(x_dict.keys()),
                         len(edge_index_dict.keys())).to(device)

            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            # y_true = data.y_dict[subject_node]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            model_name = gen_model_name(dataset_name, dic_results["GNN_Method"])

            print("start test")
            logger = Logger(runs)
            test()  # Test if inference on GPU succeeds.
            total_run_t = 0
            for run in range(runs):
                start_t = datetime.datetime.now()
                model.reset_parameters()
                for epoch in range(1, 1 + epochs):
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
            print('Calculating inference time')
            with torch.no_grad():
                time_inference_start = datetime.datetime.now()
                model.inference(x_dict, edge_index_dict, key2int)
            dic_results['Inference_Time'] = (datetime.datetime.now() - time_inference_start).total_seconds()

            total_run_t = (total_run_t + 0.00001) / runs
            gsaint_end_t = datetime.datetime.now()
            Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
            model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            gnn_hyper_params_dict = {"device": device, "num_layers": num_layers, "hidden_channels": hidden_channels,
                                     "dropout": dropout, "lr": lr, "epochs": epochs, "runs": runs,
                                     "batch_size": batch_size, "walk_length": walk_length, "num_steps": num_steps,
                                     "emb_size": emb_size, "dataset_num_classes":dataset.num_classes,}
            logger = Logger(runs, gnn_hyper_params_dict)

            dic_results["gnn_hyper_params"] = gnn_hyper_params_dict
            dic_results['model_name'] = model_name
            dic_results["init_ru_maxrss"] = init_ru_maxrss
            dic_results["model_ru_maxrss"] = model_loaded_ru_maxrss
            dic_results["model_trained_ru_maxrss"] = model_trained_ru_maxrss
            dic_results["Highest_Train_Acc"] = Highest_Train.item()
            dic_results["Highest_Valid_Acc"] = Highest_Valid.item()
            dic_results["Final_Train_Acc"] = Final_Train.item()
            gsaint_Final_Test = Final_Test.item()
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
            logs_path = os.path.join(output_path)
            model_path = os.path.join(output_path)
            create_dir([logs_path,model_path])

            with open(os.path.join(logs_path, model_name +'_log.metadata'), "w") as outfile:
                json.dump(dic_results, outfile)
            model.store_emb(model_name=model_name)
            """REMOVING EMB DICT FROM MODEL"""
            model.emb_dict = None
            """**************************"""
            torch.save(model.state_dict(), os.path.join(model_path , model_name)+".model")
            with open (os.path.join(model_path , model_name)+".param",'wb') as f:
                pickle.dump(dict_model_param,f)

            dic_results["data_obj"] = data.to_dict()
        # except Exception as e:
        #     print(e)
        #     print(traceback.format_exc())
        #     print("dataset_name Exception")
        del train_loader
    return dic_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphShadowSAINT)')
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
    parser.add_argument('--dataset_name', type=str, default="DBLP_Paper_Venue_FM_FTD_d1h1")#  DBLP_Paper_Venue_FM_FTD_d1h1_2021_2 ##
    parser.add_argument('--root_path', type=str, default= KGNET_Config.datasets_output_path)
    parser.add_argument('--output_path', type=str, default=KGNET_Config.trained_model_path)
    parser.add_argument('--include_reverse_edge', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=50)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--modelID',type=str, default='DBLP_FTD_d1h1_Zarr.model') #DBLP_Paper_Venue_FM_FTD_d1h1_PRIME_1000_GA_0_ShadowSaint Zarr = DBLP_d1h1_2021_Zarr_e10.model v2=DBLP_FTD_d1h1_Zarr
    args = parser.parse_args()
    print(args)
    print(wise_SHsaint(args.device, args.num_layers, args.hidden_channels, args.dropout, args.lr, args.epochs, args.runs, args.batch_size, args.walk_length, args.num_steps, args.loadTrainedModel, args.dataset_name, args.root_path, args.output_path, args.include_reverse_edge, args.n_classes, args.emb_size, modelID=args.modelID))

""" FOR ZARR INFERENCE"""
"""
1. Disable emb dict generation for non target nodes at model's definition
2. Change x_dict features from dense tensor to sparse tensor.
3. Disable x_dict's xavier initilization  
4. Call model.load_Zarr_emb() to load embds
5. Call model.Zarr_inference() to perform inference (full batch)
"""
""" FOR INDUCTIVE INFERENCE"""
"""
1. Uncomment the 'TOSA Subgraph Inference' part
2. Make sure you have a target CSV containing URI of the nodes to infer
3. Make sure you have the dataset D that the model was trained on.
4. Add the new 'unseen' target nodes in D's mapping and set the num_of_nodes (D/raw/) of target nodes accordingly

"""




