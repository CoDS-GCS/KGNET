import os.path as osp
from typing import Optional

import torch
from torch_sparse import SparseTensor
from tqdm import tqdm
import random
from torch_geometric.loader import  GraphSAINTRandomWalkSampler as GraphSAINTSampler
class GraphSAINTTaskBaisedRandomWalkSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).

    Args:
        walk_length (int): The length of each random walk.
    """
    def __init__(self, data, batch_size: int, walk_length: int, Subject_indices,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        self.Subject_indices=Subject_indices
        self.index=0
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)
    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __sample_nodes__(self, batch_size):
        start = torch.randint(min(self.Subject_indices), max(self.Subject_indices), (batch_size,), dtype=torch.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)
class GraphSAINTTaskWeightedRandomWalkSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).

    Args:
        walk_length (int): The length of each random walk.
    """
    def __init__(self, data, batch_size: int, walk_length: int, NodesWeightDic,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        self.NodesWeightDic = NodesWeightDic
        self.index = 0
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)
    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __sample_nodes__(self, batch_size):
        start = None
        keys_nodes_lst=[]
        for key in self.NodesWeightDic.keys():
            keys_nodes_lst.append(torch.randint(key[0], key[1], (int(batch_size * self.NodesWeightDic[key]),), dtype=torch.long))
        start=torch.cat(keys_nodes_lst,dim=0)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)