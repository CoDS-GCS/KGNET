import os
import os.path as osp
from typing import Callable, List, Optional

import pandas as pd
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class RelLinkPredDataset(InMemoryDataset):
    r"""The relational link prediction datasets from the
    `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.
    Training and test splits are given by sets of triplets.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"FB15k-237"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 14,541
          - 544,230
          - 0
          - 0
    """

    urls = {
        'FB15k-237': ('https://raw.githubusercontent.com/MichSchli/'
                      'RelationPrediction/master/data/FB-Toutanova')
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        #assert name in ['FB15k-237','biokg_DDI_lp']   #Modified
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self) -> int:
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self) -> str:
        #return os.path.join(self.root, self.name, 'raw')
        return os.path.join(self.root, self.name,)

    @property
    def processed_dir(self) -> str:
        # return os.path.join(self.root, self.name, 'processed')
        return os.path.join(self.root, self.name,)

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'entities.dict', 'relations.dict', 'test.txt', 'train.txt',
            'valid.txt'
        ]

    def download(self):
        print('Downloading bypass')
        # pass
        # for file_name in self.raw_file_names:
        #     download_url(f'{self.urls[self.name]}/{file_name}', self.raw_dir)

    def process(self):

        entities_df=pd.read_csv(osp.join(self.raw_dir, 'entities.dict'),header=None,sep="\t")
        entities_dict=dict(zip(entities_df[1], entities_df[0]))

        relations_df = pd.read_csv(osp.join(self.raw_dir, 'relations.dict'), header=None, sep="\t")
        relations_dict = dict(zip(relations_df[1], relations_df[0]))

        kwargs = {}
        for split in ['train', 'valid', 'test']:
            df = pd.read_csv(osp.join(self.raw_dir, f'{split}.txt'), header=None, sep="\t")
            df=df[(df[0].isin(entities_dict.keys())) & (df[2].isin(entities_dict.keys())) &  (df[1].isin(relations_dict.keys())) ]
            src= df[0].apply(lambda x:entities_dict[x]).tolist()
            rel= df[1].apply(lambda x: relations_dict[x]).tolist()
            dst= df[2].apply(lambda x: entities_dict[x]).tolist()
            kwargs[f'{split}_edge_index'] = torch.tensor([src, dst])
            kwargs[f'{split}_edge_type'] = torch.tensor(rel)


        # For message passing, we add reverse edges and types to the graph:
        row, col = kwargs['train_edge_index']
        edge_type = kwargs['train_edge_type']
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_type = torch.cat([edge_type, edge_type + len(relations_dict)])

        data = Data(num_nodes=len(entities_dict), edge_index=edge_index,
                    edge_type=edge_type, **kwargs)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save((self.collate([data])), self.processed_paths[0])

    def process_txt(self):
        with open(osp.join(self.raw_dir, 'entities.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: int(value) for value, key in lines}

        with open(osp.join(self.raw_dir, 'relations.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            relations_dict = {key: int(value) for value, key in lines}

        kwargs = {}
        for split in ['train', 'valid', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.txt'), 'r') as f:
                lines = [row.split('\t') for row in f.read().split('\n')[:-1] if row.split('\t')[2]!='' ]

                src = [entities_dict[row[0]] for row in lines]
                rel = [relations_dict[row[1]] for row in lines]
                dst = [entities_dict[row[2]] for row in lines]
                kwargs[f'{split}_edge_index'] = torch.tensor([src, dst])
                kwargs[f'{split}_edge_type'] = torch.tensor(rel)

        # For message passing, we add reverse edges and types to the graph:
        row, col = kwargs['train_edge_index']
        edge_type = kwargs['train_edge_type']
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_type = torch.cat([edge_type, edge_type + len(relations_dict)])

        data = Data(num_nodes=len(entities_dict), edge_index=edge_index,
                    edge_type=edge_type, **kwargs)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save((self.collate([data])), self.processed_paths[0])
    def __repr__(self) -> str:
        return f'{self.name}()'
