from torch_geometric.datasets import Planetoid
from ogb.linkproppred import PygLinkPropPredDataset
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from ogb.linkproppred import Evaluator
from enum import Enum
from collections import namedtuple

class Metrics:
    MRR='mrr'
    def HitsAt(num):
        return f'hits@{num}'

Dataset = namedtuple("Dataset", "name metric")
class Datasets:
    CiteSeer = Dataset('CiteSeer', Metrics.MRR)
    PubMed = Dataset('PubMed', Metrics.MRR)
    Cora = Dataset('Cora', Metrics.MRR)
    DrugDrugInteraction = Dataset('ogbl-ddi', Metrics.HitsAt(20))
    ProteinProteinAssociation = Dataset('ogbl-ppa', Metrics.HitsAt(20))
    
class EmbeddingModel(Enum):
    Raw = None
    DeepWalk = 'deepwalk'
    Node2Vec = 'node2vec'

def load_dataset(path, dataset: Dataset, device, embeddingmodel: EmbeddingModel=EmbeddingModel.Raw):
    if dataset.name.startswith('ogbl-'): # https://ogb.stanford.edu/docs/linkprop/#pyg
        dataset = PygLinkPropPredDataset(name=dataset.name, transform=T.ToSparseTensor())
        split_edge = dataset.get_edge_split()
        train_edge, valid_edge, test_edge = split_edge['train'], split_edge['valid'], split_edge['test']
        graph = dataset[0].to(device, non_blocking=True)
        idx = torch.randperm(train_edge['edge'].size(0))
        idx = idx[:valid_edge['edge'].size(0)]
        split_edge['eval_train'] = {'edge': train_edge['edge'][idx]}
        
        return (graph, split_edge)
    else:
        dataset = Planetoid(path, name=dataset.name, transform=T.ToSparseTensor(remove_edge_index=False))
        graph = dataset[0].to(device)
        data = train_test_split_edges(graph)
        
        train_edge, valid_edge, test_edge = data.train_pos_edge_index.t(), data.val_pos_edge_index.t(), data.test_pos_edge_index.t()
        idx = torch.randperm(train_edge.size(0))
        idx = idx[:valid_edge.size(0)]

        split_edge = {
            'train': {'edge': train_edge},
            'valid': {'edge': valid_edge, 'edge_neg': data.val_neg_edge_index.t()},
            'test': {'edge': test_edge, 'edge_neg': data.test_neg_edge_index.t()},
            'eval_train': {'edge': train_edge[idx]}
        }
        return (graph, split_edge)

class __tmp_evaluator__(Evaluator):
    def __init__(self, metric) -> None:
        self.eval_metric = metric

def create_evaluator(dataset: Dataset):
    if dataset.name.startswith('ogbl-'):
        return Evaluator(dataset.name)
    else:
        return __tmp_evaluator__(dataset.metric)
        