from torch_geometric.datasets import Planetoid
from ogb.linkproppred import PygLinkPropPredDataset
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges, to_dense_adj
from ogb.linkproppred import Evaluator, evaluate

class Datasets:
    CiteSeer = 'CiteSeer'
    PubMed = 'PubMed'
    Cora = 'Cora'
    DrugDrugInteraction = 'ogbl-ddi'
    ProteinProteinAssociation = 'ogbl-ppa'

def load_dataset(path, name, device):
    if name.startswith('ogbl-'): # https://ogb.stanford.edu/docs/linkprop/#pyg
        dataset = PygLinkPropPredDataset(name=name, transform=T.ToSparseTensor())
        split_edge = dataset.get_edge_split()
        train_edge, valid_edge, test_edge = split_edge['train'], split_edge['valid'], split_edge['test']
        graph = dataset[0].to(device)
        idx = torch.randperm(train_edge['edge'].size(0))
        idx = idx[:valid_edge['edge'].size(0)]
        split_edge['eval_train'] = {'edge': train_edge['edge'][idx]}

        
        #print(split_edge)
        
        return (graph, split_edge)
    else:
        dataset = Planetoid(path, name=name, transform=T.ToSparseTensor(remove_edge_index=False))
        graph = dataset[0].to(device)
        data = train_test_split_edges(graph)
        
        train_edge, valid_edge, test_edge = data.train_pos_edge_index.t(), data.val_pos_edge_index.t(), data.test_pos_edge_index.t()
        idx = torch.randperm(train_edge.size(0))
        idx = idx[:valid_edge.size(0)]

        #print(data)

        split_edge = {
            'train': {'edge': train_edge},
            'valid': {'edge': valid_edge, 'edge_neg': data.val_neg_edge_index.t()},
            'test': {'edge': test_edge, 'edge_neg': data.test_neg_edge_index.t()},
            'eval_train': {'edge': train_edge[idx]}
        }

        #print(split_edge)
        return (graph, split_edge)

class __tmp_evaluator__(Evaluator):
    def __init__(self, metric) -> None:
        self.eval_metric = metric

class Metrics:
    MRR='mrr'
    def HitsAt(num):
        return f'hits@{num}'

def create_evaluator(name, metric):
    if name.startswith('ogbl-'):
        return Evaluator(name)
    else:
        return __tmp_evaluator__(metric)
        