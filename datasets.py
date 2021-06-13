from torch_geometric.datasets import Planetoid
from ogb.linkproppred import PygLinkPropPredDataset
import torch
import torch_geometric.transforms as T

class Datasets:
    CiteSeer = 'CiteSeer'
    PubMed = 'PubMed'
    Cora = 'Cora'
    DrugDrugInteraction = 'ogbl-ddi'
    ProteinProteinAssociation = 'ogbl-ppa'

def load_dataset(path, name, device):
    if name.startswith('ogbl-'):  # https://ogb.stanford.edu/docs/linkprop/ Topic: Pytorch Geometric Loader
        dataset = PygLinkPropPredDataset(name = name, transform=T.ToSparseTensor())
        split_edge = dataset.get_edge_split()
        train_edge, valid_edge, test_edge = split_edge['train'], split_edge['valid'], split_edge['test']

        graph = dataset[0]
        idx = torch.randperm(train_edge['edge'].size(0))
        idx = idx[:valid_edge['edge'].size(0)]
        split_edge['eval_train'] = {'edge': train_edge['edge'][idx]}
        graph.adj_t.to(device)
        return (graph, split_edge)
    else:
        # TODO
        return Planetoid(path, name = name, split='public')
