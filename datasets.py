from torch_geometric.datasets import Planetoid
from ogb.linkproppred import PygLinkPropPredDataset
import torch
import torch_geometric.transforms as T

class Datasets:
    CiteSeer = 'CiteSeer';
    PubMed = 'PubMed';
    Cora = 'Cora';
    DrugDrugInteraction = 'ogbl-ddi';
    ProteinProteinAssociation = 'ogbl-ppa';

def load_dataset(path, name, device):
    if name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name = name, transform=T.ToSparseTensor())
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        idx = torch.randperm(split_edge['train']['edge'].size(0))
        idx = idx[:split_edge['valid']['edge'].size(0)]
        split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
        adj_t = data.adj_t.to(device)
        return (data.num_nodes, adj_t, split_edge)
    else:
        return Planetoid(path, name = name, split='public')
