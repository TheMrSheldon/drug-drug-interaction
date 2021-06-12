from torch_geometric.datasets import Planetoid
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T

class Datasets:
    CiteSeer = 'CiteSeer';
    PubMed = 'PubMed';
    Cora = 'Cora';
    DrugDrugInteraction = 'ogbl-ddi';
    ProteinProteinAssociation = 'ogbl-ppa';

def load_dataset(path, name):
    if name.startswith('ogbl-'):
        return PygLinkPropPredDataset(name = name, transform=T.ToSparseTensor())
    else:
        return Planetoid(path, name = name, split='public')
