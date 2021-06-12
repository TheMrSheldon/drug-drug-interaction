from torch_geometric.datasets import Planetoid
from ogb.linkproppred import LinkPropPredDataset

class Datasets:
    CiteSeer = 'CiteSeer';
    PubMed = 'PubMed';
    Cora = 'Cora';
    DrugDrugInteraction = 'ogbl-ddi';
    ProteinProteinAssociation = 'ogbl-ppa';

def load_dataset(path, name):
    if name.startswith('ogbl-'):
        return LinkPropPredDataset(name = name)
    else:
        return Planetoid(path, name = name, split='public')
