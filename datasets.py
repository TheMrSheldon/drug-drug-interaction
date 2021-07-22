from torch_geometric.datasets import Planetoid
from ogb.linkproppred import PygLinkPropPredDataset
import torch
from torch_geometric.nn import Node2Vec
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
class Datasets: #Relevant metric for ddi and ppa taken from https://ogb.stanford.edu/docs/linkprop/
    CiteSeer = Dataset('CiteSeer', Metrics.MRR)
    PubMed = Dataset('PubMed', Metrics.MRR)
    Cora = Dataset('Cora', Metrics.MRR)
    DrugDrugInteraction = Dataset('ogbl-ddi', Metrics.HitsAt(20))
    ProteinProteinAssociation = Dataset('ogbl-ppa', Metrics.HitsAt(100))
    
class EmbeddingModel(Enum):
    Raw = None
    DeepWalk = 'deepwalk'
    Node2Vec = 'node2vec'

def calc_embedding_model(graph, embeddingmodel: EmbeddingModel, device): #https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ppa/node2vec.py
    if embeddingmodel == EmbeddingModel.Raw:
        return None
    # p,q values for Node2Vec were taken from https://arxiv.org/pdf/1607.00653.pdf p.7
    p, q = (1, 1) if embeddingmodel == EmbeddingModel.DeepWalk else (4, 1)
    n2v = Node2Vec(graph.edge_index, 128, 40, 20, 10, sparse=True, p=p, q=q).to(device)
    loader = n2v.loader(batch_size=256, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)

    num_epochs = 2
    n2v.train()
    for epoch in range(num_epochs):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
    return n2v.embedding.weight.data

def apply_embedding_model(graph, embedding, device): #https://github.com/snap-stanford/ogb/blob/955f22515dc0e6a8231c0118f3c8760aa26c45a6/examples/linkproppred/ppa/mlp.py#L154
    if embedding is None:
        return graph
    x = embedding.to(device)
    graph.x = x
    return graph

def load_dataset(path, dataset: Dataset, device, embeddingmodel: EmbeddingModel=EmbeddingModel.Raw):
    if dataset.name.startswith('ogbl-'): # https://ogb.stanford.edu/docs/linkprop/#pyg
        dataset = PygLinkPropPredDataset(name=dataset.name, transform=T.ToSparseTensor(remove_edge_index=False))
        graph = dataset[0].to(device)
        embedding = calc_embedding_model(graph, embeddingmodel, device)

        split_edge = dataset.get_edge_split()
        train_edge, valid_edge, test_edge = split_edge['train'], split_edge['valid'], split_edge['test']
        idx = torch.randperm(train_edge['edge'].size(0))
        idx = idx[:valid_edge['edge'].size(0)]
        split_edge['eval_train'] = {'edge': train_edge['edge'][idx]}

    else:
        dataset = Planetoid(path, name=dataset.name, transform=T.ToSparseTensor(remove_edge_index=False))
        graph = dataset[0].to(device)

        embedding = calc_embedding_model(graph, embeddingmodel, device)

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
        

    graph = apply_embedding_model(graph, embedding, device)

    return (graph, split_edge)

class __tmp_evaluator__(Evaluator):
    def __init__(self, metric) -> None:
        self.eval_metric = metric

def create_evaluator(dataset: Dataset):
    if dataset.name.startswith('ogbl-'):
        return Evaluator(dataset.name)
    else:
        return __tmp_evaluator__(dataset.metric)
        