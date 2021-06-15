from torch.nn.functional import embedding
from datasets import load_dataset, Datasets
import gnn
import torch.nn
import torch.optim
from ogb.linkproppred import Evaluator, evaluate

class Metrics:
    MRR='mrr'
    def HitsAt(num):
        return f'Hits@{num}'

def init_sage(hidden_channels, num_datanodes, num_layers, dropout, device):
    model = gnn.SAGE(hidden_channels, hidden_channels, hidden_channels, num_layers, dropout).to(device)
    embedding = torch.nn.Embedding(num_datanodes, hidden_channels).to(device)
    predictor = gnn.LinkPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout).to(device)
    evaluator = Evaluator(Datasets.DrugDrugInteraction)#TODO: maybe init the evaluator to the specified metric here (would also need to add the metric to the function parameters)
    return (model, embedding, predictor, evaluator)

def train(model, embedding, predictor, metric, num_epochs, learn_rate, graph, split_edge, batch_size):
    #TODO: use metric (maybe not needed here?)

    # Reset parameters
    torch.nn.init.xavier_uniform_(embedding.weight)
    model.reset_parameters()
    predictor.reset_parameters()
    # Init the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()) + list(predictor.parameters()), lr=learn_rate)
    # Train for multiple epochs
    for epoch in range(num_epochs):
        loss = gnn.train(model, predictor, embedding.weight, graph.adj_t, split_edge, optimizer, batch_size)
        print(loss)

def init_train_eval_sage(dataset, metric, num_epochs, batch_size, device):
    #TODO: use metric (maybe part of evaluator)

    (graph, split_edge) = load_dataset("./datasets/", dataset, device)

    (model, embedding, predictor, evaluator) = init_sage(hidden_channels=256, num_datanodes=graph.num_nodes, num_layers=2, dropout=0.5, device=device)
    train(model, embedding, predictor, metric, num_epochs=num_epochs, learn_rate=0.005, graph=graph, split_edge=split_edge, batch_size=batch_size)
    # TODO: evaluate the trained model

# Check if training on GPU is possible
print(f'Cuda available: {torch.cuda.is_available()}')
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

######################################################################################################################
# TASKS                                                                                                              #
######################################################################################################################
#TODO: Use the metrics. Please note that the ogb.linkproppred.Evaluator can perform MRR and Hits@K. I just don't know yet how and if one may switch between those or if multiple evaluators are needed
### Reproduce
init_train_eval_sage(Datasets.CiteSeer, Metrics.MRR, 200, 64*1024, device)
init_train_eval_sage(Datasets.PubMed, Metrics.MRR, 200, 64*1024, device)
init_train_eval_sage(Datasets.Cora, Metrics.MRR, 200, 64*1024, device)

### New Data
init_train_eval_sage(Datasets.DrugDrugInteraction, Metrics.HitsAt(20), 200, 64*1024, device)
init_train_eval_sage(Datasets.ProteinProteinAssociation, Metrics.HitsAt(20), 200, 64*1024, device)

### Hyperparams Check