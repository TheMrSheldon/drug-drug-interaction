from torch.nn.functional import embedding
from datasets import Datasets, Metrics, load_dataset, create_evaluator
import gnn
import torch.nn
import torch.optim

def init_sage(hidden_channels, num_datanodes, num_layers, dropout, device):
    model = gnn.SAGE(hidden_channels, hidden_channels, hidden_channels, num_layers, dropout).to(device)
    embedding = torch.nn.Embedding(num_datanodes, hidden_channels).to(device)
    predictor = gnn.LinkPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout).to(device)
    return (model, embedding, predictor)

def train(model, embedding, predictor, evaluator, num_epochs, learn_rate, graph, split_edge, batch_size):
    # Reset parameters
    torch.nn.init.xavier_uniform_(embedding.weight)
    model.reset_parameters()
    predictor.reset_parameters()
    # Init the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()) + list(predictor.parameters()), lr=learn_rate)
    # Train for multiple epochs
    for epoch in range(num_epochs):
        loss = gnn.train(model, predictor, embedding.weight, graph.adj_t, split_edge, optimizer, batch_size)
        if (epoch+1) % 10 == 0:
            results = gnn.test(model, predictor, embedding.weight, graph.adj_t, split_edge, evaluator, batch_size)
            print(results)

def init_train_eval_sage(dataset, metric, num_epochs, batch_size, device):
    print(f'==Training dataset: {dataset}==')
    (graph, split_edge) = load_dataset("./datasets/", dataset, device)
    (model, embedding, predictor) = init_sage(hidden_channels=256, num_datanodes=graph.num_nodes, num_layers=2, dropout=0.5, device=device)
    evaluator = evaluator = create_evaluator(dataset, metric)#TODO: maybe init the evaluator to the specified metric here (would also need to add the metric to the function parameters)
    train(model, embedding, predictor, evaluator, num_epochs=num_epochs, learn_rate=0.005, graph=graph, split_edge=split_edge, batch_size=batch_size)
    # TODO: evaluate the trained model

# Check if training on GPU is possible
print(f'Cuda available: {torch.cuda.is_available()}')
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Training device: {device}')

######################################################################################################################
# TASKS                                                                                                              #
######################################################################################################################
init_train_eval_sage(Datasets.CiteSeer, Metrics.MRR, 200, 64*1024, device)
init_train_eval_sage(Datasets.PubMed, Metrics.MRR, 200, 64*1024, device)
init_train_eval_sage(Datasets.Cora, Metrics.MRR, 200, 64*1024, device)

### New Data
init_train_eval_sage(Datasets.DrugDrugInteraction, Metrics.HitsAt(20), 200, 64*1024, device)
init_train_eval_sage(Datasets.ProteinProteinAssociation, Metrics.HitsAt(20), 200, 64*1024, device)

### Hyperparams Check