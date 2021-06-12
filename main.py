from torch.nn.functional import embedding
from datasets import load_dataset, Datasets
import gnn
import torch.nn
import torch.optim
from ogb.linkproppred import Evaluator, evaluate

def init_sage(hidden_channels, num_datanodes, num_layers, dropout, device):
    model = gnn.SAGE(hidden_channels, hidden_channels, hidden_channels, num_layers, dropout).to(device)
    embedding = torch.nn.Embedding(num_datanodes, hidden_channels).to(device)
    predictor = gnn.LinkPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout).to(device)
    evaluator = Evaluator(Datasets.DrugDrugInteraction)
    return (model, embedding, predictor, evaluator)

def train(model, embedding, predictor, num_epochs, learn_rate, adj_t, split_edge, batch_size):
    # Reset parameters
    torch.nn.init.xavier_uniform_(embedding.weight)
    model.reset_parameters()
    predictor.reset_parameters()
    # Init the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()) + list(predictor.parameters()), lr=learn_rate)
    # Train for multiple epochs
    for epoch in range(num_epochs):
        loss = gnn.train(model, predictor, embedding.weight, adj_t, split_edge, optimizer, batch_size)
        print(loss)

def init_train_eval_sage(dataset, num_epochs, batch_size, device):
    (num_datanodes, adj_t, split_edge) = load_dataset("./datasets/", dataset, device)

    (model, embedding, predictor, evaluator) = init_sage(hidden_channels=256, num_datanodes=num_datanodes, num_layers=2, dropout=0.5, device=device)
    train(model, embedding, predictor, num_epochs=num_epochs, learn_rate=0.005, adj_t=adj_t, split_edge=split_edge, batch_size=batch_size)
    # TODO: evaluate the trained model

# Check if training on GPU is possible
print(f'Cuda available: {torch.cuda.is_available()}')
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

######################################################################################################################
# TASKS                                                                                                              #
######################################################################################################################
### Reproduce
# Does not yet work
#init_train_eval_sage(Datasets.CiteSeer, 200, 64*1024, device)
#init_train_eval_sage(Datasets.PubMed, 200, 64*1024, device)
#init_train_eval_sage(Datasets.Cora, 200, 64*1024, device)

### New Data
init_train_eval_sage(Datasets.DrugDrugInteraction, 200, 64*1024, device)
# Does not yet work
init_train_eval_sage(Datasets.ProteinProteinAssociation, 200, 64*1024, device)

### Hyperparams Check