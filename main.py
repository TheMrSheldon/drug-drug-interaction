from datasets import load_dataset, Datasets
import gnn
import torch.nn
import torch.optim
from ogb.linkproppred import Evaluator

# Reproduce



# New Data
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = load_dataset("./datasets/", Datasets.DrugDrugInteraction)
split_edge = dataset.get_edge_split()
data = dataset[0]
adj_t = data.adj_t.to(device)
idx = torch.randperm(split_edge['train']['edge'].size(0))
idx = idx[:split_edge['valid']['edge'].size(0)]
split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

hidden_channels = 256
num_layers = 2
dropout = 0.5
learn_rate = 0.005
batch_size = 64 * 1024
model = gnn.SAGE(hidden_channels, hidden_channels, hidden_channels, num_layers, dropout).to(device)
emb = torch.nn.Embedding(data.num_nodes, hidden_channels).to(device)
predictor = gnn.LinkPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout).to(device)
evaluator = Evaluator(Datasets.DrugDrugInteraction)

torch.nn.init.xavier_uniform_(emb.weight)
model.reset_parameters()
predictor.reset_parameters()
optimizer = torch.optim.Adam(list(model.parameters()) + list(emb.parameters()) + list(predictor.parameters()), lr=learn_rate)
for epoch in range(200):
    loss = gnn.train(model, predictor, emb.weight, adj_t, split_edge, optimizer, batch_size)
    print(loss)

# Hyperparams Check