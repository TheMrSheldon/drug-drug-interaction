from __future__ import nested_scopes
from torch.nn.functional import embedding
from datasets import Datasets, Metrics, load_dataset, create_evaluator
import gnn
import torch.nn
import torch.optim
from tqdm import tqdm

class EmbeddingModels:
    GraphSAGE = 'sage'
    DeepWalk = 'deepwalk'
    Node2Vec = 'node2vec'

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
    for epoch in tqdm(range(num_epochs), leave=False):
        loss = gnn.train(model, predictor, embedding.weight, graph.adj_t, split_edge, optimizer, batch_size)
        if (epoch+1) % 10 == 0:
            results = gnn.test(model, predictor, embedding.weight, graph.adj_t, split_edge, evaluator, batch_size)
            #TODO: early stopping #print(results)

def init_train_eval(embedding_model, dataset, metric, num_epochs, batch_size, device):
    print(f'==Training dataset: {dataset}==')
    (graph, split_edge) = load_dataset("./datasets/", dataset, device)
    (model, embedding, predictor) = init_sage(hidden_channels=20, num_datanodes=graph.num_nodes, num_layers=2, dropout=0.5, device=device)
    evaluator = evaluator = create_evaluator(dataset, metric)
    train(model, embedding, predictor, evaluator, num_epochs=num_epochs, learn_rate=0.005, graph=graph, split_edge=split_edge, batch_size=batch_size)
    print(gnn.test(model, predictor, embedding.weight, graph.adj_t, split_edge, evaluator, batch_size))

def run(embedding_models, datasets_and_metrics, num_epochs, device):
    batch_size = 64*1024
    for model in embedding_models:
        for dataset, metrics in datasets_and_metrics:
            init_train_eval(model, dataset, metrics, num_epochs, batch_size, device)

if __name__ == '__main__':
    # Check if training on GPU is possible
    print(f'Cuda available: {torch.cuda.is_available()}')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training device: {device}')

    ######################################################################################################################
    # TASKS                                                                                                              #
    ######################################################################################################################
    Datasets1 = [(Datasets.CiteSeer, Metrics.MRR), (Datasets.PubMed, Metrics.MRR), (Datasets.Cora, Metrics.MRR)]
    Datasets2 = [(Datasets.ProteinProteinAssociation, Metrics.HitsAt(20))]#[(Datasets.DrugDrugInteraction, Metrics.HitsAt(20)), (Datasets.ProteinProteinAssociation, Metrics.HitsAt(20))]
    ### Reproduce
    def task_reproduce():
        run([EmbeddingModels.GraphSAGE], Datasets1, 200, device)

    ### New Data
    def task_new_data():
        run([EmbeddingModels.GraphSAGE], Datasets2, 200, device)

    ### Hyperparams Check
    def task_hyperparams_check():
        #### Different Number of Neighbors
        #TODO
        #### Different Number of Epochs
        run([EmbeddingModels.GraphSAGE], Datasets1, 100, device)
        run([EmbeddingModels.GraphSAGE], Datasets1, 300, device)

        #### Different Depth
        #TODO

        ### New Algorithm Variant
        #TODO

    ### Ablation Study
    def task_ablation_study():
        run([EmbeddingModels.DeepWalk, EmbeddingModels.Node2Vec], Datasets1+Datasets2, 200, device)

    task_new_data()