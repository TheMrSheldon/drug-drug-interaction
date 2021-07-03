from typing import List
from torch_geometric import datasets
from datasets import Dataset, Datasets, EmbeddingModel, load_dataset, create_evaluator
import gnn
import torch.nn
import torch.optim
from tqdm import tqdm

def init_sage(hidden_channels, num_datanodes, num_neighbors, dropout, device: torch.device):
    #TODO: adjust the sample sizes according to num_neighbors
    num_layers = len(num_neighbors)
    model = gnn.SAGE(hidden_channels, hidden_channels, hidden_channels, num_layers, dropout).to(device)
    embedding = torch.nn.Embedding(num_datanodes, hidden_channels).to(device)
    predictor = gnn.LinkPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout).to(device)
    return (model, embedding, predictor)

def train(model, embedding, predictor, evaluator, num_epochs, learn_rate, graph, split_edge, batch_size):
    # Early stopping memory
    best_model_params = None
    best_model_score = None
    evals_since_best = 0
    # Reset parameters
    torch.nn.init.xavier_uniform_(embedding.weight)
    model.reset_parameters()
    predictor.reset_parameters()
    # Init the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()) + list(predictor.parameters()), lr=learn_rate)
    # Train for multiple epochs
    for epoch in tqdm(range(num_epochs), leave=False):
        loss = gnn.train(model, predictor, embedding.weight, graph.adj_t, split_edge, optimizer, batch_size)
        if (epoch+1) % 1 == 0:
            (val_score, test_score) = gnn.test(model, predictor, embedding.weight, graph.adj_t, split_edge, evaluator, batch_size)[0]
            if best_model_params is None or val_score >= best_model_score:
                evals_since_best = 0
                best_model_score = val_score
                best_model_params = model.state_dict().copy()
            else:
                evals_since_best += 1
                if evals_since_best > 3:
                    print(f"\rEarly stopping triggered after {epoch} epochs")
                    # restore
                    model.load_state_dict(best_model_params, strict=True)
                    break

def init_train_eval(embedding_model: EmbeddingModel, dataset: Dataset, num_epochs, num_neighbors, batch_size, device: torch.device):
    print(f'==Training dataset: {dataset.name}==')
    print(f' Loading Dataset', end='\r')
    (graph, split_edge) = load_dataset("./datasets/", dataset, device, embedding_model)
    print(f' Dataset loaded', end='\r')
    (model, embedding, predictor) = init_sage(hidden_channels=20, #graph.num_node_features
        num_datanodes=graph.num_nodes, num_neighbors=num_neighbors, dropout=0.5, device=device)
    evaluator = create_evaluator(dataset)
    train(model, embedding, predictor, evaluator, num_epochs=num_epochs, learn_rate=0.005, graph=graph, split_edge=split_edge, batch_size=batch_size)
    print(gnn.test(model, predictor, embedding.weight, graph.adj_t, split_edge, evaluator, batch_size))

def run(datasets: List[Dataset], num_epochs, num_neighbors, device: torch.device, embedding_models: List[EmbeddingModel]=[EmbeddingModel.Raw]):
    batch_size = 64*1024
    for model in embedding_models:
        for dataset in datasets:
            init_train_eval(model, dataset, num_epochs, num_neighbors, batch_size, device)
            torch.cuda.empty_cache()  # clear memory from old dataset

if __name__ == '__main__':
    # Check if training on GPU is possible
    print(f'Cuda available: {torch.cuda.is_available()}')
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training device: {device}')

    ######################################################################################################################
    # TASKS                                                                                                              #
    ######################################################################################################################
    Datasets1 = [Datasets.CiteSeer, Datasets.PubMed, Datasets.Cora]
    Datasets2 = [Datasets.DrugDrugInteraction, Datasets.ProteinProteinAssociation]
    ### Reproduce
    def task_reproduce():
        run(Datasets1, 10, [25, 15], device)

    ### New Data
    def task_new_data():
        run(Datasets2, 10, [25, 15], device)

    ### Hyperparams Check
    def task_hyperparams_check():
        #### Different Number of Neighbors
        run(Datasets1, 10, [30, 15], device)
        run(Datasets1, 10, [25, 25], device)

        #### Different Number of Epochs
        run(Datasets1, 1, [25, 15], device)
        run(Datasets1, 5, [25, 15], device)
        run(Datasets1, 100, [25, 15], device)
        run(Datasets1, 300, [25, 15], device)

        #### Different Depth
        run(Datasets1, 10, [25, 15, 15], device)
        run(Datasets1, 10, [25, 15, 30], device)

    ### New Algorithm Variant
    def task_algorithm_variant():
        pass
        #TODO

    ### Ablation Study
    def task_ablation_study():
        run(Datasets1+Datasets2, 10, [25, 15], device, [EmbeddingModel.DeepWalk, EmbeddingModel.Node2Vec])

    #task_reproduce()
    #task_new_data()
    #task_hyperparams_check()
    #task_algorithm_variant()
    task_ablation_study()