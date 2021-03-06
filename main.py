from typing import List
from torch_geometric import datasets
from datasets import Dataset, Datasets, EmbeddingModel, load_dataset, create_evaluator
import gnn
import torch.nn
import torch.optim
from tqdm import tqdm

def init_sage(hidden_channels, num_datanodes, num_layers, dropout, device: torch.device):
    model = gnn.SAGE(hidden_channels, hidden_channels, hidden_channels, num_layers, dropout).to(device)
    embedding = torch.nn.Embedding(num_datanodes, hidden_channels).to(device)
    predictor = gnn.LinkPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout).to(device)
    return (model, embedding, predictor)

def train(model, embedding, predictor, evaluator, num_epochs, learn_rate, graph, split_edge, batch_size, embedding_model):
    # Early stopping memory
    best_model_params = None
    best_model_score = None
    evals_since_best = 0
    # Reset parameters
    torch.nn.init.xavier_uniform_(embedding.weight)
    model.reset_parameters()
    predictor.reset_parameters()
    # Stuff
    x_input = embedding.weight if embedding_model == EmbeddingModel.Raw else graph.x
    embedding_parameters = list(embedding.parameters()) if embedding_model == EmbeddingModel.Raw else list()
    # Init the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + embedding_parameters + list(predictor.parameters()), lr=learn_rate)
    # Train for multiple epochs
    for epoch in tqdm(range(num_epochs), leave=False):
        loss = gnn.train(model, predictor, x_input, graph.adj_t, split_edge, optimizer, batch_size)
        if (epoch+1) % 1 == 0:
            (val_score, test_score) = gnn.test(model, predictor, x_input, graph.adj_t, split_edge, evaluator, batch_size)[0]
            if best_model_params is None or val_score >= best_model_score:
                evals_since_best = 0
                best_model_score = val_score
                best_model_params = model.state_dict().copy()
            else:
                evals_since_best += 1
                if evals_since_best > 10 and num_epochs >= 400:
                    print(f"\rEarly stopping triggered after {epoch} epochs", flush=True)
                    break
    # restore best model
    model.load_state_dict(best_model_params, strict=True)

def init_train_eval(embedding_model: EmbeddingModel, dataset: Dataset, num_epochs, num_layers, batch_size, device: torch.device, drop_probability):
    print(f'==Training dataset: {dataset.name}==', flush=True)
    print(f' Loading Dataset', end='\r', flush=True)
    (graph, split_edge) = load_dataset("./datasets/", dataset, device, drop_probability, embedding_model)
    print(f' Dataset loaded', end='\r', flush=True)
    (model, embedding, predictor) = init_sage(hidden_channels=256 if embedding_model == EmbeddingModel.Raw else graph.x.size(-1),
        num_datanodes=graph.num_nodes, num_layers=num_layers, dropout=0.5, device=device)
    evaluator = create_evaluator(dataset)
    train(model, embedding, predictor, evaluator, num_epochs=num_epochs, learn_rate=0.005, graph=graph, split_edge=split_edge, batch_size=batch_size, embedding_model=embedding_model)
    print(gnn.test(model, predictor, embedding.weight if embedding_model == EmbeddingModel.Raw else graph.x, graph.adj_t, split_edge, evaluator, batch_size), flush=True)

def run(datasets: List[Dataset], num_epochs, num_layers, device: torch.device, drop_probability=0, embedding_models: List[EmbeddingModel]=[EmbeddingModel.Raw]):
    batch_size = 64*1024
    for model in embedding_models:
        for dataset in datasets:
            init_train_eval(model, dataset, num_epochs, num_layers, batch_size, device, drop_probability)
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
        print("\n\nTask: Reproduce", flush=True)
        run(Datasets1, 10, 2, device)

    ### New Data
    def task_new_data():
        print("\n\nTask: New Data", flush=True)
        run(Datasets2, 10, 2, device)

    ### Hyperparams Check
    def task_hyperparams_check():
        print("\n\nTask: Hyperparams", flush=True)
        #### Different Number of Epochs
        # Run all datasets except PPA with 50, 100, 200, and 300 epochs
        for num_epochs in [50, 100, 200, 300]:
            print(f' Run with {num_epochs} epochs', flush=True)
            run(Datasets1+[Datasets.DrugDrugInteraction], num_epochs, 2, device)
        # Run all datasets with early stopping and a maximum of 400 epochs
        print(f' Run with 400 epochs (early stopping)', flush=True)
        run(Datasets1+Datasets2, 400, 2, device)

        #### Different Depth
        # Run all datasets except PPA with a depth of 1, 3, and 5 layers
        for depth in [1, 3, 5, 10]:
            print(f' Run with depth: {depth}', flush=True)
            run(Datasets1+[Datasets.DrugDrugInteraction], 10, depth, device)

    ### New Algorithm Variant
    def task_algorithm_variant():
        print("\n\nTask: Algorithm Variant", flush=True)
        run(Datasets1+[Datasets.DrugDrugInteraction], 10, 2, device, drop_probability=.1)
        run(Datasets1+[Datasets.DrugDrugInteraction], 10, 2, device, drop_probability=.5)

    ### Ablation Study
    def task_ablation_study():
        print("\n\nTask: Ablation Study", flush=True)
        run(Datasets1+Datasets2, 10, 2, device, embedding_models=[EmbeddingModel.DeepWalk, EmbeddingModel.Node2Vec])
        print(" Out of our own interest: Ablation Study with early stopping", flush=True)
        run(Datasets1+[Datasets.DrugDrugInteraction], 400, 2, device, embedding_models=[EmbeddingModel.DeepWalk, EmbeddingModel.Node2Vec])

    #task_reproduce()
    #task_new_data() #Nur Alex
    #task_hyperparams_check()
    task_algorithm_variant()
    #task_ablation_study()