from datasets import load_dataset, Datasets
from torch_geometric.data import DataLoader

# Reproduce
dataset = load_dataset("./datasets/", Datasets.DrugDrugInteraction)

split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False)

# New Data


# Hyperparams Check