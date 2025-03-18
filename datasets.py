from dgl.data import AmazonCoBuy, FraudYelpDataset
from sklearn.model_selection import train_test_split
import torch


raw_dir='/Users/marios/fraud_dgl'

def load_dataset(dataset):
    if dataset == 'yelp':
        yelp_dataset = FraudYelpDataset(raw_dir=raw_dir)
        graph = yelp_dataset[0]
    if dataset == 'amazon':
        amazon_dataset = AmazonCoBuy("computers")
        graph = amazon_dataset[0]
    return graph

def train_test_split_graph(yelp_graph, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, device='cpu'):

    # Get the number of nodes
    yelp_graph = yelp_graph.to(device)
    num_nodes = yelp_graph.number_of_nodes()

    # Get the labels
    labels = yelp_graph.ndata['label'].to(device)

    # Perform stratified splitting
    train_indices, temp_indices = train_test_split(
        torch.arange(num_nodes), train_size=train_ratio, stratify=labels, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_ratio / (val_ratio + test_ratio), stratify=labels[temp_indices], random_state=42
    )

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Add masks to the graph
    yelp_graph.ndata['train_mask'] = train_mask
    yelp_graph.ndata['val_mask'] = val_mask
    yelp_graph.ndata['test_mask'] = test_mask

    return train_mask, val_mask, test_mask
