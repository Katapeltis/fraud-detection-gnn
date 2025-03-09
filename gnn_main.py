import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from gnn_architectures import HeteroGNN, FocalLoss
from model import train, evaluate

from BWGNN import *


torch.set_warn_always(True)

yelp_graph = load_dataset('yelp')

in_feats = yelp_graph.ndata['feature'].shape[1]
h_feats = 16  # Hidden layer dimension
num_classes = 2  
etypes = yelp_graph.etypes
model = HeteroGNN(in_feats, h_feats, num_classes, etypes)


# Compute class weights
class_counts = torch.bincount(yelp_graph.ndata['label'])
class_weights = 1.0 / class_counts  # Inverse frequency
class_weights = class_weights / class_weights.sum()
class_weights[1] = 0.5
alpha = torch.tensor(class_weights, dtype=torch.float32)

# model criterion - optimizer
criterion = FocalLoss(alpha=alpha, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training loop
epochs = 100
for epoch in range(epochs):
    # Train the model
    loss = train(model, yelp_graph, optimizer, criterion)
    
    # Evaluate on training and validation sets
    if epoch % 10 == 0:
        train_acc, train_f1, train_report = evaluate(yelp_graph.ndata['train_mask'], model=model, graph=yelp_graph)
        val_acc, val_f1, val_report = evaluate(yelp_graph.ndata['val_mask'], model=model, graph=yelp_graph)
        
        # Print metrics
        print(f"Epoch {epoch}, Loss: {loss:.4f}, "
        #       f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
        #       f"Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}"
              )
        # print("Training Report:\n", train_report)
        # print("Validation Report:\n", val_report)

# Test the model
test_acc, test_f1, test_report = evaluate(yelp_graph.ndata['test_mask'], model=model, graph=yelp_graph)
print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
print("Test Report:\n", test_report)