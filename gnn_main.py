import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from datasets import load_dataset
from gnn_architectures import FocalLoss
from model import *


torch.set_warn_always(False)
warnings.filterwarnings("ignore", category=UserWarning, message="User provided device_type of 'cuda', but CUDA is not available. Disabling")
warnings.filterwarnings("ignore", category=UserWarning, message="indexing with dtype torch.uint8 is now deprecated")


graph = load_dataset('yelp')
model = get_model(graph=graph, gnn_type='GAT')


# class weights calculation to counter imbalance
class_counts = torch.bincount(graph.ndata['label'])
class_weights = 1.0 / class_counts  # Inverse frequency
class_weights = class_weights / class_weights.sum()
class_weights[1] = 0.5
alpha = torch.tensor(class_weights, dtype=torch.float32)

# model criterion - optimizer
criterion = FocalLoss(alpha=alpha, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)


# Training loop
epochs = 100
for epoch in range(1,epochs+1):
    # Train the model
    loss = train(model, graph, optimizer, criterion, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Evaluate on training and validation sets
    if epoch % 5 == 0:
        train_acc, train_f1, train_report = evaluate(graph.ndata['train_mask'], model=model, graph=graph)
        val_acc, val_f1, val_report = evaluate(graph.ndata['val_mask'], model=model, graph=graph)
        
        # Print metrics
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

# Results on test set
test_acc, test_f1, test_report = evaluate(graph.ndata['test_mask'], model=model, graph=graph)
print(f"Test F1: {test_f1:.4f}")
print("Test Report:\n", test_report)
