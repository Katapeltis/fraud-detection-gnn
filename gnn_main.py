import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from datasets import load_dataset, train_test_split_graph
from gnn_architectures import FocalLoss
from model import *
#from top_k import get_top_k_positions


torch.set_warn_always(False)
warnings.filterwarnings("ignore", category=UserWarning, message="User provided device_type of 'cuda', but CUDA is not available. Disabling")
warnings.filterwarnings("ignore", category=UserWarning, message="indexing with dtype torch.uint8 is now deprecated")


graph = load_dataset('yelp')
model = get_model(graph=graph, gnn_type='GCN')


# class weights calculation to counter imbalance
class_counts = torch.bincount(graph.ndata['label'])
class_weights = 1.0 / class_counts  # Inverse frequency
class_weights = class_weights / class_weights.sum()


alpha = torch.tensor(class_weights, dtype=torch.float32)

# model criterion - optimizer
criterion = FocalLoss(alpha=alpha, gamma=1.5)
#criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Split the graph into train, validation, and test sets
train_data, val_data, test_data, = train_test_split_graph(graph)

# Training loop
epochs = 10

for epoch in range(1,epochs+1):
    # Train the model
    loss = train(model, train_data, graph, optimizer, criterion)
    
    # Evaluate on training and validation sets
    if epoch % 1 == 0:
        train_acc, train_f1, _, _ = evaluate(mask=train_data, model=model, graph=graph)
        val_acc, val_f1, _, _ = evaluate(mask=val_data, model=model, graph=graph)
        
        # Print metrics
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

# Results on test set
test_acc, test_f1, test_report, probs = evaluate(mask=test_data, model=model, graph=graph)
print(f"Test F1: {test_f1:.4f}")
print("Test Report:\n", test_report)

print(probs)

# top_k = get_top_k_positions(probs, 'review', 3)
# print("Top K Positions:", top_k)
# a,_ = top_k[0]
# b,_ = top_k[1]
# c,_ = top_k[2]

# print(probs['review'][a][0], probs['review'][a][1])
# print(probs['review'][b][0], probs['review'][b][1])
# print(probs['review'][c][0], probs['review'][c][1])
# print(train_data[a])
# print(train_data[b])
# print(train_data[c])
# print(test_data[a])
# print(test_data[b])
# print(test_data[c])


# # Initialize model (output size = 1 for binary)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# # Since there's only one node type
# ntype = graph.ntypes[0]
# print(graph.ntypes)
# print(ntype)

# # Extract features and labels
# features = {ntype: graph.ndata['feature']}
# labels = graph.ndata['label']
