import torch
from sklearn.metrics import f1_score, classification_report
from datasets import train_test_split_graph
from gnn_architectures import HeteroGNN, HeteroGAT, HeteroGraphSAGE
import torch.nn.functional as F
import torch.nn as nn


def get_model(graph, gnn_type: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 2
    in_feats = graph.ndata['feature'].shape[1]  # Input feature dimension
    etypes = graph.etypes

    if gnn_type == 'GCN':
        h_feats = 16  # Hidden layer dimension
        model = HeteroGNN(in_feats, h_feats, num_classes, etypes)

    if gnn_type == 'GAT':
        h_feats = 8  # Hidden layer dimension
        num_heads = 4  # Number of attention heads
        model = HeteroGAT(in_feats, h_feats, num_classes, etypes, num_heads)

    if gnn_type == 'GraphSAGE':
        h_feats = 16
        aggregator_type = 'mean'
        model = HeteroGraphSAGE(in_feats, h_feats, num_classes, etypes, aggregator_type)

    return model.to(device)



def train(model, mask, graph, optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    optimizer.zero_grad()

    inputs = {'review': graph.ndata['feature']}

    graph = graph.to(device)
    graph.ndata['label'] = graph.ndata['label'].to(device)
    
    logits,_ = model(graph, inputs)
    logits = logits['review'][mask]
    labels = graph.ndata['label'][mask]
    
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask, model, graph):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        graph.ndata['feature'] = graph.ndata['feature'].to(device)
        inputs = {'review': graph.ndata['feature']}
        logits,probs = model(graph, inputs)
        logits = logits['review'][mask]
        labels = graph.ndata['label'][mask]
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        
        # Compute F1 score and classification report
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy(), target_names=['Class 0', 'Class 1'])
        return accuracy, f1, report, probs
