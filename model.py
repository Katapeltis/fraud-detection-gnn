import torch
from sklearn.metrics import f1_score, classification_report
from datasets import train_test_split_graph
from gnn_architectures import HeteroGNN, HeteroGAT



def get_model(graph, gnn_type: str):
    num_classes = 2
    etypes = graph.etypes

    if gnn_type == 'GCN':
        in_feats = graph.ndata['feature'].shape[1]
        h_feats = 16  # Hidden layer dimension
        model = HeteroGNN(in_feats, h_feats, num_classes, etypes)

    if gnn_type == 'GAT':
        in_feats = graph.ndata['feature'].shape[1]  # Input feature dimension
        h_feats = 8  # Hidden layer dimension
        num_heads = 4  # Number of attention heads
        model = HeteroGAT(in_feats, h_feats, num_classes, etypes, num_heads)

    return model.to('cpu')


def train(model, graph, optimizer, criterion, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    optimizer.zero_grad()

    inputs = {'review': graph.ndata['feature']}
    train_data, _,_ = train_test_split_graph(graph, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)

    graph = graph.to(device)
    graph.ndata['label'] = graph.ndata['label'].to(device)
    
    logits = model(graph, inputs)
    logits = logits['review'][train_data.to(device)]
    labels = graph.ndata['label'][train_data.to(device)]
    
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
        logits = model(graph, inputs)
        logits = logits['review'][mask]
        labels = graph.ndata['label'][mask]
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        
        # Compute F1 score and classification report
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')  # macro - micro - binary
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy(), target_names=['Class 0', 'Class 1'])
        return accuracy, f1, report
