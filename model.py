
import torch
from sklearn.metrics import f1_score, classification_report


def train(model, graph, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    inputs = {'review': graph.ndata['feature']}
    
    logits = model(graph, inputs)
    logits = logits['review'][graph.ndata['train_mask']]
    labels = graph.ndata['label'][graph.ndata['train_mask']]
    
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask, model, graph):
    model.eval()
    with torch.no_grad():
        inputs = {'review': graph.ndata['feature']}
        logits = model(graph, inputs)
        logits = logits['review'][mask]
        labels = graph.ndata['label'][mask]
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        
        # Compute F1 score and classification report
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='binary')  # Use 'macro' for multi-class
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy(), target_names=['Class 0', 'Class 1'])
        return accuracy, f1, report