import torch
from sklearn.metrics import f1_score, classification_report
from gnn_architectures import HeteroGNN, HeteroGAT, HeteroGraphSAGE


def get_model(graph, h_feats: int, num_layers: int, gnn_type: str):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Model device: {device}")

    num_classes = 2
    in_feats = graph.ndata['feature'].shape[1]  # Input feature dimension
    etypes = graph.etypes
    #num_layers = 4  # Number of layers in the GNN

    if gnn_type == 'GCN':
        #h_feats = 16  # Hidden layer dimension
        model = HeteroGNN(in_feats, h_feats, num_classes, etypes, num_layers)

    if gnn_type == 'GAT':
        #h_feats = 8  # Hidden layer dimension
        num_heads = 4  # Number of attention heads
        model = HeteroGAT(in_feats, h_feats, num_classes, etypes, num_layers, num_heads)

    if gnn_type == 'GraphSAGE':
        #h_feats = 16
        aggregator_type = 'mean'
        model = HeteroGraphSAGE(in_feats, h_feats, num_classes, etypes, num_layers, aggregator_type)

    return model


def train(model, mask, graph, optimizer, criterion):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Training on device: {device}")

    model.train()
    optimizer.zero_grad()

    # Create subgraph containing only training nodes
    train_nodes = torch.nonzero(mask, as_tuple=True)[0]
    train_subgraph = graph.subgraph(train_nodes)

    # Move data to device
    # train_subgraph = train_subgraph.to(device)


    # train_subgraph.ndata['feature'] = train_subgraph.ndata['feature'].to(device)
    # train_subgraph.ndata['label'] = train_subgraph.ndata['label'].to(device)

    # Forward pass on training subgraph only
    inputs = {train_subgraph.ntypes[0]: train_subgraph.ndata['feature']}
    logits, _ = model(train_subgraph, inputs)
    
    # Calculate loss
    logits = logits[train_subgraph.ntypes[0]]
    labels = train_subgraph.ndata['label']
    loss = criterion(logits, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(mask, model, graph):

    model.eval()
    with torch.no_grad():
        # Create subgraph containing only training nodes
        test_nodes = torch.nonzero(mask, as_tuple=True)[0]
        test_subgraph = graph.subgraph(test_nodes)

        # Move data to device
        # test_subgraph = test_subgraph.to(device)
        # test_subgraph.ndata['feature'] = test_subgraph.ndata['feature'].to(device)
        # test_subgraph.ndata['label'] = test_subgraph.ndata['label'].to(device) 
        
        inputs = {test_subgraph.ntypes[0]: test_subgraph.ndata['feature']}
        logits,probs = model(test_subgraph, inputs)
        logits = logits[test_subgraph.ntypes[0]]
        labels = test_subgraph.ndata['label']
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        
        # Compute F1 score and classification report
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy(), target_names=['Class 0', 'Class 1'])
        return accuracy, f1, report, probs