import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import argparse

from datasets import load_dataset, train_test_split_graph
from gnn_architectures import FocalLoss
from model import *
from top_k import analyze_tensor


torch.set_warn_always(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-k Anomaly Detection with GNNs")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["amazon", "yelp"],
                        help="Dataset to use: 'amazon' or 'yelp'")
    parser.add_argument("--model", type=str, required=True,
                        choices=["GCN", "GraphSAGE", "BWGNN"],
                        help="GNN architecture to train")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    gnn_type = args.model


    graph = load_dataset(
        dataset,
        enriched=False,
        load_path=f"enriched_graph_{dataset}.dgl"
    )


    model = get_model(graph=graph, h_feats=24, num_layers=2, gnn_type=gnn_type)

    # class weights calculation to counter imbalance
    class_counts = torch.bincount(graph.ndata['label'])
    class_weights = 1.0 / class_counts  # Inverse frequency
    class_weights = class_weights / class_weights.sum()
    # alpha = class_weights.clone().detach()

    # model criterion - optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 3], dtype=torch.float32))

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Split the graph into train, validation, and test sets
    train_data, val_data, test_data, = train_test_split_graph(graph, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3)

    # Training loop
    epochs = args.epochs

    for epoch in range(1,epochs+1):
        # Train the model
        loss = train(model, train_data, graph, optimizer, criterion)
        
        # Evaluate on training and validation sets
        if epoch % 10 == 0:
            train_acc, train_f1, _, _ = evaluate(mask=train_data, model=model, graph=graph)
            val_acc, val_f1, _, _ = evaluate(mask=val_data, model=model, graph=graph)
            
            # Print metrics
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    # Results on test set
    test_acc, test_f1, test_report, probs = evaluate(mask=test_data, model=model, graph=graph)
    print(f"Test F1: {test_f1:.4f}")
    print("Test Report:\n", test_report)


    accuracies = {}
    iteration_points = [0.05,0.1,0.2,0.5,1,2,5,10]

    for ratio in iteration_points:
        k = int(ratio*sum(test_data)/ 100)
        top_k_indices = analyze_tensor(probs, '', k)

        test_nodes = torch.nonzero(test_data, as_tuple=True)[0]
        test_subgraph = graph.subgraph(test_nodes)

        pred_zeros = 0
        for i in range(k):
            if test_subgraph.ndata['label'][top_k_indices[0][i]].item() == 0:
                pred_zeros += 1
            
        pred_ones = 0
        for i in range(k):
            if test_subgraph.ndata['label'][top_k_indices[1][i]].item() == 1:
                pred_ones += 1

        accuracies[ratio] = (pred_ones/k)

    def normalize_dict(dict, min_val, max_val):
        return {k: round(((v - min_val) / (max_val - min_val)), 2) for k, v in dict.items()}
    final_scores = normalize_dict(ones_dict, min_val=0, max_val=1)
    print(accuracies)
    
    with open(f'{dataset}_{gnn_type}.txt', 'a') as f:
        f.write(f"{accuracies}\n")
