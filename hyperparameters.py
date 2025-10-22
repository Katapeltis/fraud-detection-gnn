import itertools
import torch
import warnings
import tqdm
import numpy as np

from datasets import load_dataset, train_test_split_graph
from gnn_architectures import FocalLoss
from model import get_model, train, evaluate

# Suppress warnings
torch.set_warn_always(False)
warnings.filterwarnings("ignore", category=UserWarning, message=".*")

def run_experiment(dataset_name: str,
                   gnn_type: str,
                   epochs: int,
                   lr: float,
                   hidden_size: int,
                   gamma: float,
                   num_layers: int) -> float:
    """
    Train a GNN model obtained via get_model with FocalLoss(gamma), returning test F1.
    """
    graph = load_dataset(dataset_name)
    train_mask, _, test_mask = train_test_split_graph(graph, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3)

    # Instantiate model via factory
    model = get_model(graph, h_feats=hidden_size, num_layers=num_layers, gnn_type=gnn_type)

    # class weights calculation to counter imbalance
    class_counts = torch.bincount(graph.ndata['label'])
    class_weights = 1.0 / class_counts  # Inverse frequency
    class_weights = class_weights / class_weights.sum()

    # model criterion - optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        _ = train(model, train_mask, graph, optimizer, criterion)

    # Final evaluation on test set
    _, test_f1, _, _ = evaluate(mask=test_mask, model=model, graph=graph)
    return test_f1

if __name__ == '__main__':
    # Hyperparameter options
    datasets = ['yelp','amazon']
    gnn_types = ['GraphSAGE','GCN']
    learning_rates = [0.015]
    #learning_rates = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    hidden_sizes = [24]
    #hidden_sizes = [72,96,128]
    gamma_values = [1.5,2,2.5]
    num_layers = [2,3,4]
    epochs = 100

    # Prepare full experiment grid
    experiments = list(itertools.product(
        datasets,
        gnn_types,
        learning_rates,
        hidden_sizes,
        gamma_values,
        num_layers
    ))

    # Initialize summary for each dataset-gnn combination
    summary = {
        (ds, gt): {'best_score': 0.0, 'best_params': None}
        for ds in datasets for gt in gnn_types
    }
    all_experiments = {}

    # Single progress bar over all experiments
    pbar = tqdm.tqdm(total=len(experiments), desc="Total hyperparameter tuning")
    for dataset, gnn_type, lr, hs, gamma, nl in experiments:
        print(f"\nRunning experiment: {dataset}, {gnn_type}, lr={lr}, hs={hs}, gamma={gamma}, nl={nl}")
        f1 = run_experiment(dataset, gnn_type, epochs, lr, hs, gamma, nl)
        all_experiments[(dataset, gnn_type, lr, hs, gamma, nl)] = f1

        key = (dataset, gnn_type)
        if f1 > summary[key]['best_score']:
            summary[key]['best_score'] = f1
            summary[key]['best_params'] = {
                'lr': lr,
                'hidden_size': hs,
                'gamma': gamma,
                'num_layers': nl
            }
        print(f"Test F1: {f1:.4f}")
        with open('hs_242_experiments.txt', 'w') as f:
            for params, f1 in all_experiments.items():
                f.write(f"{params}: F1={f1:.4f}\n")
        pbar.update(1)
    pbar.close()

    # Save all experiment results
    with open('hs_242_experiments.txt', 'w') as f:
        for params, f1 in all_experiments.items():
            f.write(f"{params}: F1={f1:.4f}\n")

    # Print summary
    print("\n=== Hyperparameter Tuning Summary ===")
    for (ds, gt), rec in summary.items():
        print(f"Dataset: {ds}, GNN: {gt} -> Best F1={rec['best_score']:.4f} with {rec['best_params']}")
    with open('hs_242_summary.txt', 'w') as f:
        for (ds, gt), rec in summary.items():
            f.write(f"Dataset: {ds}, GNN: {gt} -> Best F1={rec['best_score']:.4f} with {rec['best_params']}\n")
