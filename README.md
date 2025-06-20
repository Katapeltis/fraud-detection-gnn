# Fraud Detection using Graph Neural Networks (GNNs)

A research on anomaly detection in large graphs using several GNN architectures and other algorithms.
Goal of the project (high-lvl): **Find top-k node anomalies** from a given graph dataset

---


## 📁 Project Structure

```bash
fraud-detection-gnn/
├── datasets.py            # Some utils functions
├── gnn_architectures.py   # Contains the GNN classes (GCN, GraphSAGE etc)
├── gnn_main.py            # Execute this to run the magic
├── model.py               # The train & evaluate functions
```

## ⚙️ Instructions

Reqirements:
dgl.data
pytorch
sklearn

1. From datasets.py, change the raw_dir to the directory where the dgl datasets are stored
2. Run gnn_main.py from an editor. There are several parameters hardcoded in this file (epochs, dataset, architecture etc)

## TODO

So far the code can train a gnn on a train set, predict and evaluate a test set. 
Next step: Get a probability score for each node, that would correlate with the probability of belonging to class 1.

Attempted: in HeteroGNN class, added a sigmoid layer (probs) after the output layer, with the hope that, big score for a node -> belongs to Class 1. Doesn't work for now.



