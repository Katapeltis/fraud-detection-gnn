# Top-k Outlier Detection in Graphs Using Graph Neural Networks

This repository contains the implementation and experimental framework for the dissertation:

**“Top-k Outlier Detection in Graphs Using Graph Neural Networks – A Novel Approach of Evaluating GNN Architectures for Anomaly Detection.”**

The work introduces and evaluates a **top-k anomaly detection methodology** for node-level outlier detection tasks in graph-structured data. It explores several GNN architectures, including **GCN, GraphSAGE, GAT (synthetic interpolation), and BWGNN**, with modifications to incorporate an additional **sigmoid probability layer** for ranking anomalies.

---


## 📁 Project Structure

```bash
fraud-detection-gnn/
├── datasets.py # Dataset loading, enrichment with structural features
├── gnn_architectures.py # GNN architectures and custom loss functions
├── model.py # Training and evaluation utilities
├── gnn_main.py # Main training + top-k evaluation loop
├── top_k.py # Functions for top-k accuracy computation
├── hyperparameters.py # Hyperparameter configuration
├── result_plots.py # Plot generation (heatmaps, error bars, accuracy curves)
```
## 📊 Datasets

Two benchmark datasets from the **DGL Fraud Detection suite** are used:

- **FraudYelp** – reviews dataset with labeled fraudulent accounts.
- **FraudAmazon** – e-commerce dataset with fraudulent review accounts.

The `datasets.py` script optionally **enriches node features** with:
- Degree (in/out/total)
- Degree centrality
- Clustering coefficient
- Triangle counts

---

## ⚙️ Instructions





