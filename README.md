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

Before running the code, set up a dedicated **Anaconda environment** to ensure dependency isolation and CUDA compatibility.

### Step 1. Create and activate a new environment
```bash
conda create -n gnn_topk python=3.10
conda activate gnn_topk
```

### Step 2. Install core dependencies

```bash
conda install -c dglteam/label/cu121 dgl
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c pytorch torchdata
conda install pyyaml
conda install -c anaconda pydantic
conda install scikit-learn
```

Step 3. Install compatible library versions
```bash
pip install setuptools==60.0.0
pip install numpy==1.26.4
pip install pandas==2.3.0
pip install matplotlib==3.10.0
```
### Step 4. Run the experiments

Once the environment is ready, you can run the experiments by passing the **dataset** and **model** as command-line arguments.

#### Basic usage:
```bash
python gnn_main.py --dataset yelp --model GraphSAGE
```
