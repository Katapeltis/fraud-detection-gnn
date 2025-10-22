import dgl
from dgl.data import FraudYelpDataset, FraudAmazonDataset
import torch
import networkx as nx
from sklearn.model_selection import train_test_split
import os

#raw_dir='/Users/marios/fraud_dgl'

def minmax(x):
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        else:
            return torch.zeros_like(x)
        
def enrich_node_features(graph):
    """
    For a single-node-type heterograph (multiple edge types):
     1) Collapse to homogeneous (so node IDs are 0..N-1)
     2) Compute degree, centrality, clustering on the undirected graph
     3) Concatenate these descriptors to graph.ndata['feature']
    """
    # Number of nodes and original features
    N       = graph.number_of_nodes()
    X_orig  = graph.ndata['feature']  # (N, D_orig)

    # 1) Homogenize: merges all edge types, keeps node order
    hom_g = dgl.to_homogeneous(graph)

    # 2) Degrees on homograph
    in_deg  = hom_g.in_degrees().float()   
    out_deg = hom_g.out_degrees().float() 
    tot_deg = in_deg + out_deg  

    in_deg   = minmax(in_deg).unsqueeze(1)   # (N,1)
    out_deg  = minmax(out_deg).unsqueeze(1)  # (N,1)
    tot_deg  = minmax(tot_deg).unsqueeze(1)  # (N,1)                        

    # 3) Build undirected NetworkX graph for global centralities
    print("Converting DGL graph to NetworkX for centrality calculations...")
    nx_g = dgl.to_networkx(hom_g)
    undirected_g = nx.Graph(nx_g).to_undirected()  # Convert to undirected

    # 4) Compute centralities
    print("Degree centrality")
    deg_cent_dict = nx.degree_centrality(nx_g)

    # cls_cent_dict = nx.closeness_centrality(nx_g)

    # btw_cent_dict = nx.betweenness_centrality(nx_g, normalized=True)
    print("Clustering coefficient")
    clust_dict    = nx.clustering(undirected_g)

    # 5) Turn dicts → aligned tensors
    deg_cent = torch.tensor([deg_cent_dict[i] for i in range(N)],
                             dtype=torch.float).unsqueeze(1)

    # cls_cent = torch.tensor([cls_cent_dict[i] for i in range(N)],
    #                          dtype=torch.float).unsqueeze(1)
    # btw_cent = torch.tensor([btw_cent_dict[i] for i in range(N)],
    #                          dtype=torch.float).unsqueeze(1)
    clust    = torch.tensor([clust_dict[i] for i in range(N)],
                             dtype=torch.float).unsqueeze(1)

    # 6) Triangle counts
    print("Calculating triangle counts...")
    tri_dict  = nx.triangles(undirected_g)
    triangles = torch.tensor([tri_dict[i] for i in range(N)],
                              dtype=torch.float)
    triangles = minmax(triangles).unsqueeze(1)  # (N,1)

    # 6) Concatenate everything in the original graph’s node order
    X_enh = torch.cat([
        X_orig, 
        in_deg, out_deg, tot_deg,
        deg_cent, triangles, clust
    ], dim=1)  # shape (N, D_orig + 8)

    graph.ndata['feature'] = X_enh
    print(f"Enriched nodes with {X_enh.shape[1] - X_orig.shape[1]} new feature(s).")
    return graph

def save_graph(graph: dgl.DGLGraph, path: str = "graph.dgl") -> None:
    """
    Save a DGLGraph object to the given path.
    """
    dgl.save_graphs(path, [graph])
    print(f"Graph saved to {path}")



def load_dataset(dataset: str, enriched: bool = False, load_path: str = None) -> dgl.DGLGraph:
    """
    Load a fraud dataset. If load_path is given and exists, load graph from disk.
    Otherwise load from DGL's built-in FraudYelp/Amazon.
    """
    if load_path is not None and os.path.exists(load_path):
        graphs, _ = dgl.load_graphs(load_path)
        graph = graphs[0]
        #print(f"Loaded graph from {load_path}")
    else:
        if dataset.lower() == 'yelp':
            data = FraudYelpDataset()
        elif dataset.lower() == 'amazon':
            data = FraudAmazonDataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        graph = data[0]
        if enriched:
            enrich_node_features(graph)

    assert 'feature' in graph.ndata and 'label' in graph.ndata, \
        "Graph must have 'feature' and 'label' in ndata"



    return graph


def train_test_split_graph(graph, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    labels = graph.ndata['label'].numpy()
    idx = list(range(graph.number_of_nodes()))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, labels, stratify=labels,
        test_size=(1 - train_ratio), random_state=random_state
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp, y_tmp, stratify=y_tmp,
        test_size=(1 - rel_val), random_state=random_state
    )

    train_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return train_mask, val_mask, test_mask
