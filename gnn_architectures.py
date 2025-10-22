import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroGNN(nn.Module):
    """
    Heterogeneous GNN using GraphConv layers.
    """
    def __init__(self,
                 in_feats: int,
                 hidden_size: int,
                 num_classes: int,
                 etypes: list,
                 num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype: dglnn.GraphConv(in_feats, hidden_size)
                 for etype in etypes},
                aggregate='mean'
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {etype: dglnn.GraphConv(hidden_size, hidden_size)
                     for etype in etypes},
                    aggregate='mean'
                )
            )

        # Output layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype: dglnn.GraphConv(hidden_size, num_classes)
                 for etype in etypes},
                aggregate='mean'
            )
        )
        self.dropout = nn.Dropout(0.14)

    def forward(self, graph, inputs):
        h = inputs
        # Apply all but last layer with ReLU
        for conv in self.layers[:-1]:
            h = conv(graph, h)
            h = {ntype: F.relu(feat) for ntype, feat in h.items()}
            h = {ntype: self.dropout(feat) for ntype, feat in h.items()}  # Apply dropout

        # Final layer without activation
        logits = self.layers[-1](graph, h)
        # Return raw logits and softmax probabilities for target node type
        primary_ntype = next(iter(logits.keys()))
        return logits, F.softmax(logits[primary_ntype], dim=1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # alpha should be a tensor of size (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt (probability of the true class)
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weights
        if self.alpha is not None:
            # Gather the alpha values corresponding to the targets
            alpha = self.alpha.gather(0, targets)
            focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    

class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GNN using SAGEConv layers.
    """
    def __init__(self,
                 in_feats: int,
                 hidden_size: int,
                 num_classes: int,
                 etypes: list,
                 num_layers: int,
                 aggregator_type: str = 'lstm'):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype: dglnn.SAGEConv(in_feats, hidden_size, aggregator_type)
                 for etype in etypes},
                aggregate='mean'
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {etype: dglnn.SAGEConv(hidden_size, hidden_size, aggregator_type)
                     for etype in etypes},
                    aggregate='mean'
                )
            )

        # Output layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype: dglnn.SAGEConv(hidden_size, num_classes, aggregator_type)
                 for etype in etypes},
                aggregate='mean'
            )
        )

    def forward(self, graph, inputs):
        h = inputs
        # Apply all but last layer with ReLU
        for conv in self.layers[:-1]:
            h = conv(graph, h)
            h = {ntype: F.relu(feat) for ntype, feat in h.items()}

        # Final layer without activation
        logits = self.layers[-1](graph, h)
        primary_ntype = next(iter(logits.keys()))
        return logits, F.softmax(logits[primary_ntype], dim=1)
    


class HeteroGAT(nn.Module):
    """
    Heterogeneous GNN using GATConv layers.
    """
    def __init__(self,
                 in_feats: int,
                 hidden_size: int,
                 num_classes: int,
                 etypes: list,
                 num_layers: int,
                 num_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype: dglnn.GATConv(in_feats, hidden_size // num_heads, num_heads)
                 for etype in etypes},
                aggregate='mean'
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {etype: dglnn.GATConv(hidden_size, hidden_size // num_heads, num_heads)
                     for etype in etypes},
                    aggregate='mean'
                )
            )

        # Output layer (single head)
        self.layers.append(
            dglnn.HeteroGraphConv(
                {etype: dglnn.GATConv(hidden_size, num_classes, 1)
                 for etype in etypes},
                aggregate='mean'
            )
        )

    def forward(self, graph, inputs):
        h = inputs
        # Apply all but last layer with ELU
        for conv in self.layers[:-1]:
            h = conv(graph, h)
            h = {ntype: F.elu(feat.flatten(1)) for ntype, feat in h.items()}

        # Final layer without activation
        logits = self.layers[-1](graph, h)
        primary_ntype = next(iter(logits.keys()))
        return logits, F.softmax(logits[primary_ntype], dim=1)