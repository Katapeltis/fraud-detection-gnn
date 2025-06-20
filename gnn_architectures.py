import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, etypes):
        super(HeteroGNN, self).__init__()
        # Layer 1: Input to hidden
        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(in_feats, h_feats) for etype in etypes
        }, aggregate='sum')
        
        # Layer 2: Hidden to hidden
        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(h_feats, h_feats) for etype in etypes
        }, aggregate='sum')
        
        # Layer 3: Hidden to hidden
        self.conv3 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(h_feats, h_feats) for etype in etypes
        }, aggregate='sum')
        
        # Layer 4: Hidden to output
        self.conv4 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(h_feats, num_classes) for etype in etypes
        }, aggregate='sum')
        
        # Dropout layer
        self.dropout = nn.Dropout(0.15)

    def forward(self, g, inputs):
        # Layer 1
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}  # dropout
        
        # Layer 2
        h = self.conv2(g, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}  # dropout
        
        # Layer 3
        h = self.conv3(g, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}  # dropout
        
        # Layer 4
        h = self.conv4(g, h)
        ##### SIGMOID LAYER THAT SHOULD GIVE PROBABILITIES #####
        probs = {p: torch.sigmoid(v) for p, v in h.items()}
        return h
  

class HeteroGAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, etypes, num_heads):
        super(HeteroGAT, self).__init__()
        # Layer 1: Input to hidden
        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(in_feats, h_feats, num_heads) for etype in etypes
        }, aggregate='mean')
        
        # Layer 2: Hidden to output
        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(h_feats * num_heads, num_classes, 1) for etype in etypes  # Single head for output
        }, aggregate='mean')

    def forward(self, g, inputs):
        # Layer 1
        h = self.conv1(g, inputs)
        h = {k: F.elu(v) for k, v in h.items()}  # Apply ELU activation
        h = {k: v.view(-1, v.size(1) * v.size(2)) for k, v in h.items()}  # Flatten the head dimensions
        
        # Layer 2
        h = self.conv2(g, h)
        h = {k: v.squeeze(1) for k, v in h.items()}  # Remove the head dimension
        return h


class HeteroGraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, etypes, aggregator_type='mean'):
        super(HeteroGraphSAGE, self).__init__()
        # Layer 1: Input to hidden
        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(in_feats, h_feats, aggregator_type) for etype in etypes
        }, aggregate='mean')
        
        # Layer 2: Hidden to output
        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(h_feats, num_classes, aggregator_type) for etype in etypes
        }, aggregate='mean')

    def forward(self, g, inputs):
        # Layer 1
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}  # Apply ReLU activation
        
        # Layer 2
        h = self.conv2(g, h)
        return h
        

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
