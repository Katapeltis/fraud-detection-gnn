import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, etypes):
        super(HeteroGNN, self).__init__()
        # Define a separate GCN layer for each edge type
        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(in_feats, h_feats) for etype in etypes
        })
        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(h_feats, num_classes) for etype in etypes
        })

    def forward(self, g, inputs):
        # Inputs is a dictionary of node types and their features
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
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