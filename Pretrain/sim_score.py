import torch
import numpy as np
import torch.nn.functional as F

def info_nce_loss(features):
    temp=0.07

    device=features.device
    batch_size=int(features.shape[0]//2)
    features=F.adaptive_avg_pool2d(features,1).squeeze()


    labels=torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temp
    return logits, labels




if __name__=='__main__':
    feat=torch.randn((120,64,7,7))
    info_nce_loss(feat)