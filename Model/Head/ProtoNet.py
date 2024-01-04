import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

import os
import sys
sys.path.append(os.getcwd())

from util.cluster import KmeansClustering

def cos_sim(a,b,eps=1e-6):
    norm_a,norm_b=torch.norm(a,dim=1),torch.norm(b,dim=1)
    prod_norm=norm_a.unsqueeze(-1)*norm_b.unsqueeze(0)
    prod_norm[prod_norm<eps]=eps

    prod_mat=torch.matmul(a,b.permute(1,0))
    cos_sim=prod_mat/prod_norm

    return cos_sim



class ProtoNet(nn.Module):
    def __init__(self,n_way,k_shot,query):
        super().__init__()
        self.loss_fn=torch.nn.NLLLoss()
        
        self.n_way=n_way
        self.k_shot=k_shot
        self.query=query
  

    
    def get_dist(self,feat):
        support=feat[:self.n_way*self.k_shot]
        queries=feat[self.n_way*self.k_shot:]
        
        support=F.adaptive_avg_pool2d(support,1).squeeze()
        queries=F.adaptive_avg_pool2d(queries,1).squeeze()

        prototype=support.reshape(self.n_way,self.k_shot,-1).mean(1)
        distance=torch.cdist(queries.unsqueeze(0),prototype.unsqueeze(0)).squeeze(0)
        return distance
    
    def forward(self,feat,label):
        dist=self.get_dist(feat)
        y_pred=(-dist).softmax(1)
        
        log_p_y = (-dist).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, label[1].to(feat.device))
        
        return y_pred,loss



class ProtoNet_meshprj(nn.Module):
    def __init__(self,n_way,k_shot,query,prj_num=6):
        super().__init__()
        self.loss_fn=torch.nn.NLLLoss()
        
        self.n_way=n_way
        self.k_shot=k_shot
        self.query=query
        self.prj_num=prj_num
  

    
    def get_dist(self,feat):
        support=feat[:self.n_way*self.k_shot*self.prj_num]
        queries=feat[self.n_way*self.k_shot*self.prj_num:]


        support=F.adaptive_avg_pool2d(support,1).squeeze()
        queries=F.adaptive_avg_pool2d(queries,1).squeeze()

        prototype=support.reshape(self.n_way,self.k_shot*self.prj_num,-1).mean(1)
        distance=torch.cdist(queries.unsqueeze(0),prototype.unsqueeze(0)).squeeze(0)
        return distance
    
    def forward(self,feat,label):
        dist=self.get_dist(feat)
        y_pred=(-dist).softmax(1)
        
        log_p_y = (-dist).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, label[1].to(feat[0].device))
        
        return y_pred,loss




