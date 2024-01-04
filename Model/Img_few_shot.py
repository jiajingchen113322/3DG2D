import os
import sys
sys.path.append(os.getcwd())

# ==== import backbone ====
from Model.Backbone.ResNet import resnet18
# ==========================

# ==== import few-shot head =====
from Model.Head.ProtoNet import ProtoNet
from Model.Head.FRN import FRN  
# ==========================


import torch
import torch.nn as nn



class Image_fewshot_Net(nn.Module):
    def __init__(self,n_way,k_shot,query,backbone='ResNet',fs='ProtoNet'):
        super().__init__()
        self.n_way=n_way
        self.k_shot=k_shot
        self.query=query
        
        self.s_label=torch.arange(n_way)
        self.q_label=torch.arange(n_way).repeat_interleave(query)        

        self.backbone=self.get_backbone(backbone)
        self.fs_head=self.get_fs(fs)
    
    def get_backbone(self,backbone):
        if backbone=='ResNet':
            return resnet18()

        else:
            raise ValueError('Not Implemented')
    
    def get_fs(self,fs):
        if fs=='ProtoNet':
            return ProtoNet(self.n_way,self.k_shot,self.query)
        
        elif fs=='FRN':
            return FRN(self.n_way,self.k_shot,self.query)
        
        else:
            raise ValueError('Not Implemented')
    
    
    def forward(self,inpt):
        # support,query=inpt[0],inpt[1]
        # inpt=torch.cat((support,query))
        
        embeding=self.backbone(inpt)
        # ==== get support and query embeding =====
        # support_feat=embeding[:self.n_way*self.n].reshape(self.n_way,self.n,-1)
        # support_feat=torch.mean(support_feat,1)
        # query_feat=embeding[self.n*self.n_way:]
        # =========================================
        
        pred,loss=self.fs_head(embeding,[self.s_label,self.q_label])
        return pred,loss
        
    

        
        

if __name__=='__main__':
    n_way=5
    k_shot=1
    q=3
    
    support=torch.randn((n_way*k_shot,3,224,224)) # n_way*k_shot images
    query=torch.randn((n_way*q,3,224,224))  # n_way*q_query images
    inpt=torch.cat((support,query),0)
    
    net=Image_fewshot_Net(n_way=n_way,k_shot=k_shot,query=q,fs='FRN')
    net(inpt)
    
    