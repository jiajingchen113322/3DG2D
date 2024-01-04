import os
import sys
sys.path.append(os.getcwd())

# ==== import backbone ====
from Model.Backbone.ResNet import resnet10
# ==========================

# ==== import few-shot head =====
from Model.Head.ProtoNet import ProtoNet_meshprj
from Model.Head.AINet import AINet

from Model.Head.FRN import FRN_Projection  
from Model.Head.RelationNet import RelationNet_prj
from Model.Head.DeepBDC import DeepBDC
# ==========================

import torch.nn.functional as F
import torch
import torch.nn as nn



class ThreeD_Support_Net(nn.Module):
    def __init__(self,n_way,k_shot,query,backbone='ResNet',fs='AINet',prj_num=14,pretrain=False,pretrain_path=None,alpha=0.3):
        super().__init__()
        self.alpha=alpha
        self.n=n_way
        self.k=k_shot
        self.query=query
        self.prj_num=prj_num
        self.composite_num=0
        self.pretrain=pretrain
        self.pretrain_path=pretrain_path

        self.s_label=torch.arange(n_way)
        self.q_label=torch.arange(n_way).repeat_interleave(query)        

        self.backbone=self.get_backbone(backbone)
        self.fs_head=self.get_fs(fs)

        
        


    def get_backbone(self,backbone):
        if backbone=='ResNet':
            backbone=resnet10(inchannel=3,Return_Featmap=False)
            if self.pretrain:
                pth_file=torch.load(self.pretrain_path)
                backbone.load_state_dict(pth_file['model_state'])
            return backbone

        else:
            raise ValueError('Not Implemented')
    
    def get_fs(self,fs):
        if fs=='ProtoNet':
            return ProtoNet_meshprj(self.n,self.k,self.query,prj_num=self.prj_num+self.composite_num)
            # return ProtoNet_pushing_3weight(self.n,self.k,self.query,prj_num=self.prj_num+self.composite_num,alpha=self.alpha)

        elif fs=='FRN':
            return FRN_Projection(self.n,self.k,self.query,self.prj_num)
        
        elif fs=='Relation':
            return RelationNet_prj(self.n,self.k,self.query,self.prj_num)

        elif fs=='BDC':
            return DeepBDC(self.n,self.k,self.query,self.prj_num)
        
        elif fs=='AINet':
            return AINet(self.n,self.k,self.query,prj_num=self.prj_num+self.composite_num,alpha=self.alpha)


        else:
            raise ValueError('Not Implemented')
    


    
    def forward(self,inpt):
        # ==== get concatenated input === #
        prj,query=inpt
        support_sample_num,prj_num=prj.shape[:2]
        prj=prj.reshape(support_sample_num*prj_num,*list(prj.shape[2:]))
        cat_inpt=torch.cat((prj,query),0) # the shape is (support_sample*prj_face+query_num,3,224,224)
        embeding =self.backbone(cat_inpt)
        pred,loss=self.fs_head(embeding,[self.s_label,self.q_label])

        return pred,loss
        
    

        
        

if __name__=='__main__':
    n_way=5
    k_shot=1
    q=10
    
    '''
    For support set, we have total n_way*k_shot 3D mesh samples, for each 3D mesh sample, we have 14 projections, each projection image has shape (3,224,224).
    For query set, we have total n_way*q query images, with each image's shape as (3,224,224)
    '''
    support_projectoin=torch.randn((n_way*k_shot,14,3,224,224)) #totally we have n_way*k_shot 3D samples, for each sample, we have 14 projection iamges.
    query_image=torch.randn((n_way*q,3,224,224)) # we have total n_way*q query images.
    
    net=ThreeD_Support_Net(n_way=n_way,k_shot=k_shot,query=q,fs='AINet')
    pred,loss=net([support_projectoin,query_image])
    
    