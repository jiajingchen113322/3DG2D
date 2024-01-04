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




class AINet(nn.Module):
    ### this works a little bit ######
    def __init__(self,n_way,k_shot,query,prj_num=14,alpha=0.3):
        super().__init__()
        self.loss_fn=torch.nn.NLLLoss()
        
        self.n_way=n_way
        self.k_shot=k_shot
        self.query=query
        self.prj_num=prj_num

        self.alpha=alpha

    
    def get_dist(self,feat):
        support=feat[:self.n_way*self.k_shot*self.prj_num]
        queries=feat[self.n_way*self.k_shot*self.prj_num:]


        support=F.adaptive_avg_pool2d(support,1).squeeze()
        queries=F.adaptive_avg_pool2d(queries,1).squeeze()

        prototype=support.reshape(self.n_way,self.k_shot*self.prj_num,-1).mean(1)
        distance=torch.cdist(queries.unsqueeze(0),prototype.unsqueeze(0)).squeeze(0)
        return distance
    
    def get_regularized_dist(self,feat):
        support=feat[:self.n_way*self.k_shot*self.prj_num]
        queries=feat[self.n_way*self.k_shot*self.prj_num:]
        
        support=F.adaptive_avg_pool2d(support,1).squeeze()
        queries=F.adaptive_avg_pool2d(queries,1).squeeze()
        
        feat_dim=queries.shape[-1]

        support=support.reshape(self.n_way,self.k_shot,self.prj_num,-1)
        support_up=support[:,:,[0,3,6,9,13],:].reshape(self.n_way,-1,feat_dim) # support feature of top view projections
        support_center=support[:,:,[1,4,7,10],:].reshape(self.n_way,-1,feat_dim) # support feature of horizontal view projections
        support_bottom=support[:,:,[2,5,8,11,12],:].reshape(self.n_way,-1,feat_dim) # support feature of bottom view projections


        up_proto=torch.mean(support_up,1,keepdim=True)
        center_proto=torch.mean(support_center,1,keepdim=True)
        bottom_proto=torch.mean(support_bottom,1,keepdim=True)
        
        support_proto=torch.cat((up_proto,center_proto,bottom_proto),1)
        queries=queries.unsqueeze(0).repeat(self.n_way,1,1)
        dist=torch.cdist(queries,support_proto).permute(1,0,2)
        return dist,support_proto
        


    def get_weighted_dist(self,area_prototype,set_dist,dist,feat):
        queries=feat[self.n_way*self.k_shot*self.prj_num:]
        queries=F.adaptive_avg_pool2d(queries,1).squeeze()


        p_a_cls=(-set_dist).softmax(-1)

        # p_cls_a=(-set_dist).softmax(-2) # added

        p_cls=(-dist).softmax(-1).unsqueeze(-1).repeat(1,1,3)
        p_a=torch.sum(p_a_cls*p_cls,1).softmax(-1)

        # p_a=torch.sum((p_a_cls*p_cls)/(p_cls_a+1e-5),1).softmax(-1) # added
        
        proto_list=[]
        for p in p_a:
            p=p.unsqueeze(0).unsqueeze(-1).repeat(5,1,64)
            proto=torch.sum(p*area_prototype,1)
            proto_list.append(proto)
        
        renewed_protolist=torch.stack(proto_list,0)
        
        final_dist=torch.cdist(queries.unsqueeze(1),renewed_protolist).squeeze()
        return final_dist



    def forward(self,feat,label):
        # === normal distance and prediction ====
        dist=self.get_dist(feat)
        y_pred=(-dist).softmax(1)
        
        log_p_y = (-dist).log_softmax(dim=1)
        cls_loss = self.loss_fn(log_p_y, label[1].to(feat[0].device)) # classification loss
        # ========================================

        # ==== get the distance from target to closet and farest set =====
        set_dist,area_prototype=self.get_regularized_dist(feat)
        sorted_dist=torch.sort(set_dist,-1)[0]
        
        AI_loss=0 # Angle Inference Loss
        for i in range(sorted_dist.shape[-1]):
            t_dist=sorted_dist[:,:,i]
            log_p_y_t = (-t_dist).log_softmax(dim=1)
            AI_loss+=self.loss_fn(log_p_y_t, label[1].to(feat[0].device))


        loss=(1-self.alpha)*cls_loss+self.alpha*AI_loss
        # =========================================
        w_dist=self.get_weighted_dist(area_prototype,set_dist,dist,feat)
        weight_y_pred=(-w_dist).softmax(1) # Final prediction 

        
        return weight_y_pred,loss






    
        
if __name__=='__main__':
    n_way=5
    k_shot=1
    q=10
    
    feat=torch.randn((n_way*k_shot*14+n_way*q,64,14,14)) # (number of total images (projections + query), channel, height, width)
    label=torch.arange(n_way).repeat_interleave(q)       # the label of each query images 
    net=AINet(n_way=5,k_shot=1,query=10,prj_num=14)
    net(feat,[None,label])