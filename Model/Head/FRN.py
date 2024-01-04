import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import math


class FRN(nn.Module):
    def __init__(self,n_way,k_shot,q_query):
        super().__init__()
        self.nway=n_way
        self.k_shot=k_shot
        self.q=q_query

        self.d=64
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.criterion = nn.CrossEntropyLoss()


    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
        reg = support.size(1)/support.size(2)

        # correspond to lambda in the paper
        lam = reg*alpha.exp()

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution

        if Woodbury:
        # correspond to Equation 10 in the paper

            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way

        return dist



    def get_neg_l2_dist(self,support_xf,query_xf):
        '''
        support_xf: (k_way*n_shot,c,h,w)
        query_xf: (k_way*query,c,h,w)
        '''


        q,c,h,w=query_xf.shape

        support_xf=support_xf/math.sqrt(self.d)
        query_xf=query_xf/math.sqrt(self.d)

        support_xf=support_xf.view(self.nway,self.k_shot,c,-1).permute(0,1,3,2).contiguous()
        support_xf=support_xf.reshape(self.nway,-1,c) #(k_way, total_pixel_num_in_each_way, channel)

        query_xf=query_xf.view(q,c,-1).permute(0,2,1).contiguous()
        query_xf=query_xf.reshape(q*h*w,c) # (total_pixel_number,channel)

        alpha = self.r[0]
        beta = self.r[1]

        recon_dist=self.get_recon_dist(query_xf,support_xf,alpha, beta)
        neg_l2_dist=recon_dist.neg().reshape(-1,h*w,self.nway).mean(1)
        return neg_l2_dist



    def forward(self,feat,label):
        support_xf=feat[:self.k_shot*self.nway] # (support_img_num,channel,h,w)
        query_xf=feat[self.k_shot*self.nway:] # (query_img_num,channel,h,w)
        
        neg_l2_dist=self.get_neg_l2_dist(support_xf,query_xf)
        
        y_pred=neg_l2_dist.softmax(1)

        logits=neg_l2_dist*self.scale
        loss=self.criterion(logits,label[1].to(feat.device))
        
        return y_pred,loss





class FRN_Projection(nn.Module):
    def __init__(self,n_way,k_shot,q_query,prj_num):
        super().__init__()
        self.n_way=n_way
        self.k_shot=k_shot
        self.q=q_query
        self.prj_num=prj_num
        
        self.d=64
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)
        # self.r = nn.Parameter(torch.FloatTensor([0.5,0.5]), requires_grad=True)

        self.criterion = nn.CrossEntropyLoss()


    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
        reg = support.size(1)/support.size(2)

        # correspond to lambda in the paper
        lam = reg*alpha.exp()

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution

        if Woodbury:
        # correspond to Equation 10 in the paper

            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way

        return dist



    def get_neg_l2_dist(self,support_xf,query_xf):
        '''
        support_xf: (k_way*n_shot,c,h,w)
        query_xf: (k_way*query,c,h,w)
        '''


        q,c,h,w=query_xf.shape

        support_xf=support_xf/math.sqrt(self.d)
        query_xf=query_xf/math.sqrt(self.d)

        support_xf=support_xf.view(self.n_way,self.k_shot*self.prj_num,c,-1).permute(0,1,3,2).contiguous()
        support_xf=support_xf.reshape(self.n_way,-1,c) #(k_way, total_pixel_num_in_each_way, channel)

        query_xf=query_xf.view(q,c,-1).permute(0,2,1).contiguous()
        query_xf=query_xf.reshape(q*h*w,c) # (total_pixel_number,channel)

        alpha = self.r[0]
        beta = self.r[1]

        recon_dist=self.get_recon_dist(query_xf,support_xf,alpha, beta)
        neg_l2_dist=recon_dist.neg().reshape(-1,h*w,self.n_way).mean(1)
        return neg_l2_dist



    def forward(self,feat,label):
        support_xf=feat[:self.n_way*self.k_shot*self.prj_num] # (support_img_num,channel,h,w)
        query_xf=feat[self.n_way*self.k_shot*self.prj_num:] # (query_img_num,channel,h,w)
        
        # support_xf=F.adaptive_max_pool2d(support_xf,query_xf.shape[-1])

        # support_xf=feat[:self.n*self.k] # (support_img_num,channel,h,w)
        # query_xf=feat[self.n*self.k:] # (query_img_num,channel,h,w)
        
        neg_l2_dist=self.get_neg_l2_dist(support_xf,query_xf)
        
        y_pred=neg_l2_dist.softmax(1)

        logits=neg_l2_dist*self.scale
        loss=self.criterion(logits,label[1].to(feat[1].device))
        
        return y_pred,loss












if __name__=='__main__':
    k_way=5
    n_shot=1
    q=3

    support_xf=torch.randn((k_way*n_shot,64,7,7))
    query_xf=torch.randn((k_way*q,64,7,7))
    feat=torch.cat((support_xf,query_xf))

    s_label=torch.arange(k_way)
    q_label=torch.arange(k_way).repeat_interleave(q)
    label=[s_label,q_label]

    net=FRN(k_way,n_shot,q)
    net(feat,label)
    