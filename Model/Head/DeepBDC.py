import numpy as np
import torch
import torch.nn as nn



def BDCovpool(x, t):
    batchSize, dim, h, w = x.data.shape
    M = h * w
    x = x.reshape(batchSize, dim, M)

    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
    I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
    x_pow2 = x.bmm(x.transpose(1, 2))
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    
    dcov = torch.clamp(dcov, min=0.0)
    dcov = torch.exp(t)* dcov
    dcov = torch.sqrt(dcov + 1e-5)
    t = dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)

    return t


def Triuvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y





class DeepBDC(nn.Module):
    def __init__(self,n_way,k_shot,q_query,prj_num):
        super().__init__()
        self.n_way=n_way
        self.k_shot=k_shot
        self.q_query=q_query
        self.prj_num=prj_num

        input_dim=[64,7,7]
        self.temp=nn.Parameter(torch.log((1. / (2 * input_dim[1]*input_dim[2])) * torch.ones(1,1)), requires_grad=True)
        self.loss_fn = nn.CrossEntropyLoss()
    
    
    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        # if self.n_support > 1:
        dist = torch.pow(x - y, 2).sum(2)
        score = -dist
        # else:
        #     score = (x * y).sum(2)
        return score
    
    
    
    
    def forward(self,feat,label):
        feat=BDCovpool(feat,self.temp)
        feat=Triuvec(feat)
        
        support=feat[:self.n_way*self.k_shot*self.prj_num,:]
        support=support.reshape(self.n_way,self.k_shot*self.prj_num,-1).mean(1)
        query=feat[self.n_way*self.k_shot*self.prj_num:]
        
        score=self.metric(query,support)
        loss = self.loss_fn(score, label[1].to(feat[0].device))
        
        return score,loss
        
        
    
    

if __name__=='__main__':
    n_way=5
    k_shot=1
    q=10

    feat=torch.randn((120,64,7,7))
    # feat=torch.randn((120,64,7,7))
    label=torch.arange(5).repeat_interleave(10)   
    net=DeepBDC(n_way,k_shot,q,prj_num=14) 
    net(feat,[1,label])
        