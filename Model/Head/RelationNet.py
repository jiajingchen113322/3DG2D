import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import math



class RelationNet(nn.Module):
    def __init__(self,n_way,k_shot,q_query,prj_num):
        super().__init__()
        self.n_way=n_way
        self.k_shot=k_shot
        self.q_query=q_query
        self.prj_num=prj_num

        self.lin=nn.Sequential(nn.Linear(128,64),
                               nn.ReLU(),
                               nn.Linear(64,16),
                               nn.ReLU(),
                               nn.Linear(16,4),
                               nn.ReLU(),
                               nn.Linear(4,1))

        self.loss_fn=torch.nn.CrossEntropyLoss()



    def forward(self,feat,label):
        # one_hot=F.one_hot(label[1],num_classes=self.n_way).to(feat.device)

        support_xf=feat[:self.n_way*self.k_shot*self.prj_num]
        query_xf=feat[self.n_way*self.k_shot*self.prj_num:]
        support_xf=F.adaptive_avg_pool2d(support_xf,1).squeeze()
        query_xf=F.adaptive_avg_pool2d(query_xf,1).squeeze()


        support_xf_proto=support_xf.reshape(self.n_way,self.k_shot*self.prj_num,-1).mean(1)
        query_xf_rp=query_xf.unsqueeze(1).repeat(1,self.n_way,1)
        support_xf_proto_pr=support_xf_proto.unsqueeze(0).repeat(self.q_query*self.n_way,1,1)
        cat_feat=torch.cat((support_xf_proto_pr,query_xf_rp),-1)
        
        score=self.lin(cat_feat).squeeze()
        
        prediction=score.softmax(-1)
        loss = self.loss_fn(score, label[1].to(feat.device))
        
        return prediction,loss





class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
		                       padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class RelationNet_prj(nn.Module):
    def __init__(self,n_way,k_shot,q_query,prj_num):
        super().__init__()
        self.n_way=n_way
        self.k_shot=k_shot
        self.q_query=q_query
        self.prj_num=prj_num

        # self.lin=nn.Sequential(nn.Linear(128,64),
        #                        nn.ReLU(),
        #                        nn.Linear(64,16),
        #                        nn.ReLU(),
        #                        nn.Linear(16,4),
        #                        nn.ReLU(),
        #                        nn.Linear(4,1),
        #                        nn.Sigmoid())
        

        self.c = 64
        self.d = 7
        # this is the input channels of layer4&layer5
        self.inplanes = 2 * self.c
        # assert repnet_sz[2] == repnet_sz[3]
        # print('repnet sz:', repnet_sz)

        # after relational module
        self.layer4 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 64, 3, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(256 , 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid())



    def forward(self,feat,label):
        one_hot=F.one_hot(label[1],num_classes=self.n_way).to(feat.device)

        support_xf=feat[:self.n_way*self.k_shot*self.prj_num]
        query_xf=feat[self.n_way*self.k_shot*self.prj_num:].unsqueeze(0)
        # support_xf=F.adaptive_avg_pool2d(support_xf,1).squeeze()
        # query_xf=F.adaptive_avg_pool2d(query_xf,1).squeeze()
        support_xf=support_xf.reshape(self.n_way,self.k_shot*self.prj_num,*support_xf.shape[1:]).mean(1).unsqueeze(0)
        

        batchsz, setsz, c_, h, w = support_xf.size()
        querysz = query_xf.size(1)
        c, d = self.c, self.d # c=1024, d=14

        # [b, setsz, c_, h, w] => [b*setsz, c_, h, w] => [b*setsz, c, d, d] => [b, setsz, c, d, d]
        # support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
        # # [b, querysz, c_, h, w] => [b*querysz, c_, h, w] => [b*querysz, c, d, d] => [b, querysz, c, d, d]
        # query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

        # concat each query_x with all setsz along dim = c
        # [b, setsz, c, d, d] => [b, 1, setsz, c, d, d] => [b, querysz, setsz, c, d, d]
        support_xf = support_xf.unsqueeze(1).expand(-1, querysz, -1, -1, -1, -1)
        # [b, querysz, c, d, d] => [b, querysz, 1, c, d, d] => [b, querysz, setsz, c, d, d]
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, setsz, -1, -1, -1)
        # cat: [b, querysz, setsz, c, d, d] => [b, querysz, setsz, 2c, d, d]
        comb = torch.cat([support_xf, query_xf], dim=3)

        comb = self.layer5(self.layer4(comb.view(batchsz * querysz * setsz, 2 * c, d, d)))
        # print('layer5 sz:', comb.size()) # (5*5*5, 256, 4, 4)
        comb = F.avg_pool2d(comb, 2)
        # print('avg sz:', comb.size()) # (5*5*5, 256, 1, 1)
        # push to Linear layer
        # [b * querysz * setsz, 256] => [b * querysz * setsz, 1] => [b, querysz, setsz, 1] => [b, querysz, setsz]
        score = self.fc(comb.view(batchsz * querysz * setsz, -1)).view(batchsz, querysz, setsz, 1).squeeze(3).squeeze(0)

        # build its label
        # [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
        # support_yf = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
        # # [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
        # query_yf = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
        # # eq: [b, querysz, setsz] => [b, querysz, setsz] and convert byte tensor to float tensor
        # label = torch.eq(support_yf, query_yf).float()



        
        # query_xf_rp=query_xf.unsqueeze(1).repeat(1,self.n_way,1)
        # support_xf_proto_pr=support_xf_proto.unsqueeze(0).repeat(self.q_query*self.n_way,1,1)
        # cat_feat=torch.cat((support_xf_proto_pr,query_xf_rp),-1)

        # score=self.lin(cat_feat).squeeze()

        prediction=score.softmax(-1)
        loss=torch.pow(score-one_hot,2).sum()

        return prediction,loss


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)







if __name__=='__main__':
    k_way=5
    n_shot=3
    q=10

    feat=torch.randn((260,64,7,7))
    # feat=torch.randn((120,64,7,7))
    label=torch.arange(5).repeat_interleave(10)        
    net=RelationNet(n_way=5,k_shot=3,q_query=10,prj_num=14)
    net(feat,[1,label])