import torch
from tqdm import tqdm
import argparse
import numpy as np
import torch.nn as nn

import os
import sys
sys.path.append('/home/jchen152/workspace/See_Like_Human/See_Like_Human/3D_help_2D')


# ====== import model =========
from Model.Backbone.ResNet import resnet10
from Pretrain.sim_score import info_nce_loss
# =============================
from torch.utils.tensorboard import SummaryWriter
import yaml


# ========= Get Configuration ==============
def get_arg():
    cfg=argparse.ArgumentParser()
    cfg.add_argument('--exp_name',default='Pretrain_ShapeNet_fold0')
    cfg.add_argument('--dataset',default='ShapeNet',choices=['ModelNet40','toy4k','ShapeNet'])
    cfg.add_argument('--multigpu',default=False)
    cfg.add_argument('--epochs',default=100,type=int)
    cfg.add_argument('--decay_ep',default=15,type=int)
    cfg.add_argument('--gamma',default=0.7,type=float)
    cfg.add_argument('--lr',default=1e-3,type=float)
    cfg.add_argument('--train',action='store_true',default=True)
    cfg.add_argument('--seed',default=0)
    cfg.add_argument('--device',default='cuda')
    cfg.add_argument('--lr_sch',default=True)
    cfg.add_argument('--fold',default=0,type=int)
    
    # ========path needed =============
    cfg.add_argument('--project_path',default='/home/jchen152/workspace/See_Like_Human/See_Like_Human/3D_help_2D')
    cfg.add_argument('--data_path',default='/home/jchen152/workspace/Data/See_Like_Human/ModelNet40-LS') # modelnet40
    # cfg.add_argument('--data_path',default='/home/jchen152/workspace/Data/See_Like_Human/TOYS4K')
    # =================================
    
    return cfg.parse_args()

cfg=get_arg()


# ======= import getset ======
if cfg.dataset=='ModelNet40':
    # cfg.data_path='/home/jchen152/workspace/Data/See_Like_Human/ModelNet40-LS'
    from Pretrain.Data_Loader.ModelNet40 import get_sets

elif cfg.dataset=='toy4k':
    # cfg.data_path='/home/jchen152/workspace/Data/See_Like_Human/TOYS4K'
    from Pretrain.Data_Loader.Toy4k import get_sets 

elif cfg.dataset=='ShapeNet':
    # cfg.data_path='/home/jchen152/workspace/Data/See_Like_Human/ShapeNet55-LS'
    from Pretrain.Data_Loader.ShapeNet55 import get_sets


def main(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
    
    train_loader=get_sets(cfg.data_path,cfg.fold)
    model=resnet10(inchannel=3,Return_Featmap=False)

    if cfg.multigpu:
        model=nn.DataParallel(model)
    
    if cfg.train:
        train_model(model,train_loader,cfg)
    
    






def train_model(model,train_loader,cfg):
    device=torch.device(cfg.device)
    model=model.to(device)
    
    #====== loss and optimizer =======
    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    if cfg.lr_sch:
        lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(10,cfg.epochs,cfg.decay_ep),gamma=cfg.gamma)
    
    
    def train_one_epoch():
        bar=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,bar,'train',optimizer=optimizer,loss_func=loss_func)
        summary={"loss/train":np.mean(epsum['loss'])}
        return summary
        
    
    
    # ======== define exp path ===========
    exp_path=os.path.join(cfg.project_path,'Pretrain','Exp',cfg.exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)


    # save config into json #
    cfg_dict=vars(cfg)
    yaml_file=os.path.join(exp_path,'config.yaml')
    with open(yaml_file,'w') as outfile:
        yaml.dump(cfg_dict, outfile, default_flow_style=False)
    # f = open(json_file, "w")
    # json.dump(cfg_dict, f)
    # f.close()
    #########################
    
    tensorboard=SummaryWriter(log_dir=os.path.join(exp_path,'TB'),purge_step=cfg.epochs)
    pth_path=os.path.join(exp_path,'pth_file')
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    # =====================================
    
    # ========= train start ===============

    tqdm_epochs=tqdm(range(cfg.epochs),unit='epoch',ncols=100)
    for e in tqdm_epochs:
        summary=train_one_epoch()      
        if cfg.lr_sch:
            lr_schedule.step()
        
        if (e%10==0) or (e==len(tqdm_epochs)-1):
            summary_saved={**summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict()}
            torch.save(summary_saved,os.path.join(pth_path,'epoch_{}'.format(e)))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    
    # =======================================    
    
    



def run_one_epoch(model,bar,mode,optimizer=None,loss_func=None,show_interval=10):
    summary={"loss":[]}
    device=next(model.parameters()).device
    
    if mode=='train':
        model.train()
    else:
        model.eval()
    
    for i, (data1_cpu,data2_cpu) in enumerate(bar):
        data=torch.cat((data1_cpu,data2_cpu),0).to(device)

        optimizer.zero_grad()
        feat=model(data)
        logits,labels=info_nce_loss(feat)
        loss = loss_func(logits, labels)
        #==take one step==#
        loss.backward()
        optimizer.step()
        #=================#
        
        
        summary['loss']+=[loss.item()]
        
        
        if i%show_interval==0:
            bar.set_description("Loss: %.3f"%(np.mean(summary['loss'])))
       
    
    return summary
            



if __name__=='__main__':
    main(cfg)
    
