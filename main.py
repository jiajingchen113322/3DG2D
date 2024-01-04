import torch
from tqdm import tqdm
import argparse
import numpy as np

from util.get_acc import cal_cfm
import torch.nn as nn

# ====== import model =========
from Model.Img_few_shot import Image_fewshot_Net

from Model.Img_few_shot_prj import ThreeD_Support_Net

# =============================


import logging

import os
from torch.utils.tensorboard import SummaryWriter
import yaml


# ========= Get Configuration ==============
def get_arg():
    cfg=argparse.ArgumentParser()
    cfg.add_argument('--exp_name',default='simple_try')

    cfg.add_argument('--dataset',default='ModelNet40',choices=['ModelNet40','toy4k','ShapeNet55'])
    cfg.add_argument('--epochs',default=80)
    cfg.add_argument('--decay_ep',default=10)
    cfg.add_argument('--gamma',default=0.7)
    cfg.add_argument('--lr',default=1e-3)
    cfg.add_argument('--train',action='store_true',default=True)
    cfg.add_argument('--seed',default=0)
    cfg.add_argument('--device',default='cuda')
    cfg.add_argument('--lr_sch',default=True)
    cfg.add_argument('--prj_num',default=14)
    cfg.add_argument('--pretrain',default=False)
    cfg.add_argument('--pretrain_path',default='/home/jchen152/workspace/See_Like_Human/See_Like_Human/3D_help_2D/Pretrain/Exp/Pretrain_ShapeNet_fold0/pth_file/epoch_99')

    # ==== specify angle or object ===
    cfg.add_argument('--point_support',default=1,type=int)
    cfg.add_argument('--mode',default='random',type=str)
    cfg.add_argument('--alpha',default=0.3,type=float)
    # ================================


    # ========= Few-shot cfg ==============
    cfg.add_argument('--n_way',default=5,type=int)
    cfg.add_argument('--k_shot',default=1,type=int)
    cfg.add_argument('--query',default=10,type=int)
    cfg.add_argument('--fold',default=0,type=int)
    # =====================================
    
    # ========= Net config =================
    cfg.add_argument('--backbone',default='ResNet',choices=['ResNet'])
    cfg.add_argument('--fs_head',default='AINet',choices=['ProtoNet','FRN','Relation','BDC','AINet'])
    # =====================================
    
    
    # ========path needed =============
    cfg.add_argument('--project_path',default='path to which you save your code')
    cfg.add_argument('--data_path',default='path of the dataset, for example:path of ModelNet40-LS folder') # modelnet40
    # =================================
    
    return cfg.parse_args()

cfg=get_arg()





# ======= import getset ======
if cfg.dataset=='ModelNet40':
    cfg.num_cls=10 # for modelnet40, each time we use 10 class for evaluation.
    cfg.exp_folder_name='ModelNet40_cross' # here should be uncomment
    # cfg.exp_folder_name='Ablation_Study'
    from Dataloader.ModelNet40_split import get_sets

elif cfg.dataset=='toy4k':
    if cfg.fold==3:
        cfg.num_cls=30
    else:
        cfg.num_cls=25
    cfg.exp_folder_name='Toy4k_cross'
    from Dataloader.Toy4K import get_sets

elif cfg.dataset=='ShapeNet55':
    if cfg.fold==3:
        cfg.num_cls=13
    else:
        cfg.num_cls=14
    cfg.exp_folder_name='ShapeNet55_cross'
    from Dataloader.ShapeNet55 import get_sets


# ============================

# ========= create logger ============
def get_logger(file_name='accuracy.log'):
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s, %(name)s, %(message)s')

    ########### this is used to set the log file ##########
    Exp_fold_path=os.path.join(cfg.project_path,'Exp',cfg.exp_folder_name,cfg.exp_name)
    if not os.path.exists(Exp_fold_path):
        os.makedirs(Exp_fold_path)
    
    file_path=os.path.join(Exp_fold_path,file_name)    
    file_handler=logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    #######################################################

    #### this is used to set the output in the terminal/screen ########
    stream_handler=logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    ##################################################################

    ## add the log file handler and terminal handerler to the logger ##
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

logger=get_logger()




def main(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
    
    train_loader,val_loader=get_sets(cfg.data_path,fold=cfg.fold,n_way=cfg.n_way,k_shot=cfg.k_shot,query_num=cfg.query,point_support=cfg.point_support,prj_num=cfg.prj_num,mode=cfg.mode)

    if not cfg.point_support:
        model=Image_fewshot_Net(n_way=cfg.n_way,k_shot=cfg.k_shot,query=cfg.query,backbone=cfg.backbone,fs=cfg.fs_head)
    else:
        model=ThreeD_Support_Net(n_way=cfg.n_way,k_shot=cfg.k_shot,query=cfg.query,backbone=cfg.backbone,fs=cfg.fs_head,prj_num=cfg.prj_num,pretrain=cfg.pretrain,pretrain_path=cfg.pretrain_path,alpha=cfg.alpha)
     
    
    if cfg.train:
        train_model(model,train_loader,val_loader,cfg)
    
    else:
        # raise ValueError('Not Implemented')
        test_model(model,val_loader,cfg)
    

def test_model(model,val_loader,cfg):
    device=torch.device(cfg.device)

    # ======== load device ===========
    pth_fold=os.path.join('./Exp',cfg.exp_folder_name,cfg.exp_name,'pth_file')
    pth_file_list=os.listdir(pth_fold)
    pth_file_list=sorted(pth_file_list,key=lambda x: int(x.split('_')[-1]))
    target_pth_file=os.path.join(pth_fold,pth_file_list[-1])
    pth_file=torch.load(target_pth_file)
    model.load_state_dict(pth_file['model_state'])
    model=model.to(device)
    # =================================

    bar=tqdm(val_loader,ncols=100,unit='batch',leave=False)
    summary=run_one_epoch(model,bar,mode='test')
    np.save('point_summary',summary['cfm'])
    



def train_model(model,train_loader,val_loader,cfg):
    device=torch.device(cfg.device)
    model=model.to(device)
    
    #====== loss and optimizer =======
    # loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    if cfg.lr_sch:
        lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(10,cfg.epochs,cfg.decay_ep),gamma=cfg.gamma)
    
    
    def train_one_epoch():
        bar=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,bar,'train',optimizer=optimizer)
        summary={"loss/train":np.mean(epsum['loss'])}
        return summary
        
        
    def eval_one_epoch():
        bar=tqdm(val_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,bar,"valid")
        mean_acc=np.mean(epsum['acc'])
        summary={'meac':mean_acc}
        summary["loss/valid"]=np.mean(epsum['loss'])
        return summary,epsum['cfm'],epsum['acc']
    
    
    # ======== define exp path ===========
    exp_path=os.path.join(cfg.project_path,'Exp',cfg.exp_folder_name,cfg.exp_name)
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
    acc_list=[]
    interval_list=[]
    
    tqdm_epochs=tqdm(range(cfg.epochs),unit='epoch',ncols=100)
    for e in tqdm_epochs:
        train_summary=train_one_epoch()
        val_summary,conf_mat,batch_acc_list=eval_one_epoch()
        summary={**train_summary,**val_summary}
        
        if cfg.lr_sch:
            lr_schedule.step()
        
        accuracy=val_summary['meac']
        acc_list.append(val_summary['meac'])
        
        # === get 95% interval =====
        std_acc=np.std(batch_acc_list)
        interval=1.960*(std_acc/np.sqrt(len(batch_acc_list)))
        interval_list.append(interval)
        # ===========================
        
        
        max_acc_index=np.argmax(acc_list)
        max_ac=acc_list[max_acc_index]
        max_interval=interval_list[max_acc_index]
        logger.debug('epoch {}: {}. Highest: {}. Interval: {}'.format(e,accuracy,max_ac,max_interval))
        # print('epoch {}: {}. Highese: {}'.format(e,accuracy,np.max(acc_list)))
        
        if np.max(acc_list)==acc_list[-1]:
            summary_saved={**summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict(),
                            'cfm':conf_mat,
                             'batch_acclist':batch_acc_list}
            torch.save(summary_saved,os.path.join(pth_path,'epoch_{}'.format(e)))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    
    # =======================================    
    
    



def run_one_epoch(model,bar,mode,optimizer=None,show_interval=10):
    confusion_mat=np.zeros((cfg.num_cls,cfg.num_cls))
    summary={"acc":[],"loss":[]}
    device=next(model.parameters()).device
    
    if mode=='train':
        model.train()
    else:
        model.eval()
    
    for i, (data_cpu,gt) in enumerate(bar):
        # === get support and query gt ====
        gt_unique=gt[:cfg.n_way*cfg.k_shot][::cfg.k_shot]
        # gt_query=gt[cfg.n_way*cfg.k_shot:]
        # =================================

        if cfg.point_support:
            x=[i.to(device) for i in data_cpu]
        else:
            x=data_cpu.to(device)
        
        if mode=='train':
            optimizer.zero_grad()
            pred,loss=model(x)
            
            #==take one step==#
            loss.backward()
            optimizer.step()
            #=================#
        else:
            with torch.no_grad():
                pred,loss=model(x)
        
        
        summary['loss']+=[loss.item()]
        
        if mode=='train':
            if i%show_interval==0:
                bar.set_description("Loss: %.3f"%(np.mean(summary['loss'])))
        else:
            batch_cfm=cal_cfm(pred,model.q_label,true_label_set=gt_unique,ncls=cfg.num_cls)
            batch_acc=np.trace(batch_cfm)/np.sum(batch_cfm)
            summary['acc'].append(batch_acc)
            if i%show_interval==0:
                bar.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))
            
            confusion_mat+=batch_cfm
    
    if mode!='train':
        summary['cfm']=confusion_mat
    
    return summary
            



if __name__=='__main__':
    main(cfg)
    
