import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import os
np.random.seed(0)
import cv2
import torchvision.transforms as transforms
# import open3d as o3d

class Toy4k_fs(Dataset):
    def __init__(self,root,split='train',fold=0,point_support=False,data_aug=True,prj_num=14):
        super().__init__()
        self.root=root
        self.split=split
        self.fold=fold
        self.point_support=point_support
        self.data_aug=data_aug
        self.prj_num=prj_num


        self.trans=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor()
                                ])



        # ====== get class name list =====
        class_name_list=sorted(os.listdir(os.path.join(self.root,'Projection_mesh_darkbg')))
        class_name_list=np.array(class_name_list)
        
        class_id=np.arange(len(class_name_list))
        mask=np.zeros(len(class_name_list))
        fold_list=[0,25,50,75,105]

        if self.split=='train':
            mask=1-mask
            mask[fold_list[self.fold]:fold_list[self.fold+1]]=0
            mask=mask.astype(np.bool)
            self.class_list=class_name_list[class_id[mask]]
        
        else:
            mask[fold_list[self.fold]:fold_list[self.fold+1]]=1
            mask=mask.astype(np.bool)
            self.class_list=class_name_list[class_id[mask]]
        # ================================
        
        # ==== get database ===
        self.support_base,self.support_label=self.get_support_data()    
        self.query_img_base, self.query_img_label= self.get_query_img_data()
        
        self.final_base, self.final_label=self.support_base+self.query_img_base,self.support_label+self.query_img_label
    
        
        # if self.point_support:
        #     self.point_base, self.point_label=self.get_point_data()
        #     self.final_database=self.query_img_base+self.point_base
        #     self.final_label=self.query_img_label+self.point_label
        # else:
        #     self.final_database=self.query_img_base
        #     self.final_label=self.query_img_label


    
    
    def get_query_img_data(self):
        img_path_list=[]
        img_label=[]
        
        img_path=os.path.join(self.root,'renders')
        for cls_ind, i in enumerate(self.class_list):
            class_path=os.path.join(img_path,i)
            sample_fold_path=[os.path.join(class_path,j,'image_output') for j in sorted(os.listdir(class_path),key=lambda x:int(x.split('_')[-1]))]

            # # ==== Except the 10 samples, rest are used for query set ====
            sample_fold_path=sample_fold_path[10:]
            # ===================================================

            for s in sample_fold_path:
                sample_img_path=[os.path.join(s,z) for z in os.listdir(s)]
                
                img_path_list+=sample_img_path
                img_label+=[cls_ind]*len(sample_img_path)
        return img_path_list, img_label
            
    

    # def translate_pointcloud(self,pointcloud):
    #     xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    #     xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
    #     translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    #     return translated_pointcloud

        
    def get_support_data(self):
        support_path_list=[]
        support_label=[]
        
        point_path=os.path.join(self.root,'Projection_mesh_darkbg')
        for cls_ind,i in enumerate(self.class_list):
            class_path=os.path.join(point_path,i)
            sample_list=sorted(os.listdir(class_path),key=lambda x: int(x.split('_')[-1]))
            sample_path_list=[os.path.join(class_path,j) for j in sample_list]
            
            if self.point_support:
                support_path_list+=sample_path_list
                support_label+=[cls_ind]*len(sample_path_list)
            else:
                for s in sample_path_list:
                    prj_img_list=os.listdir(s)
                    prj_img_path=[os.path.join(s,i) for i in prj_img_list]
                    
                    support_path_list+=prj_img_path
                    support_label+=[cls_ind]*len(prj_img_path)

        return support_path_list,support_label

    
    def pick_prj(self,prj_list):
        picked_list=[]
        picked_list+=[prj_list[-1],prj_list[-2]]
        
        side_index=np.arange(12).reshape(4,3)
        picked_index=np.random.randint(low=0,high=3,size=4)
        picked_img_index=side_index[np.arange(4),picked_index]
        
        picked_list+=[prj_list[i] for i in picked_img_index]
        return picked_list



    def __getitem__(self, index):
        data_path=self.final_base[index]
        data_label=self.final_label[index]
        
        if data_path[-3:]=='png':
            try:
                data=cv2.imread(data_path)
                data=self.trans(data)
            except:
                raise TypeError(data_path)


        else:
            prj_list=os.listdir(data_path)
            prj_list=np.random.choice(prj_list,self.prj_num,replace=False)
            prj_list=sorted(prj_list, key=lambda x:int(x.split('.')[0]))
            prj_list=[os.path.join(data_path,i) for i in prj_list]
            
            prj_img=[cv2.imread(i) for i in prj_list]
            data=torch.stack([self.trans(i) for i in prj_img],0)

        return data,data_label
        
    
    
    def __len__(self):
        return len(self.final_database)
    
    



class NShotTaskSampler(Sampler):
    def __init__(self, dataset, episode,n_way,k_shot,query_num,point_support=False,mode='random'):
        super().__init__(dataset)
        self.dataset=dataset
        self.episode=episode
        self.n_way=n_way
        self.k_shot=k_shot
        self.query_num=query_num
        self.label_set=self.get_labeset()
        self.point_support=point_support
        self.mode=mode
        
        self.support_base,self.support_label=self.dataset.support_base,self.dataset.support_label
        self.query_img_base,self.query_img_label=self.dataset.query_img_base, self.dataset.query_img_label

      
        
    def get_labeset(self):
        label_set=np.unique(self.dataset.final_label)
        return label_set


    def __iter__(self):
        for _ in range(self.episode):
            support_list=[]
            query_list=[]
            picked_cls_set=np.random.choice(self.label_set,self.n_way,replace=False)
            if self.mode=='one_angle':
                specified_angle=np.random.randint(low=0,high=14)

            for picked_cls in picked_cls_set:
                # === pick up support index ====
                support_target_index=np.where(self.support_label==picked_cls)[0]
                if self.mode=='random':
                    picked_support_index=np.random.choice(support_target_index,self.k_shot,replace=False)
                    support_list.append(picked_support_index)
                elif self.mode=='one_angle':
                    support_path=[self.dataset.final_base[i] for i in support_target_index]
                    support_path=[i for i in support_path if int(os.path.split(i)[1][:-4])==specified_angle]
                    picked_support_path=np.random.choice(support_path,self.k_shot,replace=False)
                    picked_support_index=[self.dataset.final_base.index(i) for i in picked_support_path]
                    support_list.append(picked_support_index)
                
                elif self.mode=='one_obj':
                    support_path=[self.dataset.final_base[i] for i in support_target_index]
                    obj_set=set([i.split('/')[-2].split('_')[-1] for i in support_path])
                    picked_obj=np.random.choice(list(obj_set),1,replace=False)[0]
                    legal_path=[i for i in support_path if i.split('/')[-2].split('_')[-1]==picked_obj]



                    picked_support_path=np.random.choice(legal_path,self.k_shot,replace=False)
                    picked_support_index=[self.dataset.final_base.index(i) for i in picked_support_path]
                    support_list.append(picked_support_index)



                # ==============================

                # === pick up query index ===
                img_query_index=np.where(self.query_img_label==picked_cls)[0]
                picked_query_index=np.random.choice(img_query_index,self.query_num,replace=False)
                picked_query_index=[i+len(self.support_base) for i in picked_query_index]
                query_list.append(picked_query_index)
                # ===========================
               
    
            
            s=np.concatenate(support_list)
            q=np.concatenate(query_list)
            
            '''
            For epi_index
            - it's the index used for each batch
            - the first n_way*k_shot images is the support set
            - the last n_way*query images is for the query set 
            '''    
            epi_index=np.concatenate((s,q))
            yield epi_index
    
    
    def __len__(self):
        return self.episode




def collect_fn(data):
    '''
    support set is point
    query set is image
    '''
    
    point_list=[]
    img_list=[]
    for d in data:
        if len(d[0].shape)==4:
            point_list.append(d[0])
        else:
            img_list.append(d[0])
    
    point_list=torch.stack(point_list,0)
    img_list=torch.stack(img_list,0)
    

    label_list=[d[1] for d in data]
    label_list=torch.LongTensor(label_list)

    return (point_list,img_list),label_list     


def collect_fn_onlyimg(data):
    img_list=[d[0] for d in data]
    img_list=torch.stack(img_list,0)

    label_list=[d[1] for d in data]
    label_list=torch.LongTensor(label_list)
    return img_list,label_list




def get_sets(data_path,fold=0,n_way=5,k_shot=1,query_num=10,point_support=True,prj_num=6,mode='random'):
    if point_support:
        cfn=collect_fn
    else:
        cfn=collect_fn_onlyimg
    
    train_dataset=Toy4k_fs(root=data_path,split='train',fold=fold,point_support=point_support,prj_num=prj_num)
    train_sampler=NShotTaskSampler(train_dataset,400,n_way=n_way,k_shot=k_shot,query_num=query_num,point_support=point_support,mode=mode)
    train_loader=DataLoader(dataset=train_dataset,batch_sampler=train_sampler,collate_fn=cfn)
    
    test_dataset=Toy4k_fs(root=data_path,split='test',fold=fold,point_support=point_support,prj_num=prj_num)
    test_sampler=NShotTaskSampler(test_dataset,700,n_way=n_way,k_shot=k_shot,query_num=query_num,point_support=point_support,mode=mode)
    test_loader=DataLoader(dataset=test_dataset,batch_sampler=test_sampler,collate_fn=cfn)
    
    return train_loader,test_loader

if __name__=='__main__':
    data_root='the path of TOYS4K folder'
    point_support=True
    
    
    # train_loader,test_loader=get_sets(data_path=data_root)
    # for t in train_loader:
    #     a=1
    #     break
    # for t in test_loader:
    #     a=1
    #     break
    
    
    dataset=Toy4k_fs(root=data_root,point_support=point_support,prj_num=14)
    # dataset[len(dataset)-10]
    samp=NShotTaskSampler(dataset,300,n_way=5,k_shot=1,query_num=5,point_support=point_support)
    
    
    # === get collect function ====
    if point_support:
        collect_fn=collect_fn
    else:
        collect_fn=collect_fn_onlyimg
    
    dataloader=DataLoader(dataset=dataset,batch_sampler=samp,collate_fn=collect_fn)
    for b in dataloader:
        a=1
    
    
    for batch_index in samp:
        for i in batch_index:
            path,label=dataset[i]
            print(path,label)
            print('======')