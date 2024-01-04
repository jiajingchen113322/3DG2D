import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import torchvision.transforms as transforms



np.random.seed()

class Toy4K_pretrain(Dataset):
    def __init__(self,root,fold=0):
        super().__init__()
        self.root=os.path.join(root,'Projection_mesh_darkbg')
        self.fold=fold
        
        color_jiter=transforms.ColorJitter(0.8 , 0.8, 0.8, 0.2)

        self.trans0=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor()
                                ])

        self.trans1=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomRotation(90,fill=[79,79,47]),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([color_jiter], p=0.8),
                        transforms.Resize(224),
                        transforms.ToTensor()
                                ])
        
        
        
        # ===== get class name list ======
        fold_list=[0,25,50,75,105]

        class_name_list=sorted(os.listdir(self.root))
        class_name_list=np.array(class_name_list)
        
        class_id=np.arange(len(class_name_list))
        mask=np.zeros(len(class_name_list))
        mask=1-mask
        # mask[fold_list[self.fold*10:(self.fold+1)*10]=0
        
        start=int(fold_list[self.fold])
        end=int(fold_list[self.fold+1])
        mask[start:end]=0

        mask=mask.astype(np.bool)
        self.class_list=class_name_list[class_id[mask]]
    
        self.set_list=self.get_set_list()
        
    def get_set_list(self):
        set_list=[]
        
        for cls in self.class_list:
            cls_path=os.path.join(self.root,cls)
            sample_list=os.listdir(cls_path)
            for sample in sample_list:
                sample_path=os.path.join(cls_path,sample)
                img_list=os.listdir(sample_path)
                img_list=sorted(img_list,key=lambda x:int(x.split('.')[0]))
                img_path_list=[os.path.join(sample_path,i) for i in img_list]
                
                set_list.append(img_path_list[:3])
                set_list.append(img_path_list[3:6])
                set_list.append(img_path_list[6:9])
                set_list.append(img_path_list[9:12])
                set_list.append(img_path_list[12:])
                
        return set_list

    
    def __getitem__(self,index):
        img_set=self.set_list[index]
        img_index=np.random.choice(len(img_set),2)
        picked_img_pair=[img_set[i] for i in img_index]
        img_list=[self.trans0(cv2.imread(picked_img_pair[0])), self.trans1(cv2.imread(picked_img_pair[1]))]
        
        # cv2.imshow('1',np.array(img_list[0]))
        # cv2.imshow('2',np.array(img_list[1]))
        # cv2.waitKey()
        
        return img_list[0],img_list[1]
    
    
    
    def __len__(self):
        return len(self.set_list)



def get_sets(data_path,fold=0):
    train_dataset=Toy4K_pretrain(root=data_path,fold=fold)
    train_loader=DataLoader(dataset=train_dataset,batch_size=60,shuffle=True)
    return train_loader



if __name__=='__main__':
    data_root='/data1/jiajing/dataset/see_like_human/TOYS4K'
    data_loader=get_sets(data_root)
    for (x,y) in data_loader:
        a=1

    dataset=Toy4K_pretrain(data_root)
    for i in range(len(dataset)):
        dataset[i]