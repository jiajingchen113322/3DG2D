# Letting 3D Guide the Way: 3D Guided 2D Few-Shot Image Classification (WACV 2024)
This is official pytorch implementation of WACV 2024 paper 3DG2D paper. 

For more detial,please check our paper [here](https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Letting_3D_Guide_the_Way_3D_Guided_2D_Few-Shot_Image_WACV_2024_paper.pdf). The pipeline of our proposed method is shown below: ![avatar](https://github.com/jiajingchen113322/3DG2D/blob/official/Img/flow.PNG)

## Quick Learning for Your Convenience
If you don't want to spend too much time on preparing the data, for your convenience, you could go to **./Model/Img_few_shot_prj.py** directly for quick learning of our method. I prepared a toy example in the main function there. By setting fs=='AINet' in the class *ThreeD_Support_Net*, you use AINet as few-shot head for the prediction. 

Besides, you could also got to **./Model/Head/AINet.py** to check our proposed AINet. I prepare a toy exmpale in the main function there for you quick learning.

## Data Preparation
All the query images are from [1], and support images are generated by us from 3D mesh data. We introduce how we obtain the data below:

### ModelNet40
For query images, please go to [here](https://github.com/rehg-lab/lowshot-shapebias) to download ModelNet40-LS, and save save the query image into ***./ModelNet40-LS/renders***

For support projections generated by us, you could download directly from [here](https://drive.google.com/file/d/1U6VLcY-kEhQhkWI3fkA-ObJOlC88Ytfa/view?usp=share_link). Unzip it and save the data to ***./ModelNet40-LS*** 

### ShapeNet
For the query images, please go to [here](https://github.com/rehg-lab/lowshot-shapebias) to download ShapeNet55-LS, and save it to ***./ShapeNet55-LS/renders***

For the support projections generated by us, download it [here](https://drive.google.com/file/d/14LChsEJX4hJYx-lo9nlpMg45pfoo_zUD/view?usp=share_link), unzip it and save it ***./ShapeNet55-LS***

### Toy4K 
Similar ot how to prepare ModelNet40 and ShapeNet dataset, download the query images from [here](https://github.com/rehg-lab/lowshot-shapebias), and save it to ***./TOYS4K/renders***.

As for support projections prepared by us, download it from [here](https://drive.google.com/file/d/15FACStHmzJ8Q-DhBk4GSAtiaoUfvWCwM/view?usp=share_link), unzip it and save it to ***./TOYS4K***


## Pretraining
Download the pretrain backbone from this [link](https://drive.google.com/file/d/1JsCcquiOFdGzOpNtQ5FQdY1pQZN7IrYy/view?usp=share_link), unzip it, and place it to ***path to pretrained model***

Of course, you could perform pretraining by yourself with the command below:

```python Pretrain\main_pretrain.py --exp_name $Your Exp Name$ --dataset $Dataset used for pretraining$ --fold $fold number$ --data_path $The path you save data$```


## Few-shot Classification by 3DG2D method

Key arguemnts in main.py is explained below
* --exp_name: Experiment name defined by you. The experiment folder with this name will be created in Exp Folder

* --dataset: which dataset you choose to use

* --epochs: number of training epoch

* --prj_num: projection number generated from each 3D mesh sample

* --pretrain: set true if using pretarined backbone

* --point_support: set true if using 3DG2D method for the training, otherwise, using traditional few-shot classification method for the training.

* --fs_head: few-shot head used for the experiment. All few-shot head code are in the ./Model/Head.

* --data_path: : the path of the dataset folder. For example, if you select ModelNet40-LS as experiment, you should fill the path of ModelNet40-LS folder here.

Run the following code for training:

```python main.py --exp_name $Experiment name given by you$ --pretrain_path $$The pretrained backbone path$ --dataset $dataset used for training$ --data_path $path to your dataset$```

## Reference 
[1] Stojanov, Stefan, Anh Thai, and James M. Rehg. "Using shape to categorize: Low-shot learning with an explicit shape bias." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.