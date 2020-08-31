# ReS2TIM: Reconstruct Syntactic Structures from Table Images

[![conferrence](https://img.shields.io/badge/Conference-ICDAR2019-brightgreen)](https://ieeexplore.ieee.org/abstract/document/8978027?casa_token=urFh-W9rfh0AAAAA:45S5Fd3PYkfzJ62OOoeu83CMveqpp2e_-deDyeFMZqWf2CvGOtQpMZQxVhfRyTnhGZ6triKb) ![](https://img.shields.io/badge/Cleansed%20Label-CMDD%20%7C%20ICDAR13Table-yellowgreen) ![](https://img.shields.io/badge/Pretrained%20Model-Yes-orange) 

> Xue, Wenyuan, Qingyong Li, and Dacheng Tao. "ReS2TIM: Reconstruct Syntactic Structures from Table Images." 2019 International Conference on Document Analysis and Recognition (ICDAR). IEEE, 2019.

## Abstract
Tables often represent densely packed but structured data. Understanding table semantics is vital for effective information retrieval and data mining. Unlike web tables, whose semantics are readable directly from markup language and contents, the full analysis of tables published as images requires the conversion of discrete data into structured information. This paper presents a novel framework to convert a table image into its syntactic representation through the relationships between its cells. In order to reconstruct the syntactic structures of a table, we build a cell relationship network to predict the neighbors of each cell in four directions. During the training stage, a distance-based sample weight is proposed to handle the class imbalance problem. According to the detected relationships, the table is represented by a weighted graph that is then employed to infer the basic syntactic table structure. Experimental evaluation of the proposed framework using two datasets demonstrates the effectiveness of our model for cell relationship detection and table structure inference.
# Note
- This is a reimplementation version based on PyTorch 1.5 and Python 3.7.
- We correct some wrong labels in the CMDD dataset and ICDAR2013Table dataset and present the new results and pretrained model. In ICDAR2013Table dataset, the table on the second page of 'us-035a.pdf' is splited into three regions. We combine them together and get a complete table. This is different from the original paper, in which the three regions are treated as diffrent tables.
- The RoI_pool is replace by the RoI_align.
# Getting Started
## Requirements
Create the environment from the environment.yml file `conda env create --file environment.yaml` or install the software needed in your environment independently.
```
dependencies:
  - python=3.7
  - torchvision==0.6.0
  - pytorch==1.5.0
  - pip:
    - dominate==2.5.2
    - opencv-python==4.4.0.42
    - pandas==1.1.1
    - tqdm==4.48.2
    - scipy==0.5.2
    - visdom==0.1.8
```
## Datasets Preparation
- Download the CMDD dataset from [Google Dive](https://drive.google.com/file/d/1OyMbmwVC1e1fx4P5WPLGmaVc1oXs55_6/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1HMiJ1TjTNOAN3k3qVuoYpg)(pwd:rmmi).
- Download the ICDAR2013Table dataset from [Google Dive](https://drive.google.com/file/d/1LkrDROegMqG2E41y3kpiAvJ4C5jnMbtg/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1RL7_skEMgVPtTQrvJNy0jw)(pwd:cyvn).
- Put cmdd.tar.gz and icdar13table.tar.gz in "./datasets/".
- Extract *.tar.gz files.
```
cd ./datasets
tar -zxvf cmdd.tar.gz
tar -zxvf icdar13table.tar.gz
## The './datasets/' folder should look like:
- datasets/
  - cmdd/
    - src_image/
    - src_set/
    - labels_src.json
    - prepare.py
  - icdar13table/
    - eu-dataset/
    - us-dataset/
    - eu-us-dataset/
    - src_page_image/
    - src_set/
    - prepare.py
```
- Run 'prepare.py'.
```
cd ./datasets/cmdd
python prepare.py
cd ../icdar13table
python prepare.py
cd ..
rm cmdd.tar.gz
rm icdar13table.tar.gz
``` 
## Training and evaluation
1. Train and fine tune.
```
# train on the cmdd dataset
python train.py --dataroot ./datasets/cmdd --gpu_ids 2 --model res2tim --dataset_mode cell_rel --lr 0.0005 --pair_batch 10000 --niter 5 --niter_decay 95 --use_mask --name res2tim_cmdd

# Copy the best model on CMDD to the folder of icdar13table for initialization
mkdir ./checkpoints/icdar13table
cp ./checkpoints/cmdd/best_net_Res2Tim.pth ./checkpoints/icdar13table/best_net_Res2Tim.pth

# train on the icdar13table dataset based on the CMDD pretrained model
python train.py --dataroot ./datasets/icdar13table --gpu_ids 2 --model res2tim --dataset_mode cell_rel --lr 0.0005 --pair_batch 10000 --niter 5 --niter_decay 95 --use_mask --name res2tim_icdar13table --continue_train --epoch prt
```
2. Evaluation for neighbor relationship detection and cell location inference. Use your training models, or download our pretrained models and put them under './checkpoints/res2tim_cmdd/' and './checkpoints/res2tim_icdar13table/', respectively. CMDD pretrained model: [Google Drive](https://drive.google.com/file/d/1fEE-05_EAzbbRnlF6mMbxhFH9kjgnkeZ/view?usp=sharing), [百度网盘(b7pt)](https://pan.baidu.com/s/1M32SW4fwAHz6yV9fpXsMTQ). ICDAR13Table pretrained model: [Google Drive](https://drive.google.com/file/d/1uWzFy8KpeTPObqT-HAiD8o6S7NZsVq2e/view?usp=sharing), [百度网盘(rmp8)](https://pan.baidu.com/s/1_sMddbJE1QbSMvI7gw6Jkw).
```
# CMDD 
python test.py --dataroot ./datasets/cmdd --gpu_ids 5 --model res2tim --dataset_mode cell_rel --pair_batch 10000 --use_mask --name res2tim_cmdd --epoch best

# ICDAR13TABLE
python test.py --dataroot ./datasets/icdar13table --gpu_ids 5 --model res2tim --dataset_mode cell_rel --pair_batch 10000 --use_mask --name res2tim_icdar13table --epoch best
```
3. Key options.
  - Only 1 GPU is supported for training and evaluation because of the RoI_align. If multiple GPUs are needed for a huge datset, you can change some codes to realize distributed training.
  - GPU memory requirements rise dramatically as the number of table cells increases. So, we split the pair realtions in a table into multiple batches. For each iteration during training and testing, only one table image is sent to the model and the number of pair relations is not larger than '--pair_batch'.

## Experiment Results
1. Results of neighbor relationship detection
<table>
    <tr>
        <td> </td>
        <td colspan="2">CMDD</td>
        <td colspan="2">ICDAR 2013 Dataset</td>
    <tr>
    <tr>
        <td> </td>
        <td>Precision</td>
        <td>Recall</td>
        <td>Precision</td>
        <td>Recall</td>
    <tr>
    <tr>
        <td>The paper reports</td>
        <td>0.999</td>
        <td>0.997</td>
        <td>0.926</td>
        <td>0.447</td>
    <tr>
    <tr>
        <td>This implementation</td>
        <td>0.999</td>
        <td>0.996</td>
        <td>0.811</td>
        <td>0.771</td>
    <tr>
</table>
2. Results of cell location inference
<table>
    <tr>
        <td colspan="6">CMDD</td>
    <tr>
    <tr>
        <td> </td>
        <td>cell_loc</td>
        <td>row1</td>
        <td>row2</td>
        <td>col1</td>
        <td>col2</td>
    <tr>
    <tr>
        <td>The paper reports</td>
        <td>0.999</td>
        <td>0.999</td>
        <td>0.999</td>
        <td>0.999</td>
        <td>0.999</td>
    <tr>
    <tr>
        <td>This implementation</td>
        <td>0.996</td>
        <td>0.999</td>
        <td>0.997</td>
        <td>0.999</td>
        <td>0.999</td>
    <tr>
</table>

<table>
    <tr>
        <td colspan="6">ICDAR 2013 Dataset</td>
    <tr>
    <tr>
        <td> </td>
        <td>cell_loc</td>
        <td>row1</td>
        <td>row2</td>
        <td>col1</td>
        <td>col2</td>
    <tr>
    <tr>
        <td>The paper reports</td>
        <td>0.015</td>
        <td>0.053</td>
        <td>0.064</td>
        <td>0.166</td>
        <td>0.163</td>
    <tr>
    <tr>
        <td>This implementation</td>
        <td>0.053</td>
        <td>0.167</td>
        <td>0.127</td>
        <td>0.430</td>
        <td>0.364</td>
    <tr>
</table>

## Custom dataset Preparation
Refer to `./datasets/cmdd/prepare.py` and `./datasets/icdar13table/prepare.py` for you own dataset preparation.

## Citation
Please consider citing this work in your publications if it helps your research.  
```
@inproceedings{xue2019res2tim,  
  title={ReS2TIM: Reconstruct Syntactic Structures from Table Images},  
  author={Xue, Wenyuan and Li, Qingyong and Tao, Dacheng},  
  booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},  
  pages={749--755},  
  year={2019},  
  organization={IEEE}  
}  
```
