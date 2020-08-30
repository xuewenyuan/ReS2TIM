import os
import torch
import pickle
import pandas as pd
import numpy as np
from data.base_dataset import BaseDataset
from PIL import Image
from PIL import ImageFile
import cv2
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CellRelDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--pair_batch', type=int, default=10000, help='the batch size of cell pairs')
        parser.add_argument('--use_mask', action='store_true', help='if use mask instead of original image')
        return parser

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.min_size = opt.min_size
        self.max_size = opt.max_size
        self.use_mask = opt.use_mask
        
        self.train_file = open(os.path.join(opt.dataroot, opt.phase+'.txt'), 'r')
        self.table_list = self.train_file.readlines()
        self.pair_list = self.seprate_table(self.table_list, opt.pair_batch)
        self.pair_size = len(self.pair_list)

    def __getitem__(self, index):
        tb_name = self.pair_list[index]['tb_name']

        tb_img = Image.open(self.pair_list[index]['img_path']).convert('RGB')
        tb_w, tb_h = tb_img.size
        im_scale, pdl, pdt, pdr, pdd, img_transform = self.get_transform(tb_w, tb_h)

        bboxes = self.pair_list[index]['bboxes']

        if self.use_mask:
            tb_img = self.convert_mask(tb_w, tb_h, bboxes)
        
        tb_img = img_transform(tb_img)
        _, H, W = tb_img.shape
        bboxes = bboxes*im_scale+np.array([pdl,pdt]*2)
        bboxes = torch.from_numpy(bboxes).to(torch.float32)

        spatial_feat = bboxes/torch.as_tensor([W,H]*2, dtype=torch.float32)
        
        edges = torch.from_numpy(self.pair_list[index]['edges']).to(torch.int64)
        weights = torch.from_numpy(self.pair_list[index]['weights']).to(torch.float32)
        rels = torch.from_numpy(self.pair_list[index]['rels']).to(torch.int64)

        return {'tb_name': tb_name, 'image': tb_img, 'bboxes': bboxes, 'spatial_feat': spatial_feat,\
                'edges': edges, 'weights': weights, 'rels': rels}


    def __len__(self):
        return len(self.pair_list)

    def collate_fn(self, batch):
        batch_size = len(batch)
        tb_names = []
        images = []
        bboxes = []
        spatial_feats = []
        edges = []
        weights = []
        rels = []
        #bboxes_count = 0
        for id in range(batch_size):
            tb_names.append(batch[id]['tb_name'])
            images.append(batch[id]['image'])
            bboxes.append(batch[id]['bboxes'])
            spatial_feats.append(batch[id]['spatial_feat'])
            edges.append(batch[id]['edges'])
            weights.append(batch[id]['weights'])
            rels.append(batch[id]['rels'])
            #bboxes_count += batch[id]['bboxes'].size(0)

        images = torch.stack(images, 0)
        spatial_feats = torch.cat(spatial_feats, 0)
        #edges = torch.cat(edges, 0)
        weights = torch.cat(weights,0)
        rels = torch.cat(rels,0)

        return {'tb_names':tb_names ,'images':images, 'spatial_feats': spatial_feats, 'labels': rels, \
                'bboxes': bboxes, 'edges': edges, 'weights':weights}

    def seprate_table(self, org_datasets_list, pair_batch):
        new_datasets_list = [] 
        # {'img_path': str, 'bboxes': array(x1,y1,x2,y2), 'edges': array(n1,n2),'weights': array(weight),\
        # 'spatial_feat': array(), 'rels': array(relationship)}
        for table_i in org_datasets_list:
            table_pkl, node_csv, edge_csv, _= table_i.strip().split()
            tb_dict = pickle.load(open(table_pkl, 'rb'))
            nodes = pd.read_csv(node_csv).values
            edges = pd.read_csv(edge_csv).values
            edges = np.random.permutation(edges)
            count = 0
            while count < edges.shape[0]:
                split = dict()
                split['tb_name'] = os.path.split(table_pkl)[1].replace('.pkl','')
                split['img_path'] = tb_dict['image_path']
                split['bboxes'] = nodes[:,[3,4,7,8]].astype(np.float64)
                split['edges'] = edges[count:min(count+pair_batch,edges.shape[0]), 1:3]
                split['weights'] = edges[count:min(count+pair_batch,edges.shape[0]), 3]
                split['rels'] = edges[count:min(count+pair_batch,edges.shape[0]), 4]
                count += pair_batch
                new_datasets_list.append(split)
        return new_datasets_list

    def convert_mask(self, tb_w, tb_h, bboxes):
        mask = np.ones((tb_h,tb_w,3))
        for ind in range(bboxes.shape[0]):
            bb = bboxes[ind].astype(np.int32)
            mask[bb[1]:bb[3],bb[0]:bb[2],:] = 255
        return Image.fromarray(mask.astype('uint8'))

    def get_transform(self, tb_w, tb_h):
        im_min_size = min(tb_w, tb_h)
        im_max_size = max(tb_w, tb_h)
        im_scale = float(self.min_size) / float(im_min_size)
        if int(im_scale*im_max_size) > self.max_size:
            im_scale = float(self.max_size) / float(im_max_size)
        rew = int(tb_w * im_scale)
        reh = int(tb_h * im_scale)
        pdl, pdt = ((self.max_size-rew)//2, (self.max_size-reh)//2) 
        pdr, pdd = (self.max_size-rew-pdl, self.max_size-reh-pdt)

        img_transform = []
        img_transform.append(transforms.Resize((reh,rew), Image.BICUBIC))
        #img_transform.append(transforms.Pad((pdl, pdt, pdr, pdd)))
        img_transform.append(transforms.ToTensor())
        img_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        #return im_scale, pdl, pdt, pdr, pdd, transforms.Compose(img_transform)
        return im_scale, 0, 0, 0, 0, transforms.Compose(img_transform)