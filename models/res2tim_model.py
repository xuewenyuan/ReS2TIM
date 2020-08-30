import torch
import itertools
import numpy as np
from .base_model import BaseModel
from . import networks

class Res2TimModel(BaseModel):
    """
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--rel_classes', type=int, default=5, help='the number of relationships for classification')
        parser.add_argument('--RMSprop_momentum', type=float, default=0.2, help='momentum term of RMSprop')
        parser.add_argument('--RMSprop_decay', type=float, default=0.0005, help='weight_decay term of RMSprop')
        return parser

    def __init__(self, opt):

        self.opt = opt

        BaseModel.__init__(self, self.opt)

        self.loss_names = ['rel_cls']

        self.model_names = ['Res2Tim']

        self.netRes2Tim = networks.define_Deep_Cell_Relationship(self.opt.rel_classes, gpu_ids=self.opt.gpu_ids)

        if self.isTrain:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
            self.optimizer = torch.optim.RMSprop(self.netRes2Tim.parameters(), lr=opt.lr, 
                                                        weight_decay=opt.RMSprop_decay, momentum=opt.RMSprop_momentum)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.tb_names = input['tb_names']
        self.images = input['images'].to(self.device)
        self.spatial_feats = input['spatial_feats']
        self.bboxes = input['bboxes']
        self.org_edges = input['edges']
        self.weights = input['weights'].to(self.device)
        self.labels = input['labels'].to(self.device)

        self.cat_edges = []
        bboxes_count = 0
        for ind in range(len(self.tb_names)):
            self.cat_edges.append(self.org_edges[ind]+bboxes_count)
            bboxes_count += self.bboxes[ind].size(0)
        self.cat_edges = torch.cat(self.cat_edges, 0).to(self.device)

    def forward(self):
        self.preds = self.netRes2Tim(self.images, self.bboxes, self.spatial_feats, self.cat_edges)

    def backward(self):
        self.loss_rel_cls = self.criterion(self.preds, self.labels)
        self.loss_rel_cls = torch.mean(self.loss_rel_cls * (self.weights))
        self.loss_rel_cls.backward()


    def optimize_parameters(self):
        if not self.isTrain:
            self.isTrain = True
        self.forward()
        self.optimizer.zero_grad()
        self.backward()             
        self.optimizer.step() 

    @torch.no_grad()
    def test(self):
        if self.isTrain:
            self.isTrain = False
        self.forward()
        
        pred = torch.argmax(self.preds,dim=1).cpu().int().numpy()
        edge_count = 0
        preds = {}
        for ind in range(len(self.tb_names)):
            edge_len = self.org_edges[ind].size(0)
            rel_tuple = np.concatenate((self.org_edges[ind].int().numpy(),np.expand_dims(pred[edge_count:edge_count+edge_len],1)), 1)
            preds[self.tb_names[ind]] = {'edge_rel': rel_tuple}

        return preds
        
