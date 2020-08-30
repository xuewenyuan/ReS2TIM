import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from util.table_structure_infer import tb_struc_infer

class Evaluator(object):
    def __init__(self, opt):
        self.data_list = open(os.path.join(opt.dataroot, opt.phase+'.txt'), 'r', encoding='UTF-8').readlines()
        if opt.phase == 'test' and opt.max_test_size != float("inf"):
            self.data_list = self.data_list[0:opt.max_test_size]
        self.gt_dict, self.gt_box_sum, self.gt_fgRel_sum, self.gt_bgRel_sum, self.pred_dict = self.create_gt_dict(self.data_list)

    def reset(self):
        self.matched_fgRel = 0.0
        self.matched_bgRel = 0.0

        self.pred_fgRel_sum = 0.0
        self.pred_bgRel_sum = 0.0

        self.pred_lloc_sum  = 0.0
        self.pred_rowSt_sum = 0.0
        self.pred_rowEd_sum = 0.0
        self.pred_colSt_sum = 0.0
        self.pred_colEd_sum = 0.0

    def summary(self, eval_mode = 'edge_rel | lloc'):
        metric = ''
        select_metric = 0.0
        if 'edge_rel' in eval_mode:
            print('Evaluation for neighbor relationship detection')
            for tb_name in tqdm(self.gt_dict.keys()):
                pred_fg_ind = np.where(self.pred_dict[tb_name]['edge_rel']>0)
                pred_bg_ind = np.where(self.pred_dict[tb_name]['edge_rel']==0)
                self.pred_fgRel_sum += pred_fg_ind[0].shape[0]
                self.pred_bgRel_sum += pred_bg_ind[0].shape[0]
                self.matched_fgRel += np.where(self.pred_dict[tb_name]['edge_rel'][pred_fg_ind]==self.gt_dict[tb_name]['edge_rel'][pred_fg_ind])[0].shape[0]
                self.matched_bgRel += np.where(self.pred_dict[tb_name]['edge_rel'][pred_bg_ind]==self.gt_dict[tb_name]['edge_rel'][pred_bg_ind])[0].shape[0]
            fg_precision = self.matched_fgRel / self.pred_fgRel_sum if self.pred_fgRel_sum!=0 else 0
            fg_recall = self.matched_fgRel / self.gt_fgRel_sum if self.gt_fgRel_sum!=0 else 0
            bg_precision = self.matched_bgRel / self.pred_bgRel_sum if self.pred_bgRel_sum!=0 else 0
            bg_recall = self.matched_bgRel / self.gt_bgRel_sum if self.gt_bgRel_sum!=0 else 0
            fg_f1 = 2*(fg_precision*fg_recall)/(fg_precision+fg_recall) if fg_precision+fg_recall != 0 else 0
            select_metric += fg_f1
            metric += 'Fg_Precision: {:.4f}, Fg_Recall: {:.4f}, Bg_Precision: {:.4f}, Bg_Recall: {:.4f}\n'.format(\
                        fg_precision, fg_recall, bg_precision, bg_recall)
        if 'lloc' in eval_mode:
            print('Evaluation for cell location inference')
            for tb_name in tqdm(self.gt_dict.keys()):
                self.pred_dict[tb_name]['lloc'] = tb_struc_infer(self.pred_dict[tb_name]['edge_rel'], self.gt_dict[tb_name]['bbox'])
                self.pred_rowSt_sum += np.where(self.pred_dict[tb_name]['lloc'][:,0] == self.gt_dict[tb_name]['lloc'][:,0])[0].shape[0]
                self.pred_rowEd_sum += np.where(self.pred_dict[tb_name]['lloc'][:,1] == self.gt_dict[tb_name]['lloc'][:,1])[0].shape[0]
                self.pred_colSt_sum += np.where(self.pred_dict[tb_name]['lloc'][:,2] == self.gt_dict[tb_name]['lloc'][:,2])[0].shape[0]
                self.pred_colEd_sum += np.where(self.pred_dict[tb_name]['lloc'][:,3] == self.gt_dict[tb_name]['lloc'][:,3])[0].shape[0]
                self.pred_lloc_sum  += np.where(np.sum(self.pred_dict[tb_name]['lloc']-self.gt_dict[tb_name]['lloc'],1)==0)[0].shape[0]
                
            acc_rowSt = self.pred_rowSt_sum / self.gt_box_sum if self.gt_box_sum!=0 else 0
            acc_rowEd = self.pred_rowEd_sum / self.gt_box_sum if self.gt_box_sum!=0 else 0
            acc_colSt = self.pred_colSt_sum / self.gt_box_sum if self.gt_box_sum!=0 else 0
            acc_colEd = self.pred_colEd_sum / self.gt_box_sum if self.gt_box_sum!=0 else 0
            acc_lloc  = self.pred_lloc_sum  / self.gt_box_sum if self.gt_box_sum!=0 else 0
            metric += 'Acc_lloc: {:.4f}, Acc_rowSt: {:.4f}, Acc_rowEd: {:.4f}, Acc_colSt: {:.4f}, Acc_colEd: {:.4f}'.format(\
                        acc_lloc, acc_rowSt, acc_rowEd, acc_colSt, acc_colEd)
        return metric, select_metric

    def update(self, preds):
        # pred: {'table_name':{'edge_rel':Array[L, 3],'lloc':Array[L, 4]},...}
        for tb_name in preds.keys():
            if 'edge_rel' in preds[tb_name].keys():
                edge_ind = (preds[tb_name]['edge_rel'][:,0].astype(np.int32), preds[tb_name]['edge_rel'][:,1].astype(np.int32))
                self.pred_dict[tb_name]['edge_rel'][edge_ind] =  preds[tb_name]['edge_rel'][:,2]


    def create_gt_dict(self, data_list):
        gt_dict = {} #'img_name': bboxes[], llocs[], pair_rel[]
        pred_dict = {} #'img_name': pair_rel[]
        gt_size = len(data_list)
        gt_box_sum = 0.0
        gt_fgrel_sum = 0.0
        gt_bgrel_sum = 0.0
        for ind in range(gt_size):
            table_pkl, node_csv, edge_csv, _= data_list[ind].strip().split()
            table_anno = pickle.load(open(table_pkl, 'rb'))
            edges = pd.read_csv(edge_csv).values
            table_name = os.path.split(table_pkl)[1].replace('.pkl','')
            cells_bbox = np.array([cell_i['bbox'] for cell_i in table_anno['cells_anno']], dtype=np.float64)
            cells_lloc = np.array([cell_i['lloc'] for cell_i in table_anno['cells_anno']], dtype=np.int32)
            edges_rel = -1*np.ones((cells_bbox.shape[0],cells_bbox.shape[0]))
            edges_rel[edges[:,1].astype(np.int32), edges[:,2].astype(np.int32)] = edges[:,4]
            gt_dict[table_name] = {'bbox': cells_bbox, 'lloc': cells_lloc, 'edge_rel': edges_rel}
            pred_dict[table_name] = {'edge_rel': -1*np.ones(edges_rel.shape)}
            gt_box_sum += len(cells_bbox)
            gt_fgrel_sum += np.where(edges_rel>0)[0].shape[0]
            gt_bgrel_sum += len(np.where(edges_rel==0)[0])
        return gt_dict, gt_box_sum, gt_fgrel_sum, gt_bgrel_sum, pred_dict