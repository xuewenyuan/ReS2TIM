#encoding=utf-8
import json
import os
import cv2
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

def edge_weight(n1_bbox, n2_bbox, H, W):
    centr_x1 = (n1_bbox[0]+n1_bbox[2]+n1_bbox[4]+n1_bbox[6]) / 4
    centr_y1 = (n1_bbox[1]+n1_bbox[3]+n1_bbox[5]+n1_bbox[7]) / 4
    centr_x2 = (n2_bbox[0]+n2_bbox[2]+n2_bbox[4]+n2_bbox[6]) / 4
    centr_y2 = (n2_bbox[1]+n2_bbox[3]+n2_bbox[5]+n2_bbox[7]) / 4
    edge_lambda = np.exp(-(np.square((centr_x1-centr_x2)*3.0/W)+np.square((centr_y1-centr_y2)*3.0/H)))+2

    return edge_lambda

def edge_rel(cell1_lloc,cell2_lloc):
    # cell1_lloc : [start_row, end_row, start_col, end_col]
    # cell2_lloc : [start_row, end_row, start_col, end_col]
    rel_dict = {'none':0,'left':1,'right':2,'up':3,'down':4}
    if not (cell2_lloc[0] > cell1_lloc[1]) and not (cell2_lloc[1] < cell1_lloc[0]):
        if cell2_lloc[3] == cell1_lloc[2]-1: return rel_dict['left']
        elif cell2_lloc[2] == cell1_lloc[3]+1: return rel_dict['right']
    if not (cell2_lloc[2] > cell1_lloc[3]) and not (cell2_lloc[3] < cell1_lloc[2]):
        if cell2_lloc[1] == cell1_lloc[0]-1: return rel_dict['up']
        elif cell2_lloc[0] == cell1_lloc[1]+1: return rel_dict['down']
    return rel_dict['none']

def graph_gen(page_img, table_anno):
    node_dict = {'ind':[],'x1':[],'y1':[],'x2':[],'y2':[],'x3':[],'y3':[],'x4':[],'y4':[]}
    edge_dict = {'n1':[], 'n2':[], 'weight': [], 'rel': []} # feature: directed euclidean distance along x and y axis, respectively.
    target_dict = {'ind':[], 'start-row':[], 'end-row':[], 'start-col':[], 'end-col':[]}
    new_anno = []

    for cell_id, anno in enumerate(table_anno):
        x1, y1 = int(anno['x']), int(anno['y'])
        x2, y2 = int(anno['x'])+int(anno['width']), int(anno['y'])
        x3, y3 = int(anno['x'])+int(anno['width']), int(anno['y'])+int(anno['height'])
        x4, y4 = int(anno['x']), int(anno['y'])+int(anno['height'])

        node_dict['ind'].append(cell_id)
        node_dict['x1'].append(x1)
        node_dict['y1'].append(y1)
        node_dict['x2'].append(x2)
        node_dict['y2'].append(y2)
        node_dict['x3'].append(x3)
        node_dict['y3'].append(y3)
        node_dict['x4'].append(x4)
        node_dict['y4'].append(y4)

        target_dict['ind'].append(cell_id)
        target_dict['start-row'].append(int(anno['cell_row'])-1)
        target_dict['end-row'].append(int(anno['cell_row'])-1)
        target_dict['start-col'].append(int(anno['cell_col'])-1)
        target_dict['end-col'].append(int(anno['cell_col'])-1)

        cell_dict = dict()
        cell_dict['cell_id'] = cell_id
        cell_dict['bbox'] = [x1,y1,x2,y2,x3,y3,x4,y4]
        cell_dict['lloc'] = [int(anno['cell_row'])-1,int(anno['cell_row'])-1,int(anno['cell_col'])-1,int(anno['cell_col'])-1]

        new_anno.append(cell_dict)

    x1mi, y1mi = min(node_dict['x1']), min(node_dict['y1'])
    x2ma, y2ma = max(node_dict['x3']), max(node_dict['y3'])
    table_img = page_img[y1mi-10:y2ma+10,x1mi-10:x2ma+10]
    h, w, _ = table_img.shape

    for key_i in node_dict.keys():
        if 'x' in key_i:
            node_dict[key_i] = np.array(node_dict[key_i],dtype='int')-x1mi+10
            node_dict[key_i] = node_dict[key_i].tolist()
        if 'y' in key_i:
            node_dict[key_i] = np.array(node_dict[key_i],dtype='int')-y1mi+10
            node_dict[key_i] = node_dict[key_i].tolist()

    for ci in new_anno:
        np_temp = np.array(ci['bbox'],dtype='int')
        np_temp[[0,2,4,6]] = np_temp[[0,2,4,6]]-x1mi+10
        np_temp[[1,3,5,7]] = np_temp[[1,3,5,7]]-y1mi+10
        ci['bbox'] = np_temp.tolist()
    
    for n1_ind in node_dict['ind']:
        for n2_ind in node_dict['ind']:
            if n1_ind == n2_ind: continue
            n1_bbox = [node_dict['x1'][n1_ind], node_dict['y1'][n1_ind], node_dict['x2'][n1_ind], node_dict['y2'][n1_ind],\
                    node_dict['x3'][n1_ind], node_dict['y3'][n1_ind], node_dict['x4'][n1_ind], node_dict['y4'][n1_ind]]
            n2_bbox = [node_dict['x1'][n2_ind], node_dict['y1'][n2_ind], node_dict['x2'][n2_ind], node_dict['y2'][n2_ind],\
                    node_dict['x3'][n2_ind], node_dict['y3'][n2_ind], node_dict['x4'][n2_ind], node_dict['y4'][n2_ind]]
            n1_lloc = [target_dict['start-row'][n1_ind], target_dict['end-row'][n1_ind], \
                    target_dict['start-col'][n1_ind], target_dict['end-col'][n1_ind]]
            n2_lloc = [target_dict['start-row'][n2_ind], target_dict['end-row'][n2_ind], \
                    target_dict['start-col'][n2_ind], target_dict['end-col'][n2_ind]]
            edge_lambda = edge_weight(n1_bbox, n2_bbox, h, w)
            edge_relation = edge_rel(n1_lloc,n2_lloc)
            edge_dict['n1'].append(n1_ind)
            edge_dict['n2'].append(n2_ind)
            edge_dict['weight'].append(edge_lambda)
            edge_dict['rel'].append(edge_relation)
    
    nodes   = pd.DataFrame(node_dict)
    edges   = pd.DataFrame(edge_dict)
    targets = pd.DataFrame(target_dict)

    return table_img, new_anno, nodes, edges, targets

if __name__ == '__main__':
    src_label = json.load(open('./labels_src.json','r'))
    train_set = [ line.strip() for line in open('./src_set/trainval.txt','r').readlines()]
    test_set  = [ line.strip() for line in open('./src_set/test.txt','r').readlines()]

    graph_folders = {'node': './graph_node', 'edge': './graph_edge', 'target': './graph_target'}
    for name in graph_folders.keys():
        if not os.path.exists(graph_folders[name]):
            os.mkdir(graph_folders[name])
    table_folders = {'image': './image', 'gt': './gt'}
    for name in table_folders.keys():
        if not os.path.exists(table_folders[name]):
            os.mkdir(table_folders[name])

    new_train = [] 
    new_test = []
    max_col = 0
    max_row = 0

    for page in tqdm(src_label):
        img_id = page['filename'].split('.')[0]
        #img_path = os.path.join('/data/xuewenyuan/data/cmdd/src_image',page['filename'])
        img_path = os.path.join('./src_image',page['filename'])
        page_img = cv2.imread(img_path)

        for tb_id in ['1', '2']:
            annos = [anno for anno in page['annotations'] if anno['table_no'] == tb_id]
            table_img, cell_annos, nodes, edges, targets = graph_gen(page_img, annos)

            max_row = max(max_row, max(targets['end-row'].values))
            max_col = max(max_col, max(targets['end-col'].values))

            table_name = img_id+'_t'+tb_id

            table_img_path = os.path.join(os.getcwd(),'image',table_name+'.jpg')
            cv2.imwrite(table_img_path, table_img)

            table_gt = {'table_ind':table_name, 'image_path': table_img_path, 'cells_anno': cell_annos}
            anno_file = './gt/'+table_name+'.pkl'
            with open(anno_file, 'wb') as fin:
                pickle.dump(table_gt, fin, pickle.HIGHEST_PROTOCOL)

            node_file = os.path.join(graph_folders['node'], table_name + '_node' +'.csv')
            edge_file = os.path.join(graph_folders['edge'], table_name + '_edge' +'.csv')
            target_file = os.path.join(graph_folders['target'], table_name + '_target' +'.csv')
            nodes.to_csv(node_file)
            edges.to_csv(edge_file)
            targets.to_csv(target_file)

            curren_path = os.getcwd()
            set_file = os.path.join(curren_path, anno_file[2:]) + ' ' + os.path.join(curren_path, node_file[2:]) + ' ' +\
                        os.path.join(curren_path, edge_file[2:]) + ' ' + os.path.join(curren_path, target_file[2:]) + '\n'

            if img_id in train_set:
                new_train.append(set_file)
            elif img_id in test_set:
                new_test.append(set_file)

    with open('./train.txt','w',encoding='UTF-8') as fin:
        fin.writelines(new_train)
    with open('./test.txt','w',encoding='UTF-8') as fin:
        fin.writelines(new_test)

    print('train length: {:.0f}'.format(len(new_train)))
    print('test length: {:.0f}'.format(len(new_test)))
    print('max row: {:.0f}'.format(max_row))
    print('max col: {:.0f}'.format(max_col))
