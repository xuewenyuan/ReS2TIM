import numpy as np
from .shortest_path import ShortestDis

def tb_struc_infer(rel_mat, cells_bbox):
    # rel_mat: Array[n,n]
    # cells_bbox: Array[n,4]
    num_cells = rel_mat.shape[0]
    # create adjacency list
    cells_adj = [[[],[],[],[]] for ind in range(num_cells)]
    for sub in range(rel_mat.shape[0]):
        for obj in range(rel_mat.shape[1]):
            rel = rel_mat[sub,obj].astype(np.int32)
            if sub == obj or rel == 0: continue
            cells_adj[sub][rel-1].append(obj)
    
    left_list = set([])
    top_list = set([])
    right_list = set([])
    down_list = set([])
    cells = []
    isolated_cells = []

    #sort cells in every direction
    for cell_ind in range(num_cells):
        cells.append({'cell_ind':cell_ind})
        cells_adj[cell_ind][0] = sorted(cells_adj[cell_ind][0], key = lambda x: cells_bbox[x,1]) #l
        cells_adj[cell_ind][1] = sorted(cells_adj[cell_ind][1], key = lambda x: cells_bbox[x,1]) #r
        cells_adj[cell_ind][2] = sorted(cells_adj[cell_ind][2], key = lambda x: cells_bbox[x,0]) #u
        cells_adj[cell_ind][3] = sorted(cells_adj[cell_ind][3], key = lambda x: cells_bbox[x,0]) #d
        len_l = len(cells_adj[cell_ind][0])
        len_r = len(cells_adj[cell_ind][1])
        len_u = len(cells_adj[cell_ind][2])
        len_d = len(cells_adj[cell_ind][3])
        if len_l == 0 and len_r == 0 and len_u == 0 and len_d == 0:
            isolated_cells.append(cell_ind)
            cells[cell_ind]['isolated'] = True
        else: cells[cell_ind]['isolated'] = False

    #find multirow/colomn for each cell
    for cell_ind in range(num_cells):
        mrow, mcol = findMrowMcol(cells_adj, cell_ind)
        cells[cell_ind]['multirow'] = mrow
        cells[cell_ind]['multicol'] = mcol
        # find cells at left and top boundery
        if cells[cell_ind]['isolated']: continue
        if len(cells_adj[cell_ind][0]) == 0 : left_list.add(cell_ind)
        if len(cells_adj[cell_ind][2]) == 0 : top_list.add(cell_ind)

    # build graph
    cells_graph = [[0 for column in range(num_cells)]
                   for row in range(num_cells)]
    for cell_ind in range(num_cells):
        for dir_ind in range(4):
            for adj_ind in cells_adj[cell_ind][dir_ind]:
                edge = 0
                if dir_ind == 0:
                    edge = cells[adj_ind]['multicol']
                elif dir_ind == 1:
                    edge = cells[cell_ind]['multicol']
                elif dir_ind == 2:
                    edge = cells[adj_ind]['multirow']
                else:
                    edge = cells[cell_ind]['multirow']

                cells_graph[cell_ind][adj_ind] = edge

    shortestP = ShortestDis(cells_graph)

    l_remove = set([])
    t_remove = set([])
    for left1 in left_list:
        for left2 in left_list:
            if left1 == left2: continue
            l_minpath = shortestP.cpmin(left1, [left2])
            l_dis = cmp_dis(cells_adj, l_minpath, 0)
            if l_dis > 0: l_remove.add(left1)
    for top1 in top_list:
        for top2 in top_list:
            if top1 == top2: continue
            t_minpath = shortestP.cpmin(top1, [top2])
            t_dis = cmp_dis(cells_adj, t_minpath, 2)
            if t_dis > 0: t_remove.add(top1)
    for l_re in l_remove:
        left_list.remove(l_re)
    for t_re in t_remove:
        top_list.remove(t_re)

    #find total rows and cols
    #rows,cols = findHW(cells_adj, left_list, top_list)

    max_row = 0
    max_col = 0

    #find coordinate for each cell
    for cell_ind in range(num_cells):
        x_minpath = shortestP.cpmin(cell_ind, left_list)
        y_minpath = shortestP.cpmin(cell_ind, top_list)
        start_col = cmp_dis(cells_adj, x_minpath, 0)
        start_row = cmp_dis(cells_adj, y_minpath, 2)
        end_row = start_row + cells[cell_ind]['multirow'] - 1
        end_col = start_col + cells[cell_ind]['multicol'] - 1
        max_row = max(max_row, end_row)
        max_col = max(max_col, end_col)
        cells[cell_ind]['lloc'] = [start_row, end_row, start_col, end_col]

    output = np.array([ cell_i['lloc'] for cell_i in cells ], dtype=np.int32)
    
    return output

def findMrowMcol(cells_adj, start):
    start_list = [start]
    # direction,  0: left, 1: right, 2: top, 3: down
    _, left_width, _  = depth(cells_adj, start_list, 0, ifhorizontal=False)
    _, right_width, _ = depth(cells_adj, start_list, 1, ifhorizontal=False)
    _, top_width, _   = depth(cells_adj, start_list, 2, ifhorizontal=False)
    _, down_width, _  = depth(cells_adj, start_list, 3, ifhorizontal=False)
    multi_row = max(left_width, right_width)
    multi_col = max(top_width, down_width)
    return multi_row, multi_col

def findHW(cells_adj, left_list, top_list):

    w,_,_ = depth(cells_adj, left_list, 1, ifhorizontal=True)
    h,_,_ = depth(cells_adj, top_list, 3, ifhorizontal=True)

    return h, w

def depth(cells_adj, start_list, dir_ind, ifhorizontal=False):
    dpts = [1]
    bdts = [len(start_list)]
    levs = []
    for start in start_list:
        stack = [(1,set([start]))] #(layer, [cell_ind])
        while stack:
            dpt, lev = stack.pop() #depth, breadth, level list
            levs.append(lev)
            all_next_lev = set([])
            for cell in lev:
                next_lev = set([])
                for ncell in cells_adj[cell][dir_ind]:
                    next_lev.add(ncell)

                if ifhorizontal:
                    next_hor_lev = set([])
                    dir_set = [2,3] if dir_ind in [0,1] else [0,1]
                    for ncell in next_lev:
                        for dir_h in dir_set:
                            _, _, h_levs= depth(cells_adj, [ncell], dir_h, ifhorizontal=False)
                            for h_levs_i in h_levs:
                                next_hor_lev = next_hor_lev | h_levs_i

                    next_lev = next_lev | next_hor_lev
                all_next_lev = all_next_lev | next_lev
            if len(all_next_lev) != 0:
                dpts.append(dpt+1)
                bdts.append(len(all_next_lev))
                stack.append((dpt+1,all_next_lev))
                    #levs.append(all_next_lev)
                    #else: last_lev = last_lev | lev
    return max(dpts), max(bdts), levs

def cmp_dis(cells_adj, path, direction):
    dis = 0

    for cell_i in range(len(path)):
        if cell_i+1 >= len(path): break
        if direction == 0:
            if path[cell_i+1] in cells_adj[path[cell_i]][0]: dis += 1
            elif path[cell_i+1] in cells_adj[path[cell_i]][1]: dis -= 1
        elif direction == 2:
            if path[cell_i+1] in cells_adj[path[cell_i]][2]: dis += 1
            elif path[cell_i+1] in cells_adj[path[cell_i]][3]: dis -= 1
    #if path[0]==19 and path[-1]==0: print(dis,path)
    return dis