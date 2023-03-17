# -*- coding: UTF-8 -*-
import os,math,sys,pdb,argparse,re
from tools.RTKRecordReader import RTKRecordReader
from tools.tools import read_file, pd2csv
import os.path as osp
from glob import glob

import numpy as np
np.set_printoptions(precision=10, suppress=True, formatter={'float_kind':'{:f}'.format})
import pandas as pd
pd.set_option('display.float_format','{:.5f}'.format)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import warnings
warnings.filterwarnings('ignore')

# 计算覆盖率
# 返回值:
# uncover_dis_list  : TDCP未覆盖段长度 (m)
# uncover_time_list : TDCP未覆盖段时长 (s)
# cover_rate : TDCP覆盖率 = 覆盖长度/路径总长 * 100
def get_uncover_dis_time(gt_df, sol_df):
    # sol_df: ['epoch0_timeUTC','epoch1_timeUTC','x','y','z']
    # gt_df:  ['timeUTC', 'x', 'y', 'z', 'GNGGAQuality']

    # 根据时间戳对齐ground truth和tdcp结果
    # 标记ground truth中有tdcp的地方cover_mask=1，没有的地方cover_mask=0
    gt_df['cover_mask'] = np.zeros(len(gt_df))

    # 因为ground truth采样为5Hz, 所以需要根据计算error时对齐的ground truth时间戳, 
    # 将epoch0_timeUTC_GT和epoch1_timeUTC_GT中间的ground truth都标记为被覆盖
    for idx in range(len(sol_df)):
        epoch0_timeUTC_GT = sol_df.iloc[idx]['epoch0_timeUTC_GT']
        epoch1_timeUTC_GT = sol_df.iloc[idx]['epoch1_timeUTC_GT']

        gt_idx0 = gt_df.query(f'timeUTC=={epoch0_timeUTC_GT}').index.tolist()[0]
        gt_idx1 = gt_df.query(f'timeUTC=={epoch1_timeUTC_GT}').index.tolist()[0]

        for gt_idx in range(gt_idx0,gt_idx1+1):
            gt_df.loc[gt_idx,'cover_mask'] = 1
    
    
    # 计算未覆盖段的长度和时间
    uncover_dis_list = [] # unit:m
    uncover_time_list = [] # unit:s
    # 从头到尾遍历 gt_df, 设置最开始状态为 未覆盖
    # 每当遇到当前遍历条目的 cover_mask 与当前状态不一样
    # 则发现了一段连续的 覆盖/未覆盖 段, 记录下来
    # 然后修改当前状态与 cover_mask 相同, 继续遍历
    seg_start_index = 0
    now_state = 0
    for idx in range(len(gt_df)):
        if gt_df.loc[idx, 'cover_mask'] != now_state:
            if idx > seg_start_index:
                if now_state == 1: # 发现了一个连续的 覆盖段
                    pass
                elif now_state == 0: # 发现了一个连续的 未覆盖段
                    # 三轴绝对位置 (N, 3)
                    seg_loc = gt_df.loc[seg_start_index:idx, ['x','y','z']].values
                    # 三轴前后时刻相对位移 (N-1, 3)
                    seg_dis = np.diff(seg_loc, axis=0)
                    # 运动路程总长度 float
                    seg_len = np.sum(np.linalg.norm(seg_dis, axis=1))

                    uncover_dis_list.append(seg_len)
                             
                    begin_time = gt_df.loc[seg_start_index, ['timeUTC']].values[0]
                    end_time = gt_df.loc[idx, ['timeUTC']].values[0]
                    uncover_time_list.append((end_time-begin_time)/1000)

            now_state = gt_df.loc[idx, 'cover_mask']
            seg_start_index = idx

    # 不要忘了最后一段未覆盖
    if (now_state == 0) and (len(gt_df)-1 > seg_start_index): 
        seg_loc = gt_df.loc[seg_start_index:len(gt_df)-1, ['x','y','z']].values
        seg_dis = np.diff(seg_loc, axis=0)
        seg_len = np.sum(np.linalg.norm(seg_dis, axis=1))
        uncover_dis_list.append(seg_len)
                    
        begin_time = gt_df.loc[seg_start_index, ['timeUTC']].values[0]
        end_time = gt_df.loc[len(gt_df)-1, ['timeUTC']].values[0]
        uncover_time_list.append((end_time-begin_time)/1000)

    # 计算 覆盖率 = 覆盖的长度/路径总长
    total_dis = 0.0
    for idx in range(0, len(gt_df)-1):
        pos0 = gt_df.loc[idx,['x','y','z']].values
        pos1 = gt_df.loc[idx+1,['x','y','z']].values
        total_dis += np.linalg.norm(pos0-pos1)
    cover_rate = 1.0 - sum(uncover_dis_list)/total_dis

    total_time = (gt_df['timeUTC'].iloc[-1] - gt_df['timeUTC'].iloc[0])/1000
    
    return uncover_dis_list, uncover_time_list, cover_rate*100, total_dis, total_time

def cal_one_err(gt_path, sol_df):
    gt_np = RTKRecordReader().Process(gt_path).getECEFData() # ['timeUTC', 'x', 'y', 'z', 'GNGGAQuality']
    gt_df = pd.DataFrame(gt_np, columns=['timeUTC', 'x', 'y', 'z', 'GNGGAQuality'])
    gt_df['timeUTC'] = [int(i) for i in gt_df['timeUTC']]
    
    # sol_df: ['epoch0_timeUTC','epoch1_timeUTC','x','y','z']
    # gt_df:  ['timeUTC', 'x', 'y', 'z', 'GNGGAQuality']
    if len(gt_df)>0 and len(sol_df)>0:
        gt_timestamps = gt_df['timeUTC'].unique()
        sol_df = sol_df.query(f"epoch0_timeUTC>={min(gt_timestamps)} and epoch1_timeUTC<={max(gt_timestamps)}") # 获取ground truth时间范围内的solution
        gnss_timestamps = np.unique(sol_df[['epoch0_timeUTC','epoch1_timeUTC']]) 
        
        indexes = np.searchsorted(gt_timestamps, gnss_timestamps) # searchsorted(<-) 找出某个元素放在哪个位置上才能保持原有的排列顺序
        from_t_to_fix_gnss = dict(zip(gnss_timestamps, gt_timestamps[indexes]))
        
        sol_df['epoch0_timeUTC_GT'] = np.array(list(map(lambda v: from_t_to_fix_gnss[v], sol_df['epoch0_timeUTC']))) 
        sol_df['epoch1_timeUTC_GT'] = np.array(list(map(lambda v: from_t_to_fix_gnss[v], sol_df['epoch1_timeUTC']))) 
        
        if len(sol_df):
            # solution的时间戳和ground truth的时间戳不能超过1s
            sol_df = sol_df[abs(sol_df['epoch0_timeUTC_GT']-sol_df['epoch0_timeUTC'])<1000]
            sol_df = sol_df[abs(sol_df['epoch1_timeUTC_GT']-sol_df['epoch1_timeUTC'])<1000]
            
            from_t_to_gtxyz = dict(zip(gt_df['timeUTC'], gt_df[['x','y','z']].to_numpy()))
            sol_df['x_gt'] = sol_df.apply(lambda x: from_t_to_gtxyz[x['epoch1_timeUTC_GT']][0]-from_t_to_gtxyz[x['epoch0_timeUTC_GT']][0], axis=1)
            sol_df['y_gt'] = sol_df.apply(lambda x: from_t_to_gtxyz[x['epoch1_timeUTC_GT']][1]-from_t_to_gtxyz[x['epoch0_timeUTC_GT']][1], axis=1)
            sol_df['z_gt'] = sol_df.apply(lambda x: from_t_to_gtxyz[x['epoch1_timeUTC_GT']][2]-from_t_to_gtxyz[x['epoch0_timeUTC_GT']][2], axis=1)
            
            sol_df[['x_err','y_err','z_err']] = np.array(sol_df[['x_gt','y_gt','z_gt']]) - np.array(sol_df[['x','y','z']])
            sol_df[['h_err']] = np.reshape(np.linalg.norm(sol_df[['x_err','y_err']], axis=1), [-1,1])
            sol_df['err'] = np.reshape(np.linalg.norm(sol_df[['x_err','y_err','z_err']], axis=1), [-1,1])

            err_pd = pd.DataFrame({'ErrHM':["%.2f"%sol_df['h_err'].mean()], 'ErrM':["%.2f"%sol_df['err'].mean()], \
                            'ErrXM':["%.2f"%sol_df['x_err'].abs().mean()], 'ErrYM':["%.2f"%sol_df['y_err'].abs().mean()], 'ErrZM':["%.2f"%sol_df['z_err'].abs().mean()], \
                            'minErrXM':["%.2f"%sol_df['x_err'].abs().min()], 'minErrYM':["%.2f"%sol_df['y_err'].abs().min()], 'minErrZM':["%.2f"%sol_df['z_err'].abs().min()], \
                            'maxErrXM':["%.2f"%sol_df['x_err'].abs().max()], 'maxErrYM':["%.2f"%sol_df['y_err'].abs().max()], 'maxErrZM':["%.2f"%sol_df['z_err'].abs().max()]})
            print(err_pd[['ErrHM','ErrM','ErrXM','ErrYM','ErrZM']])
            # pd2csv(err_pd, pntpos_err_mean_file)
            # pd2csv(sol_df, osp.join(osp.dirname(gt_path), 'tdcp_err.csv'))
            
            _,_,cover_rate,_,_ = get_uncover_dis_time(gt_df, sol_df)
            print(f'TDCP cover rate: {cover_rate:.2f}')

            return sol_df
    return None
    
def calc_err_df(datadir):
    err_df = pd.DataFrame()
   
    dirs = read_file(osp.join(osp.dirname(datadir), 'devices.txt'))
    for _dir in dirs:
        if not os.path.isdir(osp.join(datadir,_dir)): continue
        subdirs = os.listdir(os.path.join(datadir,_dir))
        subdirs.sort()
        for _subdir in subdirs: #'01_12_12_11'        
            folder = osp.join(datadir, _dir, _subdir, 'supplementary')
            if not osp.isdir(folder): continue

            gt_path = glob(osp.join(folder,'*gngga*'))
            if len(gt_path)>0:
                gt_path = gt_path[0]

            tdcp_path = osp.join(folder, 'tdcp.csv')
            sol_df = pd.read_csv(tdcp_path)
            
            print(_dir, _subdir)
            tmp_df = cal_one_err(gt_path, sol_df)

            if tmp_df is not None:
                tmp_df.insert(0, 'phone', _dir)
                tmp_df.insert(0, 'trip', _subdir)
                err_df = pd.concat((err_df, tmp_df), ignore_index=True)      
    return err_df

def main():
    data_path = 'IGR230312'
    datadir = os.path.join(data_path, 'processed')

    err_path = osp.join(datadir, "tdcp_err.csv")
    OVERWRITE = False

    if os.path.exists(err_path) and OVERWRITE == False:
        err_df = pd.read_csv(err_path, index_col=0)
    else:
        err_df = calc_err_df(datadir)
        pd2csv(err_df, err_path)
    
    
    ##### 误差随时间变化
    # sns.set(font_scale=2)
    # sns.set_style("whitegrid", {"axes.edgecolor": "0"})
    
    # trips = err_df['trip'].unique()
    # for i in range(0, len(trips), 4):
    #     now_df = err_df[err_df['trip'].isin(trips[i:i+4])]

    #     g = sns.relplot(data=now_df, kind="line",
    #         x='timestamp', y='err_h', hue='phone', 
    #         col='trip', col_wrap=2, height=6, aspect=3, 
    #         facet_kws={'sharey': False, 'sharex': False},
    #     )
    #     g.set_ylabels("Error (m)")
        
    #     if args.exe=='demo5':
    #         g.savefig(f"./fig/Err_vs_time/Err_vs_time_{i}.png", dpi=200)  
    #     elif args.exe=='tomojitakasu':
    #         g.savefig(f"./fig/Err_vs_time/Err_vs_time_tomojitakasu_{i}.png", dpi=200)  
             

    # des_err = err_df.describe(percentiles=[.25, .5, .75, .95, .99])
    # rst = pd.concat([des_err, count_df])
    # rst.T.to_csv("./rst.csv")

if __name__ == "__main__":
    main()