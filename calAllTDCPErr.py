# -*- coding: UTF-8 -*-
'''
    计算train目录下所有 RTKLIB 的 RTK 定位方式给出的POS位置和gt之间的误差
'''
import os,math,sys,pdb,argparse,re
from tools.RTKRecordReader import RTKRecordReader
from utm import from_latlon as ll2en
import numpy as np
np.set_printoptions(precision=10, suppress=True, formatter={'float_kind':'{:f}'.format})

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
pd.set_option('display.float_format','{:.10f}'.format)

from tools.tools import read_file
import os.path as osp
from glob import glob


def cal_one_err(gt_path, sol_df):
    # 计算真实的相对位移
    gt_np = RTKRecordReader().Process(gt_path).getECEFData() # ['utcTimestampinMS', 'efefx', 'ecefy', 'ecefz', 'GNGGAQuality']
    gt_df = pd.DataFrame(gt_np, columns=['utcTimestampinMS', 'efefx', 'ecefy', 'ecefz', 'GNGGAQuality'])
    gt_df['utcTimestampinMS'] = [int(i) for i in gt_df['utcTimestampinMS']]
    
    gt_rel_np = np.hstack((gt_np[:-1,0].reshape(-1,1), gt_np[1:,0].reshape(-1,1), gt_np[1:,1:4] - gt_np[:-1,1:4]))
    gt_rel_df = pd.DataFrame(gt_rel_np, columns=['epoch0_UTCMs', 'epoch1_UTCMs', 'XEcefM', 'YEcefM', 'ZEcefM']) 

    # sol_df:       ['epoch0_timeUTC','epoch1_timeUTC','x','y','z']
    # gt_rel_df:    ['epoch0_UTCMs', 'epoch1_UTCMs', 'XEcefM', 'YEcefM', 'ZEcefM']
    if len(gt_df)>0 and len(sol_df)>0:
        import ipdb;ipdb.set_trace()
        gnss_timestamps = np.unique([sol_df['epoch0_timeUTC'].unique() , sol_df['epoch1_timeUTC'].unique()])
        gt_timestamps = gt_df['utcTimestampinMS'].unique()

        indexes = np.searchsorted(gt_timestamps, gnss_timestamps) # searchsorted(<-) 找出某个元素放在哪个位置上才能保持原有的排列顺序
        from_t_to_fix_gnss = dict(zip(gnss_timestamps, gt_timestamps[indexes])) # [indexes-1]))
        
        # col_name = data.columns.tolist()
        # col_name.insert(col_name.index('UTCTimeS')+1,'UTCTimeMS_GT')  # 在UTCTimeS列后插入对齐的ground truth的时间
        # data = data.reindex(columns=col_name)
        sol_df['epoch0_UTCMs_GT'] = np.array(list(map(lambda v: from_t_to_fix_gnss[v], sol_df['epoch0_UTCMs']))) 
        sol_df['epoch1_UTCMs_GT'] = np.array(list(map(lambda v: from_t_to_fix_gnss[v], sol_df['epoch1_UTCMs']))) 

        
        sol_df = sol_df[abs(sol_df['epoch0_UTCMs_GT']-sol_df['epoch0_UTCMs'])<1]
        
        
        
    #     gt_MAP = {}
    #     for record in gt:
    #         gt_MAP[int(record[0]//1000)] = record[1:]

    #     all_count = len(gt)
    #     failed_count = 0
    #     success_count = 0
    #     err_list = []
    #     for pred_record in SOL:
    #         key = int(pred_record[0]//1000)
    #         if key in gt_MAP:
    #             real_record = gt_MAP[key]

    #             real_utm = ll2en(real_record[0], real_record[1])
    #             pred_utm = ll2en(pred_record[1], pred_record[2])

    #             err_e = pred_utm[0]-real_utm[0]
    #             err_n = pred_utm[1]-real_utm[1]
    #             err_h = np.linalg.norm((err_e, err_n))
                
    #             err_list.append([key, err_h, err_e, err_n, real_utm[0], real_utm[1], pred_utm[0], pred_utm[1]])
    #             success_count += 1
    #         else:
    #             failed_count += 1
        
    # return np.array(err_list), success_count, all_count
    

def calc_err_df(datadir):
    # err_df = pd.DataFrame()
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
            cal_one_err(gt_path, sol_df)
            # err_list, success_count, all_count = cal_one_err(gt_path, sol_df)

            # tmp_df = pd.DataFrame(err_list, columns=['timestamp', 'err_h', 'err_e', 'err_n', 'real_utm_e', 'real_utm_n', 'pred_utm_e', 'pred_utm_n'])
            # tmp_df.insert(0, 'phone', phone)
            # tmp_df.insert(0, 'trip', trip)

            # err_df = pd.concat((err_df, tmp_df), ignore_index=True)
            
            # m_index = pd.MultiIndex.from_arrays([[trip], [phone]], names=['trip', 'phone'])

            # # 统计误差
            # tmp_df = pd.DataFrame(err_list, columns=m_index)
            # err_df = pd.concat([err_df, tmp_df], axis=1)
            
            # # 统计覆盖率
            # tmp_df = pd.DataFrame([all_count, success_count], columns=m_index, index=['all_count', "success_count"])
            # count_df = pd.concat([count_df, tmp_df], axis=1)

            # print(f"{trip} {phone} Done")
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