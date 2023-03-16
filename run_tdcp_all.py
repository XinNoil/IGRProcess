import os, sys, argparse, time, gzip, requests
import os.path as osp
from datetime import datetime
from glob import glob
from tools.tools import load_json, read_file, get_info

data_path = 'IGR230312'
datadir = os.path.join(data_path, 'processed')

run_tdcp_file = 'run_tdcp.py'
exe_file = osp.join('tools', 'rtklib_tdcp.exe')
OVERWRITE_TDCP = True

dirs = read_file(osp.join(data_path, 'devices.txt'))
for _dir in dirs:
    if not os.path.isdir(osp.join(datadir,_dir)): continue
    subdirs = os.listdir(os.path.join(datadir,_dir))
    subdirs.sort()
    for _subdir in subdirs: #'01_12_12_11'        
        folder = osp.join(datadir, _dir, _subdir, 'supplementary')
        if not osp.isdir(folder): continue
        if OVERWRITE_TDCP or not osp.isfile(osp.join(folder,'tdcp.csv')):
            print('\n')
            print(_dir, _subdir, '....................................................') 
            info = get_info(folder)
            # print(info)
            # 获取obs, nav文件名
            nav_file = info['eph']
            obs_file = 'gnss_log.obs'
            
            term = f"python {run_tdcp_file} " +\
                    "-alg 1 -sys GCE -etype 2 -eph 0 -dts 1 -snr 20 -ele 15 -dop 0 -lli 0 -cs 0 " +\
                    "-get_LPDSNRLLI 0 -get_gt_N 0 -cs_repair 0 -get_mid 0 -max_speed 60 " +\
                   f"-exe_file {exe_file} -obs_file {osp.join(osp.realpath(folder), obs_file)} -beph_file {osp.join(osp.realpath(folder), nav_file)} "
            # print(term)
            os.system(term)
            import ipdb;ipdb.set_trace()