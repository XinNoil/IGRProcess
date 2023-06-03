import os, sys, argparse, time, gzip, requests
import os.path as osp
from datetime import datetime
from glob import glob
from tools.tools import load_json, read_file, get_info

data_path = 'IGR230312'
datadir = os.path.join(data_path, 'processed')

run_pntpos_file = 'run_pntpos.py'
exe_file = osp.join('tools', 'rtklib_pntpos.exe')
OVERWRITE_PNTPOS = True

dirs = read_file(osp.join(data_path, 'devices.txt'))
for _dir in dirs:
    if not os.path.isdir(osp.join(datadir,_dir)): continue
    subdirs = os.listdir(os.path.join(datadir,_dir))
    subdirs.sort()
    for _subdir in subdirs: #'01_12_12_11'        
        folder = osp.join(datadir, _dir, _subdir, 'supplementary')
        if not osp.isdir(folder): continue
        if OVERWRITE_PNTPOS or not osp.isfile(osp.join(folder,'pntpos.csv')):
            print('\n')
            print(_dir, _subdir, '....................................................') 
            info = get_info(folder)
            # print(info)
            # 获取obs, nav文件名
            nav_file = info['eph']
            obs_file = 'gnss_log.obs'
            
            term =  "python run_pntpos.py " +\
                    "-sys GCE -eph 0 -snr 20 -ele 15 " +\
                    "-pntpos_validate 1 -get_satinfo 0 " +\
                   f"-exe_file {exe_file} -obs_file {osp.join(osp.realpath(folder), obs_file)} -beph_file {osp.join(osp.realpath(folder), nav_file)} "
            # print(term)
            os.system(term)
            