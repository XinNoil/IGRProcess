"run all doppler for lab dataset"

import os, argparse, sys
import os.path as osp
from datetime import datetime

# sys.path.append(join(split(realpath(__file__))[0],'../../Tools'))
# import tools as t
# import tools_pd as tpd
# from mtools import load_json
from tools.tools import load_json, read_file, get_info

data_path = 'IGR230312'
datadir = os.path.join(data_path, 'processed')

run_doppler_file = 'run_doppler.py'
exe_file = osp.join('tools', 'rtklib_doppler.exe')
OVERWRITE_DOP = False

dirs = read_file(osp.join(data_path, 'devices.txt'))
for _dir in dirs:
    if not os.path.isdir(osp.join(datadir,_dir)): continue
    subdirs = os.listdir(os.path.join(datadir,_dir))
    subdirs.sort()
    for _subdir in subdirs: #'01_12_12_11'        
        folder = osp.join(datadir, _dir, _subdir, 'supplementary')
        if not osp.isdir(folder): continue
        if OVERWRITE_DOP or not osp.isfile(osp.join(folder,'doppler.csv')):
            print(_dir, _subdir, '....................................................') 
            info = get_info(folder)
            # print(info)
            # # 获取obs, nav文件名
            nav_file = info['eph']
            obs_file = 'gnss_log.obs'
            
            term = f"python {run_doppler_file} "\
                    "-snr 25 -ele 15 "\
                    "-GPSa -1 -GPSb -1 -GLOa -1 -GLOb -1 "\
                    "-GALa -1 -GALb -1 -BDSa -1 -BDSb -1 "\
                f"-exe_file {exe_file} -obs_file {osp.join(osp.realpath(folder), obs_file)} -beph_file {osp.join(osp.realpath(folder), nav_file)} "
            print(term)
            print('\n')
            os.system(term)     
