""" 
下载广播星历
"""

import os
import re
import shutil
from datetime import datetime
import requests
import gzip
import os.path as osp
from tools.tools import *

data_path = r'IGR230312/processed'

# Use these to access the nav files from CDDIS.  
# This requires and account and setup of a .netrc file as described at https://cddis.nasa.gov/Data_and_Derived_Products/CreateNetrcFile.html.  
# Make sure this file is in the correct location which will be system dependent
nav_url_base = 'https://cddis.nasa.gov/archive/gnss/data/daily' #/2021/342/21p/ 
nav_file_base = 'BRDM00DLR_S_' # 20213420000_01D_MN.rnx.gz'' 

walk_path = osp.realpath(data_path)
print(walk_path)
paths = os.walk(walk_path)
urls = []
fname_list = []
for path, dir_list, file_list in paths:
    for file_name in file_list:
        if re.match(r'gnss_log_[0-9_]*.txt', file_name):
            print(os.path.join(path, file_name))
            ymd = file_name.split('_')
            ymd = list_ind(ymd, [2, 3, 4])
            doy = datetime(int(ymd[0]), int(ymd[1]), int(ymd[2])).timetuple().tm_yday # get day of year
            doy = str(doy).zfill(3) # day of year 不足3个字符的用0填充满
            fname = nav_file_base + ymd[0] + doy + '0000_01D_MN' + '.rnx.gz'
            url = '/'.join([nav_url_base, ymd[0], doy, ymd[0][-2:]+'p', fname])
            if url not in urls:
                urls.append(url)
            rel_path = '' if walk_path==path else path.replace(walk_path+osp.sep, '')
            fname_list.append([rel_path, file_name, fname[:-3]])

eph_path = osp.join(osp.dirname(walk_path), 'ephemeris')
os.makedirs(eph_path, exist_ok=True)
save_json(osp.join(eph_path, 'ephemeris.json'), fname_list)

for url in urls:
    fname = url.split('/')[-1][:-3]
    file_name = os.path.join(eph_path, fname)
    if os.path.exists(file_name):
        print(f'{fname} exist')
        continue  # file already exists
    try:
        print(f'download from {url} to {fname}')
        obs = gzip.decompress(requests.get(url).content) # get obs and decompress    
        # write nav data
        open(file_name, "wb").write(obs)
        print(url, 'success')
    except:
        print(f'Fail nav: {url}')

for path, file_name, fname in fname_list:
    shutil.copyfile(osp.join(eph_path, fname), osp.join(walk_path, path, fname))
    info = read_file(osp.join(walk_path, path, 'info.yaml'))
    eph_info = f'eph: {fname}'
    if eph_info not in info:
        info.append(eph_info)
        write_file(osp.join(walk_path, path, 'info.yaml'), info)