from mtools import load_json, save_json
import os.path as osp
import pandas as pd
import numpy as np
import ipdb as pdb
import sys
sys.path.append('tools')
from tools import load_paths, convert_RTKLite_log

lat_lon_dict = {}
lat_lon_dict_file = 'Configs/lat_lon_dict.json'
if osp.exists(lat_lon_dict_file):
    lat_lon_dict = load_json(lat_lon_dict_file)

info_list_file = 'Configs/area_rtk_list1.txt'
infos = load_paths(info_list_file)
rtkfile_path = infos[0][0]
rtkfiles = [_[0] for _ in infos[1:]]
areas = [_[1] for _ in infos[1:]]
print(rtkfile_path)
print(rtkfiles)
print(areas)

for area, rtkfile in zip(areas, rtkfiles):
    convert_RTKLite_log(osp.join(rtkfile_path, rtkfile), osp.join(rtkfile_path, rtkfile.replace('txt', 'csv')))
    data = pd.read_csv(osp.join(rtkfile_path, rtkfile.replace('txt', 'csv')))
    mean_lat_lon = data[['LatitudeDegrees', 'LongitudeDegrees']].mean(axis=0).values
    if area not in lat_lon_dict:
        lat_lon_dict[area] = mean_lat_lon.tolist()
    else:
        _areas = list(lat_lon_dict.keys())
        near_area = _areas[np.argmin([np.linalg.norm(mean_lat_lon - np.array(lat_lon_dict[_])) for _ in _areas])]
        if near_area != area:
            print(f'WARNING area {area} current loc {mean_lat_lon}, dict loc {lat_lon_dict[area]}, near_area {near_area} loc {lat_lon_dict[near_area]}')

save_json(lat_lon_dict_file, lat_lon_dict)