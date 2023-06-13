import os, sys
import os.path as osp
from mtools import read_file
sys.path.append('tools')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb as pdb
from tools import get_gnss_files, get_rnx_files, get_rtkfiles, load_json, convert_RTKLite_log, write_file, get_info

data_path = 'IGRData/IGR_cjy'

devices = read_file(osp.join(data_path, 'devices.txt'))

lat_lon_dict_file = 'Configs/lat_lon_dict.json'
lat_lon_dict = load_json(lat_lon_dict_file)
all_trace = []

def get_near_area(rtkfiles_path):
    if not osp.exists(rtkfiles_path.replace('txt', 'csv')):
        convert_RTKLite_log(rtkfiles_path, rtkfiles_path.replace('txt', 'csv'))
    data = pd.read_csv(rtkfiles_path.replace('txt', 'csv'))
    all_areas = list(lat_lon_dict.keys())
    mean_lat_lon = data[['LatitudeDegrees', 'LongitudeDegrees']].mean(axis=0).values
    return all_areas[np.argmin([np.linalg.norm(mean_lat_lon - np.array(lat_lon_dict[_])) for _ in all_areas])]

def plot_traj(csv_name, latlon):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(latlon[:, 0], latlon[:, 1], label="Ground Truth")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # ax.axis('equal')
    ax.set_title(csv_name.replace('.csv', ''))
    fig.tight_layout()
    fig_name = csv_name.replace('.csv', '.png')
    fig.savefig(fig_name, dpi=300)
    print(f'save png to {fig_name}')
    plt.close(fig)

def write_yaml(path_dir, _data_dir, _device, trace, gnss_file, rnx_file, sensor_file, rtkfile, _area, _shape, _type, _people):
    write_file(os.path.join(path_dir, 'info.yaml'), 
        [   
            f'data_dir: {_data_dir}', 
            f'device: {_device}', 
            f'trace: {trace}', 
            f'gnss_file: {gnss_file}',
            f'rnx_file: {rnx_file}',
            f'sensor_file: {sensor_file}',
            f'rtkfile: {rtkfile}', 
            f'area: {_area}', 
            f'shape: {_shape}', 
            f'type: {_type}', 
            f'people: {_people}',
        ]
        )
    
for device in devices:
    traces = os.listdir(osp.join(data_path, 'processed', device))
    print(traces)
    for trace in traces:
        trace_path = osp.join(data_path, 'processed', device, trace, 'supplementary')
        files = os.listdir(trace_path)
        print(files)
        gnss_file = get_gnss_files(files)[0]
        rnx_file = get_rnx_files(files)[0]
        rtkfile = get_rtkfiles(files)[0]
        _area = get_near_area(osp.join(trace_path, rtkfile))
        _old_info = get_info(trace_path)
        _old_area = _old_info['route']
        _shape = _old_info['shape']
        if _old_area != _area:
            # s = input(f'chose area: 0. {_area} 1. {_old_area} (old), else')
            # if s=='1':
            #     _area = _old_area
            # elif s!= '0':
            #     _area = s
            _area = _old_area
        info = ','.join((osp.basename(data_path), device, trace, gnss_file, rnx_file, '', rtkfile, _area, _shape, '', 'cuijiayang'))
        write_yaml(trace_path, osp.basename(data_path), device, trace, gnss_file, rnx_file, '', rtkfile, _area, _shape, '', 'cuijiayang')
        all_trace.append(info)
        csv_name = osp.join(trace_path, rtkfile).replace('txt', 'csv')
        gngga_df = pd.read_csv(csv_name)
        latlon = gngga_df[['LatitudeDegrees', 'LongitudeDegrees']].values
        if not osp.exists(osp.join(trace_path, rtkfile).replace('txt', 'png')):
            plot_traj(csv_name, latlon)
write_file(osp.join(data_path, 'info_list.txt'), all_trace)