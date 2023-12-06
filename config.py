import os, re, argparse
import os.path as osp
from mtools import write_file, print_each, str2bool, list_con, load_json
import pandas as pd
import numpy as np
import ipdb as pdb
from tools.tools import get_path_o, get_gnss_files, get_rnx_files, get_sensor_files, scan_devices, scan_rtkfiles, get_trace_names, match_rtkfiles, get_subtypes, get_attribute, convert_RTKLite_log

# python config.py -d IGR230415 -a Tai_Lei -s Circle -t Open -p cuijiayang
# python config.py -d IGR230419 -a Tai_Lei -s Straight -t Open -p cuijiayang
# python config.py -d IGR230425 -a Around_55 -s Straight,Straight,Circle,All -t SemiOpen -p lizhaobang,hushunkang,zhangyupeng
# python config.py -d IGR230426 -a Tai_Lei -s Straight,Straight,Circle,All -t Open -p lizhaobang,hushunkang,zhangyupeng
# python config.py -d IGR_indoor_test -a 55 -s Straight -t Indoor -i True
# python config.py -d IGR_indoor_230506 -s Straight -t Indoor -i True -st route -p cuijiayang
# python config.py -d IGR230503 -a Around_55,Tai_Lei,Around_Lib,Between_Building,Around_Playground,Playground -s All -t SemiOpen,Open,SemiOpen,NarrowOpen,SemiOpen,Open -st people
# python config.py -d IGR230510 -a Around_55,Tai_Lei,Around_Lib,Between_Building,Around_Playground,Playground -s All -t SemiOpen,Open,SemiOpen,NarrowOpen,SemiOpen,Open

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str)
parser.add_argument('-a', '--areas',    type=str, default='')
parser.add_argument('-s', '--shapes',   type=str, default='') # , choices=['Straight', 'Circle', 'All']
parser.add_argument('-t', '--types',    type=str, default='') # , choices=['SemiOpen', 'Open', 'Indoor']
parser.add_argument('-p', '--peoples',  type=str, default='')
parser.add_argument('-st', '--subtype', type=str, default=None)
parser.add_argument('-i', '--indoor',   type=str2bool, default=False)
args = parser.parse_args()

origin_path = osp.join('IGRData', args.data_dir, 'origin')
devices = scan_devices(origin_path)
processed_path = osp.join('IGRData', args.data_dir, 'processed')
areas = args.areas.split(',')
shapes = args.shapes.split(',')
types  = args.types.split(',')
peoples = args.peoples.split(',')
if len(peoples)!=len(devices):
    peoples = peoples * len(devices)
print(peoples, devices)
subtypes = None
devices = list(filter(lambda x: osp.isdir(osp.join(origin_path, x)), devices))
if not args.indoor:
    all_rtkfiles = scan_rtkfiles(origin_path)
    print(all_rtkfiles)
write_file(osp.join('IGRData', args.data_dir, 'devices.txt'), devices)

all_trace = []

if args.areas == '':
    lat_lon_dict_file = 'Configs/lat_lon_dict.json'
    lat_lon_dict = load_json(lat_lon_dict_file)

def get_near_area(rtkfiles_path):
    if not osp.exists(rtkfiles_path.replace('txt', 'csv')):
        convert_RTKLite_log(rtkfiles_path, rtkfiles_path.replace('txt', 'csv'))
    data = pd.read_csv(rtkfiles_path.replace('txt', 'csv'))
    all_areas = list(lat_lon_dict.keys())
    mean_lat_lon = data[['LatitudeDegrees', 'LongitudeDegrees']].mean(axis=0).values
    return all_areas[np.argmin([np.linalg.norm(mean_lat_lon - np.array(lat_lon_dict[_])) for _ in all_areas])]

def get_areas_with_rtktiles(rtkfiles, subtype, peoples, origin_path, device):
    if subtype == 'people':
        rtkfiles_paths = [osp.join(origin_path, device, people, rtkfile) for rtkfile, people in zip(rtkfiles, peoples)]
    else:
        rtkfiles_paths = [osp.join(origin_path, 'rtklite', rtkfile) for rtkfile in rtkfiles]
    return [get_near_area(_) for _ in rtkfiles_paths]

for device, _people in zip(devices, peoples):
    print(f'\nscan {device}...\n')
    files = os.listdir(get_path_o(origin_path, device))
    gnss_files = get_gnss_files(files, args.subtype, origin_path, device)
    rnx_files = get_rnx_files(files, args.subtype, origin_path, device)
    sensor_files = get_sensor_files(files, args.subtype, origin_path, device)

    if len(gnss_files) and len(sensor_files):
        assert len(gnss_files) == len(sensor_files)
    trace_names = get_trace_names(gnss_files, sensor_files)
    trace_num = len(trace_names)
    rtkfiles = [''] * trace_num if args.indoor else match_rtkfiles(trace_names, all_rtkfiles, files, args.subtype, origin_path, device)

    # print_each(gnss_files)
    # print_each(sensor_files)
    # print_each(rtkfiles)

    if len(gnss_files)==0:
        gnss_files = [''] * trace_num
    if len(rnx_files)==0:
        rnx_files = [''] * trace_num
    if len(sensor_files)==0:
        sensor_files = [''] * trace_num
    if args.subtype is not None:
        subtypes = get_subtypes(origin_path, device, files)
    if args.subtype == 'people':
        peoples = list_con(subtypes)

    if args.areas == '' and not args.indoor:
        _areas = get_areas_with_rtktiles(rtkfiles, args.subtype, peoples, origin_path, device)
    else:
        _areas = get_attribute(trace_num, 'area', areas, args.subtype, subtypes)
    _shapes = get_attribute(trace_num, 'shape', shapes, args.subtype, subtypes)
    _types = get_attribute(trace_num, 'type', types, args.subtype, subtypes)
    _peoples = get_attribute(trace_num, 'people', [_people], args.subtype, subtypes)
    
    # for people, area in zip(_peoples, _areas):
    #     print(people, area)
    traces = [','.join((args.data_dir, device, trace, gnss_file, rnx_file, sensor_file, rtkfile, _area, _shape, _type, _people)) for trace, gnss_file, rnx_file, sensor_file, rtkfile, _area, _shape, _type, _people in zip(trace_names, gnss_files, rnx_files, sensor_files, rtkfiles, _areas, _shapes, _types, _peoples)]
    
    print_each(traces)
    write_file(get_path_o(origin_path, device, 'info_list.csv'), traces)
    for trace in traces:
        if trace not in all_trace:
            all_trace.append(trace)
write_file(osp.join('IGRData', args.data_dir, 'info_list.csv'), all_trace)