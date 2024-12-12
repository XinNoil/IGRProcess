import os, argparse
import os.path as osp
import pandas as pd
from tools import load_paths
from mtools import read_file, load_json, save_json, write_file
import ipdb as pdb

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dirs', type=str, nargs='+', default=['IGRData/IGR_indoor_241116_hsk', 'IGRData/IGR_indoor_241116_ljl', 'IGRData/IGR_indoor_241116_lmy', 'IGRData/IGR_indoor_241116_zyp'])
# parser.add_argument('-m', '--mark_json', type=str, default='IGRProcessed/Indoor_paths/04-21_22_mark.json')
parser.add_argument('-o', '--output_name', type=str, default='IGR_indoor_241116_dict.json')
args = parser.parse_args()

route_dicts = {}
route_list = []

for data_dir in args.data_dirs:
    paths = load_paths(osp.join(data_dir, 'info_list.csv'))
    devices = read_file(osp.join(data_dir, 'devices.txt'))
    route_dict = {}
    for device in devices:
        for _path in paths:
            _data_dir, _device, trace, gnss_file, rnx_file, sensor_file, rtkfile, route, _shape, _type, _people = _path
            print(_path)
            route_dict[trace] = route
            route_list.append(','.join([osp.join(_device, trace), _people]))
    route_dicts[_people] = route_dict

save_json(osp.join('IGRProcessed/Indoor_paths', args.output_name), route_dicts)
write_file(osp.join('IGRProcessed/Indoor_paths', 'indoor_list.txt'), route_list)