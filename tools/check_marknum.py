import os, argparse
import os.path as osp
import pandas as pd
from tools import load_paths
from mtools import read_file, load_json
import ipdb as pdb

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str, default='IGRData/IGR_indoor_230506')
parser.add_argument('-m', '--mark_json', type=str, default='IGRProcessed/Indoor_paths/04-21_22_mark.json')
args = parser.parse_args()

marks_dict = load_json(args.mark_json)
os.chdir(osp.join('IGRData', args.data_dir))
paths = load_paths('path_list.txt')
devices = read_file('devices.txt')

for device in devices:
    for _path in paths:
        trace, route, shape, _type, rtklite, people = _path
        mark_df = pd.read_csv(f"processed/{device}/{trace}/Mark.csv")
        marks = marks_dict[route]
        record_mark_num = len(mark_df['LID'].unique())+2
        config_mark_num = len(marks)
        print(trace, route, record_mark_num, config_mark_num)
        if record_mark_num != config_mark_num:
            print(f'Error: mark number error: record/config: {record_mark_num}/{config_mark_num}')
            pdb.set_trace()

# python tools/check_marknum.py -d IGR_indoor_230506