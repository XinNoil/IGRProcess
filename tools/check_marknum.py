import os, argparse
import os.path as osp
import pandas as pd
from tools import load_paths
from mtools import read_file, load_json
import ipdb as pdb

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str, default='IGRData/IGR_indoor_230506')
parser.add_argument('-m', '--mark_json', type=str, default='IGRProcessed/Indoor_paths/04-21_22_mark.json')
parser.add_argument('-ma', '--mark_add', type=int, default=1)
parser.add_argument('-il', '--info_list', type=str, default='info_list.csv')
args = parser.parse_args()

marks_dict = load_json(args.mark_json)
os.chdir(osp.join('IGRData', args.data_dir))
paths = load_paths(args.info_list)
devices = read_file('devices.txt')

for device in devices:
    for _path in paths:
        if args.info_list == 'info_list.csv':
            _data_dir, _device, trace, gnss_file, rnx_file, sensor_file, rtkfile, route, _shape, _type, _people = _path
        else:
            trace, route, shape, _type, rtklite, people = _path
        mark_df = pd.read_csv(f"processed/{device}/{trace}/Mark.csv")
        marks = marks_dict[route]
        record_mark_num = len(mark_df['LID'].unique())+args.mark_add
        config_mark_num = len(marks)
        print(trace, route, record_mark_num, config_mark_num)
        if record_mark_num != config_mark_num:
            print(f'Error: mark number error: record/config: {record_mark_num}/{config_mark_num}')
            pdb.set_trace()

# python tools/check_marknum.py -d IGR_indoor_230506 -md 2
# python tools/check_marknum.py -d IGR_indoor_241116_hsk
# python tools/check_marknum.py -d IGR_indoor_241116_ljl
# python tools/check_marknum.py -d IGR_indoor_241116_lmy
# python tools/check_marknum.py -d IGR_indoor_241116_zyp