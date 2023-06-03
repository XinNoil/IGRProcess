import os, re, argparse
import os.path as osp
from mtools import write_file, print_each, str2bool, list_con
import numpy as np
import ipdb as pdb

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str)
parser.add_argument('-r', '--routes',   type=str, default='')
parser.add_argument('-s', '--shapes',   type=str, default='') # , choices=['Straight', 'Circle', 'All']
parser.add_argument('-t', '--types',    type=str, default='') # , choices=['SemiOpen', 'Open', 'Indoor']
parser.add_argument('-p', '--peoples',  type=str, default='')
parser.add_argument('-st', '--subtype', type=str, default=None)
parser.add_argument('-i', '--indoor',   type=str2bool, default=False)
args = parser.parse_args()

origin_path = osp.join(args.data_dir, 'origin')
processed_path = osp.join(args.data_dir, 'processed')
routes = args.routes.split(',')
shapes = args.shapes.split(',')
types  = args.types.split(',')
peoples = args.peoples.split(',')

def get_path_o(*path):
    return osp.join(origin_path, *path)

def get_path_p(*path):
    return osp.join(processed_path, *path)

def get_gnss_files(files, not_subtype=True):
    if (args.subtype is not None) and not_subtype:
        files = sorted(list(filter(lambda x: osp.isdir(get_path_o(device, x)), files)))
        return list_con([sorted(get_gnss_files(os.listdir(get_path_o(device, file)), False)) for file in files])
    else:
        return sorted(list(filter(lambda x: re.match(r'gnss_log_[0-9_]*.txt', x), files)))
    
def get_sensor_files(files, not_subtype=True):
    if (args.subtype is not None) and not_subtype:
        files = sorted(list(filter(lambda x: osp.isdir(get_path_o(device, x)), files)))
        return list_con([sorted(get_sensor_files(os.listdir(get_path_o(device, file)), False)) for file in files])
    else:
        return sorted(list(filter(lambda x: re.match(r'[0-9.]*-[0-9_]*.csv', x), files)))

def get_rtkfiles(files, not_subtype=True):
    if (args.subtype is not None) and not_subtype:
        files = sorted(list(filter(lambda x: osp.isdir(get_path_o(device, x)), files)))
        return list_con([sorted(get_rtkfiles(os.listdir(get_path_o(device, file)), False)) for file in files])
    else:
        return sorted(list(filter(lambda x: re.match(r'[0-9]*gngga.txt', x), files)))
    
def get_subtypes(files):
    files = sorted(list(filter(lambda x: osp.isdir(get_path_o(device, x)), files)))
    return [len(get_sensor_files(os.listdir(get_path_o(device, file)), False))*[file] for file in files]

def get_trace_names(gnss_files, sensor_files):
    if len(gnss_files):
        return [_[11:-4] for _ in gnss_files]
    else:
        return [_[3:-4].replace('.', '_') for _ in sensor_files]

def get_closest_ind(all_rtkfiles_nums, datenum):
    ind = np.argmin(np.abs(datenum-np.array(all_rtkfiles_nums)))
    try:
        assert np.abs(all_rtkfiles_nums[ind] - datenum) < 60
    except:
        pdb.set_trace()
    return ind

def get_rtkfiles_from_all(all_rtkfiles, trace_names):
    all_rtkfiles_nums = [int(_[2:-9]) for _ in all_rtkfiles]
    if len(all_rtkfiles):
        rtkfiles = []
        for trace_name in trace_names:
            datenum = trace_name.replace('_', '')
            assert len(datenum)==12
            datenum = int(datenum)
            rtkfiles.append(all_rtkfiles[get_closest_ind(all_rtkfiles_nums, datenum)])
    else:
        rtkfiles = [''] * len(trace_names)
    return rtkfiles

devices = os.listdir(origin_path)
if 'rtklite' in devices:
    devices.remove('rtklite')
all_rtkfiles = []
if not args.indoor:
    if osp.exists(get_path_o('rtklite')):
        all_rtkfiles = os.listdir(get_path_o('rtklite'))
print(all_rtkfiles)
write_file(osp.join(args.data_dir, 'devices.txt'), devices)

all_paths = []
for device in devices:
    print(f'\nscan {device}...\n')
    files = os.listdir(get_path_o(device))
    gnss_files = get_gnss_files(files)
    sensor_files = get_sensor_files(files)
    print_each(gnss_files)
    print_each(sensor_files)
    
    if args.subtype is not None:
        subtypes = get_subtypes(files)
    if args.subtype == 'route':
        routes = list_con(subtypes)
    elif args.subtype == 'shape':
        shapes = list_con(subtypes)
    elif args.subtype == 'type':
        types = list_con(subtypes)
    elif args.subtype == 'people':
        peoples = list_con(subtypes)
    
    if args.indoor:
        trace_num = len(sensor_files)
        trace_names = get_trace_names([], sensor_files)
        rtkfiles  = get_rtkfiles_from_all(all_rtkfiles, trace_names)
    else:
        if len(gnss_files) and len(sensor_files):
            assert len(gnss_files) == len(sensor_files)
        trace_num = max(len(gnss_files), len(sensor_files))
        trace_names = get_trace_names(gnss_files, sensor_files)
        if len(all_rtkfiles):
            rtkfiles  = get_rtkfiles_from_all(get_rtkfiles(all_rtkfiles), trace_names)
        else:
            rtkfiles  = get_rtkfiles_from_all(get_rtkfiles(files), trace_names)
        print_each(rtkfiles)

    if len(routes)==1:
        routes = routes * trace_num
    elif (args.subtype is not None) and (args.subtype != 'route'):
        routes = routes * len(subtypes)

    if len(shapes)==1:
        shapes = shapes * trace_num
    elif (args.subtype is not None) and (args.subtype != 'shape'):
        shapes = shapes * len(subtypes)

    if len(types)==1:
        types = types * trace_num
    elif (args.subtype is not None) and (args.subtype != 'type'):
        types = types * len(subtypes)

    if len(peoples)==1:
        peoples = peoples * trace_num
    elif (args.subtype is not None) and (args.subtype != 'people'):
        peoples = peoples * len(subtypes)
    
    print(trace_names, routes, shapes, types, rtkfiles, peoples)
    paths = [','.join((trace, route, shape, _type, rtklite, people)) for trace, route, shape, _type, rtklite, people in zip(trace_names, routes, shapes, types, rtkfiles, peoples)]
    print_each(paths)
    write_file(get_path_o(device, 'path_list.txt'), paths)
    for path in paths:
        if path not in all_paths:
            all_paths.append(path)
write_file(osp.join(args.data_dir, 'path_list.txt'), all_paths)

# python tools/config.py -d IGRData/IGR230425 -r Around_55 -s Straight,Straight,Circle,All -t SemiOpen
# python tools/config.py -d IGRData/IGR230426 -r Tai_Lei -s Straight,Straight,Circle,All -t Open
# python tools/config.py -d IGRData/IGR_indoor_test -r 55 -s Straight -t Indoor -i True
# python tools/config.py -d IGRData/IGR_indoor_230506 -s Straight -t Indoor -i True -st route -p cuijiayang
# python tools/config.py -d IGRData/IGR230503 -r Around_55,Tai_Lei,Around_Lib,Between_Building,Around_Playground,Playground -s All -t SemiOpen,Open,SemiOpen,NarrowOpen,SemiOpen,Open -st people
# python tools/config.py -d IGRData/IGR230510 -r Around_55,Tai_Lei,Around_Lib,Between_Building,Around_Playground,Playground -s All -t SemiOpen,Open,SemiOpen,NarrowOpen,SemiOpen,Open