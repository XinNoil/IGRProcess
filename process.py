"""
Usage:
  process.py <data_path> [options]

Options:
  -i <indoor>, --indoor <indoor>  室内 [default: 0]
"""

import os, shutil, copy
import os.path as osp
from tools.tools import read_file, write_file, load_paths
from GNSSLogger_convert import convert_RTKLite_log
from process_rtk import process_rtk_files
from docopt import docopt
from mtools import print_each

def get_device_paths_files(device):
    if os.path.exists(os.path.join('origin', device, 'path_list.txt')):
        _paths = load_paths(os.path.join('origin', device, 'path_list.txt'))
    else:
        _paths = copy.copy(paths)
    files = os.listdir(os.path.join('origin', device))
    files.sort()
    return _paths, files

def process_paths(_paths, device):
    for path in _paths:
        trace, route, shape, _type, rtklite, txt, _23o = path
        path_dir = os.path.join('processed', device, trace, 'supplementary')
        os.makedirs(path_dir, exist_ok=True)
        print(f"{os.path.join('origin', device, txt)}")
        shutil.copyfile(os.path.join('origin', device, txt), os.path.join(path_dir, txt))
        if len(_23o):
            print(f"{os.path.join('origin', device, _23o)}")
            shutil.copyfile(os.path.join('origin', device, _23o), os.path.join(path_dir, _23o))
        if len(rtklite):
            print(f"{os.path.join('origin', 'rtklite', rtklite)}")
            shutil.copyfile(os.path.join('origin', 'rtklite', rtklite), os.path.join(path_dir, rtklite))
        print('')
        write_yaml(path_dir, device, trace, route, shape, _type, rtklite, txt, _23o)
        
def process_outdoor(devices):
    for device in devices:
        _paths, files = get_device_paths_files(device)
        txtfiles = list(filter(lambda x: ('23o' not in x) and ('rnx' not in x) and ('nmea' not in x) and ('txt' in x), files))
        rnxfiles = list(filter(lambda x: '23o' in x, files))
        if len(rnxfiles):
            _paths = [path+[txtfile,rnxfile] for path,txtfile,rnxfile in zip(_paths, txtfiles, rnxfiles)]
        else:
            _paths = [path+[txtfile,''] for path,txtfile in zip(_paths, txtfiles)]
        process_paths(_paths, device)
    process_rtk_files()

def process_indoor(devices):
    for device in devices:
        _paths, files = get_device_paths_files(device)
        txtfiles = list(filter(lambda x: ('csv' in x), files))
        _paths = [path+[txtfile,''] for path,txtfile in zip(_paths, txtfiles)]
        process_paths(_paths, device)

def write_yaml(path_dir, device, trace, route, shape, _type, rtklite, txt, _23o):
    write_file(os.path.join(path_dir, 'info.yaml'), 
        [f'device: {device}', 
            f'trace: {trace}', 
            f'route: {route}', 
            f'shape: {shape}', 
            f'type: {_type}', 
            f'rtklite: {rtklite}', 
            f'txt: {txt}',
            f'23o: {_23o}']
        )

if __name__ == "__main__":
    arguments = docopt(__doc__)
    data_path = arguments.data_path
    indoor = int(arguments.indoor)

    os.chdir(osp.join('IGRData', data_path))
    print(f'{indoor=}')
    paths = load_paths('path_list.txt')
    devices = read_file('devices.txt')
    print_each(paths)
    print(devices)
    
    if indoor:
        process_indoor(devices)
    else:
        process_outdoor(devices)
