"""
Usage:
  process.py <data_path> [options]

Options:
  -i <indoor>, --indoor <indoor>  indoor [default: 0]
  -s <subtype>, --subtype <subtype> subtype [default: 0]
"""
import ipdb as pdb
import os, shutil, copy
import os.path as osp
from tools.tools import read_file, write_file, load_paths
from GNSSLogger_convert import convert_RTKLite_log
from tools.process_rtk import process_rtk_files
from docopt import docopt
from mtools import print_each, list_con

def get_device_paths_files(device):
    if os.path.exists(os.path.join('origin', device, 'path_list.txt')):
        _paths = load_paths(os.path.join('origin', device, 'path_list.txt'))
    else:
        _paths = copy.copy(paths)
    files = os.listdir(os.path.join('origin', device))
    files.sort()
    if subtype==1:
        files = sorted(list(filter(lambda x: osp.isdir(os.path.join('origin', device, x)), files)))
        files = list_con([[osp.join(file, _) for _ in sorted(os.listdir(os.path.join('origin', device, file)))] for file in files])
    return _paths, files

def process_paths(_paths, device):
    for path in _paths:
        trace, route, shape, _type, rtklite, people, txt, _23o = path
        path_dir = os.path.join('processed', device, trace, 'supplementary')
        os.makedirs(path_dir, exist_ok=True)
        print(f"{os.path.join('origin', device, txt)}")
        shutil.copyfile(os.path.join('origin', device, txt), os.path.join(path_dir, os.path.basename(txt)))
        if len(_23o):
            print(f"{os.path.join('origin', device, _23o)}")
            shutil.copyfile(os.path.join('origin', device, _23o), os.path.join(path_dir, os.path.basename(_23o)))
        if len(rtklite):
            if osp.exists(os.path.join('origin', 'rtklite', rtklite)):
                print(f"{os.path.join('origin', 'rtklite', rtklite)}")
                shutil.copyfile(os.path.join('origin', 'rtklite', rtklite), os.path.join(path_dir, rtklite))
            elif subtype:
                print(f"{os.path.join('origin', device, osp.join(osp.dirname(txt), rtklite))}")
                shutil.copyfile(os.path.join('origin', device, osp.join(osp.dirname(txt), rtklite)), os.path.join(path_dir, rtklite))
        print('')
        write_yaml(path_dir, device, trace, route, shape, _type, rtklite, people, os.path.basename(txt), os.path.basename(_23o))
        
def process_outdoor(devices):
    for device in devices:
        _paths, files = get_device_paths_files(device)
        txtfiles = list(filter(lambda x: ('23o' not in x) and ('rnx' not in x) and ('nmea' not in x) and ('gngga' not in x) and ('txt' in x), files))
        rnxfiles = list(filter(lambda x: '23o' in x, files))
        if len(rnxfiles):
            _paths = [path+[txtfile,rnxfile] for path,txtfile,rnxfile in zip(_paths, txtfiles, rnxfiles)]
        else:
            _paths = [path+[txtfile,''] for path,txtfile in zip(_paths, txtfiles)]
        process_paths(_paths, device)
    # process_rtk_files()

def process_indoor(devices):
    for device in devices:
        _paths, files = get_device_paths_files(device)
        txtfiles = list(filter(lambda x: ('csv' in x), files))
        _paths = [path+[txtfile,''] for path,txtfile in zip(_paths, txtfiles)]
        process_paths(_paths, device)

def write_yaml(path_dir, device, trace, route, shape, _type, rtklite, people, txt, _23o):
    write_file(os.path.join(path_dir, 'info.yaml'), 
        [   f'device: {device}', 
            f'trace: {trace}', 
            f'route: {route}', 
            f'shape: {shape}', 
            f'type: {_type}', 
            f'rtklite: {rtklite}', 
            f'people: {people}',
            f'txt: {txt}',
            f'23o: {_23o}']
        )

if __name__ == "__main__":
    arguments = docopt(__doc__)
    data_path = arguments.data_path
    indoor = int(arguments.indoor)
    subtype = int(arguments.subtype)

    os.chdir(osp.join('IGRData', data_path))
    print(f'{indoor=}')
    print(f'{subtype=}')
    paths = load_paths('path_list.txt')
    devices = read_file('devices.txt')
    print_each(paths)
    print(devices)
    
    if indoor:
        process_indoor(devices)
    else:
        process_outdoor(devices)
