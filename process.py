import argparse
import ipdb as pdb
import os, shutil, copy
import os.path as osp
from tools.tools import read_file, write_file, load_paths
from mtools import print_each

def process_paths(_infos):
    for info in _infos:
        _data_dir, _device, trace, gnss_file, rnx_file, sensor_file, rtkfile, _area, _shape, _type, _people = info
        path_dir = os.path.join('processed', _device, trace, 'supplementary')
        os.makedirs(path_dir, exist_ok=True)

        if args.subtype == 'people':
            if osp.isfile(os.path.join('origin', _device, _people, rnx_file)):
                print(f"{os.path.join('origin', _device, _people, gnss_file)}")
                shutil.copyfile(os.path.join('origin', _device, _people, gnss_file), os.path.join(path_dir, gnss_file))
            if osp.isfile(os.path.join('origin', _device, _people, rnx_file)):
                print(f"{os.path.join('origin', _device, _people, rnx_file)}")
                shutil.copyfile(os.path.join('origin', _device, _people, rnx_file), os.path.join(path_dir, rnx_file))
            if osp.isfile(os.path.join('origin', _device, _people, rnx_file)):
                print(f"{os.path.join('origin', _device, _people, sensor_file)}")
                shutil.copyfile(os.path.join('origin', _device, _people, sensor_file), os.path.join(path_dir, sensor_file))
            if osp.isfile(os.path.join('origin', 'rtklite', rtkfile)):
                print(f"{os.path.join('origin', 'rtklite', rtkfile)}")
                shutil.copyfile(os.path.join('origin', 'rtklite', rtkfile), os.path.join(path_dir, rtkfile))
        else:
            if osp.isfile(os.path.join('origin', _device, gnss_file)):
                print(f"{os.path.join('origin', _device, gnss_file)}")
                shutil.copyfile(os.path.join('origin', _device, gnss_file), os.path.join(path_dir, gnss_file))
            if osp.isfile(os.path.join('origin', _device, rnx_file)):
                print(f"{os.path.join('origin', _device, rnx_file)}")
                shutil.copyfile(os.path.join('origin', _device, rnx_file), os.path.join(path_dir, rnx_file))
            if osp.isfile(os.path.join('origin', _device, sensor_file)):
                print(f"{os.path.join('origin', _device, sensor_file)}")
                shutil.copyfile(os.path.join('origin', _device, sensor_file), os.path.join(path_dir, sensor_file))
            if osp.isfile(os.path.join('origin', 'rtklite', rtkfile)):
                print(f"{os.path.join('origin', 'rtklite', rtkfile)}")
                shutil.copyfile(os.path.join('origin', 'rtklite', rtkfile), os.path.join(path_dir, rtkfile))
        write_yaml(path_dir, _data_dir, _device, trace, gnss_file, rnx_file, sensor_file, rtkfile, _area, _shape, _type, _people)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_path',     type=str)
    parser.add_argument('-st', '--subtype',     type=str, default=None)
    args = parser.parse_args()

    data_path = args.data_path
    os.chdir(osp.join('IGRData', data_path))
    infos = load_paths('info_list.csv')
    devices = read_file('devices.txt')
    devices = list(filter(lambda x: not x.startswith('#'), devices))
    print_each(infos)
    print(devices)
    
    process_paths(infos)
