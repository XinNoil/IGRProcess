import os.path as osp
from mtools import read_file, save_json, write_file
import sys
sys.path.append('tools')
from tools import load_paths
IGR_DIRS = ['IGR_cjy', 'IGR230503', 'IGR230510'] # , 'IGR230415', 'IGR230307', 'IGR230312'

route_type_dict = {}
all_paths = []
for IGR_DIR in IGR_DIRS:
    paths = load_paths(osp.join('IGRData', IGR_DIR, 'info_list.txt'))
    for path in paths:
        print(path)
        _data_dir, _device, trace, gnss_file, rnx_file, sensor_file, rtkfile, _area, _shape, _type, _people = path
        all_paths.append(','.join(path))

write_file(osp.join('IGRProcessed', 'info_list.txt'), all_paths)