from tools.tools import load_paths
from mtools import list_con, save_json

data_dirs = ['IGR230421', 'IGR230422']
paths_list = list_con([load_paths(f'IGRData/{data_dir}/path_list.txt') for data_dir in data_dirs])

data_names  = [_[0] for _ in paths_list]
route_names = [_[1][:-2] for _ in paths_list]
print(dict(zip(data_names, route_names)))
data_route_dict = dict(zip(data_names, route_names))
save_json('IGRProcessed/Indoor_paths/data_route_dict.json', data_route_dict)