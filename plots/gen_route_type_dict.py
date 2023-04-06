import os.path as osp
from mtools import read_file, save_json
IGR_DIRS = ['IGR', 'IGR230307', 'IGR230312']

route_type_dict = {}
for IGR_DIR in IGR_DIRS:
    paths = read_file(osp.join(IGR_DIR, 'path_list.txt'))
    paths = list(filter(lambda x: not x.startswith('#'), paths))
    for path in paths:
        # print(path)
        path = path.split(',')
        trace, route, shape, _type, rtklite = path
        route_type_dict[trace] = _type
        
save_json(osp.join('plots', 'route_type_dict.json'), route_type_dict)