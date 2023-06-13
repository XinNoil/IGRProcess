import os.path as osp
from mtools import read_file, save_json
IGR_DIRS = ['IGR', 'IGR230307', 'IGR230312', 'IGR230415']

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

train_list = read_file(osp.join('plots', 'list_train.txt'))
test_list = read_file(osp.join('plots', 'list_test.txt'))

train_type_dict = {}
for _ in train_list:
    train_type_dict[_] = 'train'

for _ in test_list:
    train_type_dict[_] = 'test'

save_json(osp.join('plots', 'train_type_dict.json'), train_type_dict)