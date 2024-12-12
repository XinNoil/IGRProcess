import os.path as osp
from tools.tools import read_data_list
from shutil import copyfile
from mtools import check_dir, write_file

train_infos, train_data_list, _ = read_data_list('IGR', osp.join('Configs', 'list_used_train.txt'))
test_infos, test_data_list, _ = read_data_list('IGR', osp.join('Configs', 'list_used_test.txt'))
data_list = sorted(list(set(train_data_list + test_data_list)))
src_path = 'IGRProcessed'
tgt_path = 'TJU_IMU_DS'

# for data in data_list:
#     if not osp.exists(osp.join('IGRProcessed', data, 'data.h5')):
#         print(data)
#     else:
#         check_dir(osp.join(tgt_path, data))
#         copyfile(osp.join(src_path, data, 'data.h5'), osp.join(tgt_path, data, 'data.h5'))
#         copyfile(osp.join(src_path, data, 'supplementary', 'info.yaml'), osp.join(tgt_path, data, 'info.yaml'))

print(len(data_list))

train_infos = sorted(list(set(train_infos)))
test_infos = sorted(list(set(test_infos)))

write_file(osp.join(tgt_path, 'train.txt'), train_infos)
write_file(osp.join(tgt_path, 'test.txt'), test_infos)

# zip -r TJU_IMU_DS.zip TJU_IMU_DS/*