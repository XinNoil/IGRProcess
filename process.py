import os, shutil
from tools.tools import read_file, write_file
from GNSSLogger_convert import convert_RTKLite_log

data_path = 'IGR230312'
os.chdir(data_path)

lines = read_file('path_list.txt')
devices = read_file('devices.txt')
# for device in devices:
#     paths = [line.split(',') for line in lines]
#     files = os.listdir(os.path.join('origin', device))
#     files.sort()
#     txtfiles = list(filter(lambda x: ('23o' not in x) and ('rnx' not in x), files))
#     rnxfiles = list(filter(lambda x: '23o' in x, files))
#     paths = [path+[txtfile,rnxfile] for path,txtfile,rnxfile in zip(paths, txtfiles, rnxfiles)]
#     for path in paths:
#         # print(path)
#         path_dir = os.path.join('processed', device, path[0], 'supplementary')
#         os.makedirs(path_dir, exist_ok=True)
#         print(f"{os.path.join('origin', device, path[4])}, {os.path.join(path_dir, path[4])}")
#         shutil.copyfile(os.path.join('origin', device, path[4]), os.path.join(path_dir, path[4]))
#         shutil.copyfile(os.path.join('origin', device, path[5]), os.path.join(path_dir, path[5]))
#         shutil.copyfile(os.path.join('origin', 'rtklite', path[3]), os.path.join(path_dir, path[3]))
#         write_file(os.path.join(path_dir, 'info.yaml'), 
#                    [f'device: {device}', 
#                     f'trace: {path[0]}', 
#                     f'route: {path[1]}', 
#                     f'shape: {path[2]}', 
#                     f'rtklite: {path[3]}', 
#                     f'txt: {path[4]}',
#                     f'23o: {path[5]}']
#                    )

rtk_path = os.path.join('origin','rtklite')
rtk_files = list(filter(lambda x: x.endswith('txt'), os.listdir(rtk_path)))

for rtk_file in rtk_files:
    rtk_file_name = os.path.join(rtk_path, rtk_file)
    csv_name = os.path.join(rtk_path, rtk_file.replace('txt','csv'))
    print(rtk_file_name, csv_name)
    convert_RTKLite_log(rtk_file_name=rtk_file_name, csv_name=csv_name)