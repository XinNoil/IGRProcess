import os, shutil
from tools.tools import read_file, write_file
from GNSSLogger_convert import convert_RTKLite_log

data_path = 'IGR230312'
os.chdir(data_path)

lines = read_file('path_list.txt')
lines = list(filter(lambda x: not x.startswith('#'), lines))
devices = read_file('devices.txt')
for device in devices:
    paths = [line.split(',') for line in lines]
    files = os.listdir(os.path.join('origin', device))
    files.sort()
    txtfiles = list(filter(lambda x: ('23o' not in x) and ('rnx' not in x), files))
    rnxfiles = list(filter(lambda x: '23o' in x, files))
    paths = [path+[txtfile,rnxfile] for path,txtfile,rnxfile in zip(paths, txtfiles, rnxfiles)]
    for path in paths:
        # print(path)
        trace, route, shape, _type, rtklite, txt, _23o = path
        path_dir = os.path.join('processed', device, trace, 'supplementary')
        os.makedirs(path_dir, exist_ok=True)
        print(f"{os.path.join('origin', device, txt)}, {os.path.join(path_dir, txt)}")
        shutil.copyfile(os.path.join('origin', device, txt), os.path.join(path_dir, txt))
        shutil.copyfile(os.path.join('origin', device, _23o), os.path.join(path_dir, _23o))
        shutil.copyfile(os.path.join('origin', 'rtklite', rtklite), os.path.join(path_dir, rtklite))
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

rtk_path = os.path.join('origin','rtklite')
rtk_files = list(filter(lambda x: x.endswith('txt'), os.listdir(rtk_path)))

for rtk_file in rtk_files:
    rtk_file_name = os.path.join(rtk_path, rtk_file)
    csv_name = os.path.join(rtk_path, rtk_file.replace('txt','csv'))
    print(rtk_file_name, csv_name)
    convert_RTKLite_log(rtk_file_name=rtk_file_name, csv_name=csv_name)
