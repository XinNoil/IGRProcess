"""
Usage:
  export_data.py <data_path>... [options]

Options:
  -e <export_type>, --export_type <export_type>       note

"""

import os
from mtools import read_file, write_file, list_remove
from mtools import monkey as mk
import pandas as pd
import ipdb as pdb
import numpy as np
from docopt import docopt

def read_h5(file_path):
    store = pd.HDFStore(file_path, mode='r')
    acc_df = store.get('acc')
    gys_df = store.get('gys')
    mag_df = store.get('mag')
    ori_df = store.get('ori')
    fix_df = store.get('fix')
    gngga_df = store.get('gngga')
    # dop_df = store.get('dop')
    store.close()
    return acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df # , dop_df

def scan_all_h5(IGR_dirs=None):
    if IGR_dirs is None:
        IGR_dirs = IGR_DIRS
    for IGR_DIR in IGR_dirs:
        phone_dirs = sorted(read_file(f"{IGR_DIR}/devices.txt"))
        for phone_dir in phone_dirs:
            if not os.path.isdir(f"{IGR_DIR}/{DATA_DIR}/{phone_dir}"): 
                continue
            trip_dirs = sorted(os.listdir(f"{IGR_DIR}/{DATA_DIR}/{phone_dir}"))
            trip_dirs = list_remove(trip_dirs, exclude_paths)
            for trip_dir in trip_dirs:
                if not os.path.isdir(f"{IGR_DIR}/{DATA_DIR}/{phone_dir}/{trip_dir}"):
                    continue
                if os.path.exists(f"{IGR_DIR}/{DATA_DIR}/{phone_dir}/{trip_dir}/data.h5"):
                    mk.magic_append([f"{IGR_DIR}/{DATA_DIR}/{phone_dir}/{trip_dir}/data.h5"], "file_paths")
    [file_paths] = mk.magic_get("file_paths")
    return file_paths

def file_path_loop(func, file_paths):
    dfs = []
    for file_path in file_paths:
        print(file_path)
        data = read_h5(file_path)
        df = func(data, file_path)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df

def stat_phone_err():
    file_paths = scan_all_h5()
    df = file_path_loop(_stat_phone_err, file_paths)
    df.to_csv('../Output/stat_df.csv', index=False)

def _stat_phone_err(data, file_path):
    _, _, phone_dir, trip_dir, _ = str.split(file_path, '/')
    acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df = data # , dop_df
    src_np = fix_df['Source'].values == 'Sensor'
    phone_pos_np = fix_df[['PosE', 'PosN']].values[src_np]
    gt_pos_np = gngga_df[['PosE', 'PosN']].values[src_np]
    phone_label_np = phone_pos_np[1:] - phone_pos_np[:-1]
    gt_label_np = gt_pos_np[1:] - gt_pos_np[:-1]
    phone_err = phone_label_np - gt_label_np
    phone_err_norm = np.linalg.norm(phone_err, axis=1, keepdims=True)
    df_data = np.hstack((phone_err, phone_err_norm, gt_label_np))
    df = pd.DataFrame(df_data, columns=['err_x', 'err_y', 'err_h', 'gt_x', 'gt_y'])
    df['phone'] = phone_dir
    df['trip'] = trip_dir
    return df

def stat_phone_pos():
    file_paths = scan_all_h5()
    df = file_path_loop(_stat_phone_pos, file_paths)
    df.to_csv('../Output/stat_pos_df.csv', index=False)

def _stat_phone_pos(data, file_path):
    _, _, phone_dir, trip_dir, _ = str.split(file_path, '/')
    acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df = data # , dop_df
    src_np = fix_df['Source'].values == 'Sensor'
    fix_df = fix_df[src_np]
    gngga_df = gngga_df[src_np]
    phone_pos_np = fix_df[['timestamp', 'PosE', 'PosN']].values
    gt_pos_np = gngga_df[['PosE', 'PosN', 'Quality']].values
    df_data = np.hstack((phone_pos_np, gt_pos_np))
    df = pd.DataFrame(df_data, columns=['timestamp', 'pos_x', 'pos_y', 'gt_x', 'gt_y', 'Quality'])
    df['phone'] = phone_dir
    df['trip'] = trip_dir
    return df

def export_imu_data():
    for IGR_DIR in IGR_DIRS:
        file_paths = scan_all_h5([IGR_DIR])
        print(file_paths)
        df = file_path_loop(_export_imu_data, file_paths)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_hdf(f'../Output/all_imu_data_{IGR_DIR}.h5', 'all_imu', mode='w')

def _export_imu_data(data, file_path):
    _, _, phone_dir, trip_dir, _ = str.split(file_path, '/')
    acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df = data # , dop_df
    df_data = np.hstack((acc_df.values, gys_df.values[:, 2:], mag_df.values[:, 2:], ori_df.values[:, 2:]))
    df = pd.DataFrame(df_data, columns=acc_df.columns.to_list()+gys_df.columns.to_list()[2:]+mag_df.columns.to_list()[2:]+ori_df.columns.to_list()[2:])
    df['phone'] = phone_dir
    df['trip'] = trip_dir
    return df

export_func_dict = {
    'phone_err': stat_phone_err,
    'phone_fix': stat_phone_pos,
    'imu_data': export_imu_data
}

# IGR_DIRS = ['IGR_cjy'] # 'IGR', 'IGR230307', 'IGR230312'
# IGR_DIRS = ['IGR_cjy', 'IGR230307', 'IGR230312', 'IGR230415']
# IGR_DIRS = ['IGR230415']

DATA_DIR = "processed"
exclude_paths = ['03_07_15_51', '03_12_15_38']
OVER_WRITE = True

def main():
    export_func_dict[export_type]()
   
if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    IGR_DIRS = arguments.data_path
    export_type = arguments.export_type
    os.chdir('IGRData')
    main()
