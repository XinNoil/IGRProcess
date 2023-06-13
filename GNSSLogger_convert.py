# -*- coding: UTF-8 -*-

import os, argparse
import ipdb as pdb
from mtools import read_file, str2bool
from tools.tools import get_info, convert_RTKLite_log, convert_GNSS_log
# from docopt import docopt

DATA_DIR = r"processed"

def convert_one_dir(trip_dir):
    print(f"Converting {DATA_DIR}/{trip_dir}")
    gnss_file_name = None
    rtk_file_name = None
    folder = os.path.join(trip_dir, "supplementary")
    info = get_info(folder)
    if 'gnss_file' not in info:
        pdb.set_trace()
    gnss_file_name = info['gnss_file']
    convert_GNSS_log(trip_dir, os.path.join(trip_dir, "supplementary", gnss_file_name))
    rtk_file_name = info['rtkfile']
    convert_RTKLite_log(f"{trip_dir}/supplementary/{rtk_file_name}", f"{trip_dir}/GNGGA.csv")

    if not gnss_file_name:
        print(f"[ERROR] file_dir does not contain gnss_log.txt")

    if not rtk_file_name:
        print(f"[ERROR] file_dir does not contain gngga.txt")

def all_convert():
    phone_dirs = read_file('devices.txt')
    phone_dirs = list(filter(lambda x: not x.startswith('#'), phone_dirs))
    for phone_dir in phone_dirs:
        for trip_dir in os.listdir(f"{DATA_DIR}/{phone_dir}"):
            if os.path.isdir(f"{DATA_DIR}/{phone_dir}/{trip_dir}"):
                if not os.path.exists(f"{DATA_DIR}/{phone_dir}/{trip_dir}/Raw.csv") or ALL_CONVERT_OVERRIDE_FLAG:
                    convert_one_dir(f"{DATA_DIR}/{phone_dir}/{trip_dir}")
                else:
                    print(f"Skipping {DATA_DIR}/{phone_dir}/{trip_dir}")

def single_convert(phone_dir, trip_dir):
    convert_one_dir(f"{DATA_DIR}/{phone_dir}/{trip_dir}")

"""
Usage:
  AllSensorLogger_convert.py -d <data_path> [options]

Options:
  -o <overwrite>, --overwrite <overwrite>  室内 [default: True]
"""
'''
    将 GNSSLogger 和 RTKLite 采集得到的 日志文件中的各种数据转换为 csv 文件
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str)
    parser.add_argument('-o', '--overwrite', type=str2bool, default=True)
    args = parser.parse_args()
    IGR_DIR = args.data_path
    ALL_CONVERT_OVERRIDE_FLAG = args.overwrite
    os.chdir(os.path.join('IGRData', IGR_DIR))
    all_convert()