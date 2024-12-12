# -*- coding: UTF-8 -*-
import os, argparse
import ipdb as pdb
from mtools import read_file, str2bool
from tools.tools import get_info, convert_AllSenosr_log

DATA_DIR = "processed"

def convert_one_dir(trip_dir):
    print(f"Converting {trip_dir}")
    folder = os.path.join(trip_dir, "supplementary")
    info = get_info(folder)
    txt_filename = info['sensor_file']
    convert_AllSenosr_log(os.path.join(trip_dir, "supplementary", txt_filename), trip_dir, OVERWRITE, IMU_OVERWRITE)

    if not txt_filename:
        print(f"[ERROR] file_dir does not contain txt file")

def all_convert():
    phone_dirs = read_file('devices.txt')
    phone_dirs = list(filter(lambda x: not x.startswith('#'), phone_dirs))
    for phone_dir in phone_dirs:
        for trip_dir in os.listdir(f"{DATA_DIR}/{phone_dir}"):
            if os.path.isdir(f"{DATA_DIR}/{phone_dir}/{trip_dir}"):
                convert_one_dir(f"{DATA_DIR}/{phone_dir}/{trip_dir}")

def single_convert(phone_dir, trip_dir):
    convert_one_dir(f"{DATA_DIR}/{phone_dir}/{trip_dir}")

"""
Usage:
  AllSensorLogger_convert.py -d <data_path> [options]

Options:
  -o <overwrite>, --overwrite <overwrite>  覆盖 [default: False]
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str)
    parser.add_argument('-o', '--overwrite', type=str2bool, default=False)
    parser.add_argument('-io', '--imu_overwrite', type=str2bool, default=False)
    args = parser.parse_args()
    IGR_DIR = args.data_path
    OVERWRITE = args.overwrite
    IMU_OVERWRITE = args.imu_overwrite
    os.chdir(os.path.join('IGRData', IGR_DIR))
    all_convert()