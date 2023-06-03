# -*- coding: UTF-8 -*-
"""
Usage:
  AllSensorLogger_convert.py <data_path> [options]

Options:
  -o <overwrite>, --overwrite <overwrite>  室内 [default: 1]
"""
import os
import ipdb as pdb
import datetime
from datetime import timezone
import pandas as pd
import pynmea2
from mtools import read_file
from tools.tools import get_info
from docopt import docopt

HEADER_DEF = {
    "UncalAccel": "UncalAccel,elapsedRealtimeNanos,utcTimeMillis,UncalAccelXMps2,UncalAccelYMps2,UncalAccelZMps2,BiasXMps2,BiasYMps2,BiasZMps2",
    "UncalGyro": "UncalGyro,elapsedRealtimeNanos,utcTimeMillis,UncalGyroXRadPerSec,UncalGyroYRadPerSec,UncalGyroZRadPerSec,DriftXRadPerSec,DriftYRadPerSec,DriftZRadPerSec",
    "GameRot": "GameRot,elapsedRealtimeNanos,utcTimeMillis,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Rot": "Rot,elapsedRealtimeNanos,utcTimeMillis,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Mark":"Loc,elapsedRealtimeNanos,utcTimeMillis,LID", 
}

DATA_DIR = r"processed"

def convert_AllSenosr_log(trip_dir, txt_filename):
    # gnss_log_2022_12_30_12_31_01.txt -> 12_30_12_31

    uncal_accel_f = open(f"{trip_dir}/UncalAccel.csv", "w")
    uncal_accel_f.write(HEADER_DEF["UncalAccel"])
    uncal_accel_f.write("\n")

    uncal_gyro_f = open(f"{trip_dir}/UncalGyro.csv", "w")
    uncal_gyro_f.write(HEADER_DEF["UncalGyro"])
    uncal_gyro_f.write("\n")

    game_orientation_deg_f = open(f"{trip_dir}/GameRot.csv", "w")
    game_orientation_deg_f.write(HEADER_DEF["GameRot"])
    game_orientation_deg_f.write("\n")

    orientation_deg_f = open(f"{trip_dir}/Rot.csv", "w")
    orientation_deg_f.write(HEADER_DEF["Rot"])
    orientation_deg_f.write("\n")

    mark_f = open(f"{trip_dir}/Mark.csv", "w")
    mark_f.write(HEADER_DEF["Mark"])
    mark_f.write("\n")
    pdb.set_trace()
    with open(os.path.join(trip_dir, "supplementary", txt_filename), 'r', encoding='utf-8') as f:
        while (line := f.readline()):
            if line.startswith("#"):
                continue
            elif line.startswith("UAcc"):
                uncal_accel_f.write(line)
            elif line.startswith("UGys"):
                uncal_gyro_f.write(line)
            elif line.startswith("GameRot"):
                game_orientation_deg_f.write(line)
            elif line.startswith("Rot"):
                orientation_deg_f.write(line)
            elif line.startswith("Loc"):
                mark_f.write(line)

    uncal_accel_f.close()
    uncal_gyro_f.close()
    game_orientation_deg_f.close()
    orientation_deg_f.close()
    mark_f.close()


def convert_one_dir(trip_dir):
    print(f"Converting {DATA_DIR}/{trip_dir}")
    folder = os.path.join(trip_dir, "supplementary")
    info = get_info(folder)
    txt_filename = info['txt']
    convert_AllSenosr_log(trip_dir, txt_filename)

    if not txt_filename:
        print(f"[ERROR] file_dir does not contain txt file")

def all_convert():
    phone_dirs = read_file('devices.txt')
    for phone_dir in phone_dirs:
        for trip_dir in os.listdir(f"{DATA_DIR}/{phone_dir}"):
            if os.path.isdir(f"{DATA_DIR}/{phone_dir}/{trip_dir}"):
                if not os.path.exists(f"{DATA_DIR}/{phone_dir}/{trip_dir}/UncalAccel.csv") or ALL_CONVERT_OVERRIDE_FLAG:
                    convert_one_dir(f"{DATA_DIR}/{phone_dir}/{trip_dir}")
                else:
                    print(f"Skipping {DATA_DIR}/{phone_dir}/{trip_dir}")

def single_convert(phone_dir, trip_dir):
    convert_one_dir(f"{DATA_DIR}/{phone_dir}/{trip_dir}")



if __name__ == "__main__":
    # single_convert("Mate50", "03_07_14_49")
    arguments = docopt(__doc__)
    IGR_DIR = arguments.data_path
    ALL_CONVERT_OVERRIDE_FLAG = arguments.overwrite
    os.chdir(os.path.join('IGRData', IGR_DIR))
    all_convert()
# python AllSensorLogger_convert.py IGR_indoor_test