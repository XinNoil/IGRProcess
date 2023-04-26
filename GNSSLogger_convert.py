# -*- coding: UTF-8 -*-
"""
Usage:
  AllSensorLogger_convert.py <data_path> [options]

Options:
  -o <overwrite>, --overwrite <overwrite>  室内 [default: 1]
"""
'''
    将 GNSSLogger 和 RTKLite 采集得到的 日志文件中的各种数据转换为 csv 文件
'''
import os
import ipdb as pdb
import datetime
from datetime import timezone
import pynmea2
from mtools import read_file
from tools.tools import get_info
from docopt import docopt

DATA_DIR = r"processed"

HEADER_DEF = {
    "Raw": "Raw,utcTimeMillis,TimeNanos,LeapSecond,TimeUncertaintyNanos,FullBiasNanos,BiasNanos,BiasUncertaintyNanos,DriftNanosPerSecond,DriftUncertaintyNanosPerSecond,HardwareClockDiscontinuityCount,Svid,TimeOffsetNanos,State,ReceivedSvTimeNanos,ReceivedSvTimeUncertaintyNanos,Cn0DbHz,PseudorangeRateMetersPerSecond,PseudorangeRateUncertaintyMetersPerSecond,AccumulatedDeltaRangeState,AccumulatedDeltaRangeMeters,AccumulatedDeltaRangeUncertaintyMeters,CarrierFrequencyHz,CarrierCycles,CarrierPhase,CarrierPhaseUncertainty,MultipathIndicator,SnrInDb,ConstellationType,AgcDb,BasebandCn0DbHz,FullInterSignalBiasNanos,FullInterSignalBiasUncertaintyNanos,SatelliteInterSignalBiasNanos,SatelliteInterSignalBiasUncertaintyNanos,CodeType,ChipsetElapsedRealtimeNanos",
    "UncalAccel": "UncalAccel,utcTimeMillis,elapsedRealtimeNanos,UncalAccelXMps2,UncalAccelYMps2,UncalAccelZMps2,BiasXMps2,BiasYMps2,BiasZMps2",
    "UncalGyro": "UncalGyro,utcTimeMillis,elapsedRealtimeNanos,UncalGyroXRadPerSec,UncalGyroYRadPerSec,UncalGyroZRadPerSec,DriftXRadPerSec,DriftYRadPerSec,DriftZRadPerSec",
    "UncalMag": "UncalMag,utcTimeMillis,elapsedRealtimeNanos,UncalMagXMicroT,UncalMagYMicroT,UncalMagZMicroT,BiasXMicroT,BiasYMicroT,BiasZMicroT",
    "OrientationDeg": "OrientationDeg,utcTimeMillis,elapsedRealtimeNanos,yawDeg,rollDeg,pitchDeg",
    "Fix": "Fix,Provider,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SpeedMps,AccuracyMeters,BearingDegrees,UnixTimeMillis,SpeedAccuracyMps,BearingAccuracyDegrees,elapsedRealtimeNanos,VerticalAccuracyMeters",
    "GNGGA": "utcTimeMillis,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SatNum,Quality,hdop",
}

def convert_GNSS_log(trip_dir, gnss_file_name):
    # gnss_log_2022_12_30_12_31_01.txt -> 12_30_12_31
    date_time = os.path.splitext(gnss_file_name)[0][14:-3]

    raw_f = open(f"{trip_dir}/Raw.csv", "w")
    raw_f.write(HEADER_DEF["Raw"])
    raw_f.write("\n")

    uncal_accel_f = open(f"{trip_dir}/UncalAccel.csv", "w")
    uncal_accel_f.write(HEADER_DEF["UncalAccel"])
    uncal_accel_f.write("\n")

    uncal_gyro_f = open(f"{trip_dir}/UncalGyro.csv", "w")
    uncal_gyro_f.write(HEADER_DEF["UncalGyro"])
    uncal_gyro_f.write("\n")

    uncal_mag_f = open(f"{trip_dir}/UncalMag.csv", "w")
    uncal_mag_f.write(HEADER_DEF["UncalMag"])
    uncal_mag_f.write("\n")

    orientation_deg_f = open(f"{trip_dir}/OrientationDeg.csv", "w")
    orientation_deg_f.write(HEADER_DEF["OrientationDeg"])
    orientation_deg_f.write("\n")

    fix_f = open(f"{trip_dir}/Fix.csv", "w")
    fix_f.write(HEADER_DEF["Fix"])
    fix_f.write("\n")

    Accel_found = False
    with open(os.path.join(trip_dir, "supplementary", gnss_file_name), 'r', encoding='utf-8') as f:
        while (line := f.readline()):
            if line.startswith("#"):
                continue
            elif line.startswith("Raw"):
                raw_f.write(line)
            elif line.startswith("UncalAccel"):
                uncal_accel_f.write(line)
            elif line.startswith("Accel"): # 有些手机可能会采集得到Accel而不是UncalAccel
                if Accel_found==False:
                    print("[Waring] Accel Found")
                    Accel_found = True
                uncal_accel_f.write(f'{line.strip().replace("Accel", "UncalAccel")},0,0,0\n')
            elif line.startswith("UncalGyro"):
                uncal_gyro_f.write(line)
            elif line.startswith("UncalMag"):
                uncal_mag_f.write(line)
            elif line.startswith("OrientationDeg"):
                orientation_deg_f.write(line)
            elif line.startswith("Fix"):
                fix_f.write(line)

    raw_f.close()
    uncal_accel_f.close()
    uncal_gyro_f.close()
    uncal_mag_f.close()
    orientation_deg_f.close()
    fix_f.close()

def convert_RTKLite_log(trip_dir=None, rtk_file_name=None, csv_name=None):
    if csv_name is None:
        csv_name = f"{trip_dir}/GNGGA.csv"
    gngga_f = open(csv_name, "w")
    gngga_f.write(HEADER_DEF["GNGGA"])
    gngga_f.write("\n")
    rtk_file_path = rtk_file_name if trip_dir is None else f"{trip_dir}/supplementary/{rtk_file_name}"
    
    with open(rtk_file_path, 'r', encoding='utf-8') as f:
        while (line := f.readline()):
            # 空格前后是时间戳和实际记录项
            tempList = line.split("     ")

            if len(tempList) != 2:
                print("[ERROR] len(tempList) != 2", tempList)
                continue
                
            # GPGGA 记录的是当天的时分秒 因此 使用这条记录的软件记录时间的年月日补充 形成完整的时间
            # 同时还要加上8小时 转换到北京时间
            recordTime = tempList[0] # RTK软件记录的时间
            # 解析软件时间字符串
            recordDatetime = datetime.datetime.strptime(recordTime, "%Y%m%d-%H%M%S")
            # 拿到GPGGA中的世界标准UTC时间
            msg = pynmea2.parse(tempList[1])
            # 使用软件记录的年月日补充GPGGA中的缺失, 注意 GNGGA 给出的是 UTC 时区的时间, 而不是东八区, 所以要显式指定时区
            utc_datetime = datetime.datetime(year=recordDatetime.year, month=recordDatetime.month, day=recordDatetime.day, \
                    hour=msg.timestamp.hour, minute=msg.timestamp.minute, second=msg.timestamp.second, \
                    microsecond=msg.timestamp.microsecond, tzinfo=timezone.utc)

            # 毫秒下的北京时间戳
            utc_timestamp_ms = int(utc_datetime.timestamp()*1000) # 该条记录的时间戳 以毫秒为单位 也就是13位

            # 经纬度 卫星数量 质量 hdop
            latitude = msg.latitude
            longitude = msg.longitude
            altitude = msg.altitude
            sats_num = msg.num_sats
            quality = msg.gps_qual
            hdop = msg.horizontal_dil

            to_write = f"{utc_timestamp_ms},{latitude},{longitude},{altitude},{sats_num},{quality},{hdop}\n"
            gngga_f.write(to_write)
            
    # pdb.set_trace()
    gngga_f.close()




def convert_one_dir(trip_dir):
    print(f"Converting {DATA_DIR}/{trip_dir}")
    gnss_file_name = None
    rtk_file_name = None
    folder = os.path.join(trip_dir, "supplementary")
    info = get_info(folder)
    gnss_file_name = info['txt']
    convert_GNSS_log(trip_dir, gnss_file_name)
    rtk_file_name = info['rtklite']
    convert_RTKLite_log(trip_dir, rtk_file_name)

    if not gnss_file_name:
        print(f"[ERROR] file_dir does not contain gnss_log.txt")

    if not rtk_file_name:
        print(f"[ERROR] file_dir does not contain gngga.txt")

def all_convert():
    phone_dirs = read_file('devices.txt')
    for phone_dir in phone_dirs:
        for trip_dir in os.listdir(f"{DATA_DIR}/{phone_dir}"):
            if os.path.isdir(f"{DATA_DIR}/{phone_dir}/{trip_dir}"):
                if not os.path.exists(f"{DATA_DIR}/{phone_dir}/{trip_dir}/Raw.csv") or ALL_CONVERT_OVERRIDE_FLAG:
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