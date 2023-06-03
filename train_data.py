# -*- coding: UTF-8 -*-
"""
Usage:
  train_data.py <data_path> [options]

Options:
  -o <overwrite>, --overwrite <overwrite>       overwrite   [default: 1]
  -i <indoor>, --indoor <indoor>                indoor      [default: 0]
  -t <preparetime>, --preparetime <preparetime> prepare     [default: 18]
  -d <debuglevel>, --debuglevel <debuglevel>    debuglevel  [default: 0]
"""

#根据提取出的各种csv整理形成数据集

import os
import ipdb as pdb
from docopt import docopt

import numpy as np
np.set_printoptions(precision=10, suppress=True, formatter={'float_kind':'{:f}'.format})

import pandas as pd
pd.set_option('display.float_format','{:.6f}'.format)

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from scipy.spatial.transform import Rotation as R
import pymap3d as pm
import h5py

from mtools import monkey as mk
from mtools import read_file, write_file

DATA_DIR = "processed"

USELESS_COLUMNS = set((
    'UncalAccel',
    'UncalGyro',
    'UncalMag',
    'OrientationDeg',
    'Fix'
))

HEADER_DEF = {
    "Raw": "Raw,utcTimeMillis,TimeNanos,LeapSecond,TimeUncertaintyNanos,FullBiasNanos,BiasNanos,BiasUncertaintyNanos,DriftNanosPerSecond,DriftUncertaintyNanosPerSecond,HardwareClockDiscontinuityCount,Svid,TimeOffsetNanos,State,ReceivedSvTimeNanos,ReceivedSvTimeUncertaintyNanos,Cn0DbHz,PseudorangeRateMetersPerSecond,PseudorangeRateUncertaintyMetersPerSecond,AccumulatedDeltaRangeState,AccumulatedDeltaRangeMeters,AccumulatedDeltaRangeUncertaintyMeters,CarrierFrequencyHz,CarrierCycles,CarrierPhase,CarrierPhaseUncertainty,MultipathIndicator,SnrInDb,ConstellationType,AgcDb,BasebandCn0DbHz,FullInterSignalBiasNanos,FullInterSignalBiasUncertaintyNanos,SatelliteInterSignalBiasNanos,SatelliteInterSignalBiasUncertaintyNanos,CodeType,ChipsetElapsedRealtimeNanos",
    "UncalAccel": "UncalAccel,utcTimeMillis,elapsedRealtimeNanos,UncalAccelXMps2,UncalAccelYMps2,UncalAccelZMps2,BiasXMps2,BiasYMps2,BiasZMps2",
    "UncalGyro": "UncalGyro,utcTimeMillis,elapsedRealtimeNanos,UncalGyroXRadPerSec,UncalGyroYRadPerSec,UncalGyroZRadPerSec,DriftXRadPerSec,DriftYRadPerSec,DriftZRadPerSec",
    "UncalMag": "UncalMag,utcTimeMillis,elapsedRealtimeNanos,UncalMagXMicroT,UncalMagYMicroT,UncalMagZMicroT,BiasXMicroT,BiasYMicroT,BiasZMicroT",
    "OrientationDeg": "OrientationDeg,utcTimeMillis,elapsedRealtimeNanos,yawDeg,rollDeg,pitchDeg",
    "Fix": "Fix,Provider,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SpeedMps,AccuracyMeters,BearingDegrees,UnixTimeMillis,SpeedAccuracyMps,BearingAccuracyDegrees,elapsedRealtimeNanos,VerticalAccuracyMeters",
    "GNGGA": "utcTimeMillis,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SatNum,Quality,hdop",
}

# not_neg_rollDeg_devices = ['Mi11', 'RedmiK40']
not_neg_rollDeg_devices = []

# 天津大学磁偏角
MAG_DECLINATION = -7.516667

# ENU 基准坐标点
ENU_BASE = [38.9961, 117.3050, 2]

def load_GNSSLogger_csv(csv_path:str) -> pd.DataFrame:
    # 读取csv, 删除无用列
    df = pd.read_csv(csv_path)
    df = df.drop(columns=set(df.columns).intersection(USELESS_COLUMNS), axis=1)
    # 不同的数据类型时间戳命名不一样 只有 Fix 时间戳那一列的名称是 UnixTimeMillis 其余的均为 utcTimeMillis
    if 'UnixTimeMillis' in df.columns.to_list():
        df.rename(columns={"UnixTimeMillis": "utcTimeMillis"}, inplace=True)
    if 'elapsedRealtimeNanos' in df.columns.to_list():    
        df = df.set_index('elapsedRealtimeNanos', drop=False)
    if len(df):
        assert(all(df.index.duplicated())==False)
    return df

'''
    加载 Doppler 测速结果的CSV
    INPUT: csv_path
    OUTPUT: pd.Dataframe index=utcTimeMillis data=[utcTimeMillis, vx, vy, vz] 是 ECEF 坐标系下的
'''
def load_Doppler_csv(csv_path:str, fix_df:pd.DataFrame) -> pd.DataFrame:
    # 读取csv 
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 修改时间戳名称
        df.rename(columns={"timestampUTC": "utcTimeMillis"}, inplace=True)
        # 设置时间戳为索引
        df['utcTimeMillis'] = (np.ceil(df['utcTimeMillis']/1e3)*1e3).astype("int64")
        
        df.drop_duplicates(subset=['utcTimeMillis'], keep='last', inplace=True)
        df.set_index('utcTimeMillis', drop=False, inplace=True)
        assert(all(df.index.duplicated())==False)
    else:
        df = get_empty_dop_df(fix_df['utcTimeMillis'], len(fix_df))
    return df

'''
    将 ECEF 下的多普勒测速结果 转换为 ENU 坐标系下
    同时将时间戳粒度控制在10ms

    In: dop_ecef_df: ECEF 坐标系下的 多普勒测速结果DF, 由 load_Doppler_csv 给出
    Out: dop_enu_df: ENU 坐标系下的 多普勒测速结果DF index=utcTimeMillis data=[utcTimeMillis, ve, vn, vu]
'''
def get_dop_enu_df(dop_ecef_df:pd.DataFrame):
    dop_ecef = dop_ecef_df[['utcTimeMillis', 'vx', 'vy', 'vx']].values # (N, 4): [utcTimeMillis, vx, vy, vz]

    # 起始位置的ECEF坐标
    ORI_BASE_ECEF = pm.geodetic2ecef(ENU_BASE[0], ENU_BASE[1], ENU_BASE[2])

    # 将ECEF速度转换到ENU的速度
    dop_ecef_dest = dop_ecef[:, 1:4] + ORI_BASE_ECEF
    dop_ve, dop_vn, dop_vu = pm.ecef2enu(dop_ecef_dest[:, 0], dop_ecef_dest[:, 1], dop_ecef_dest[:, 2], ENU_BASE[0], ENU_BASE[1], ENU_BASE[2])

    # dop_enu_df: [utcTimeMillis, ve, vn, vu]
    dop_enu_df = pd.DataFrame(data={
        "utcTimeMillis": ((dop_ecef[:, 0]/10).round(0)*10).astype("int64"),
        "ve": dop_ve,
        "vn": dop_vn,
        "vu": dop_vu,
    })

    dop_enu_df.set_index("utcTimeMillis", drop=False, inplace=True)
    return dop_enu_df


def apply_IMU_bias(df:pd.DataFrame, name:str):
    # 对于需要校准的 acc gys 和 mag 数据 他们都是 [timestamp, rawx, rawy, rawz, biasx, biasy, biasz] 的列顺序
    # 因此 只要把后三列 加在 中间三列上就行了
    calibrated_data = df.iloc[:, 2:5].values + df.iloc[:, -3:].values

    calibrated_df = pd.DataFrame(
        data = {
            'utcTimeMillis': df['utcTimeMillis'],
            'elapsedRealtimeNanos': df['elapsedRealtimeNanos'],
            f"{name}X": calibrated_data[:, 0],
            f"{name}Y": calibrated_data[:, 1],
            f"{name}Z": calibrated_data[:, 2],
        }, 
        index=df.index
    )
    
    return calibrated_df

def get_empty_dop_df(index, df_len):
    return pd.DataFrame({
            "timestamp": index.values,
            "utcTimeMillis": index.values,
            "vx":np.zeros((df_len,)),
            "vy":np.zeros((df_len,)),
            "vz":np.zeros((df_len,)) 
        })

def align_utc_ela(df, fix_df):
    desired_ela_index = fix_df.index.union(df['elapsedRealtimeNanos']).unique()
    df.reindex(desired_ela_index)
    df['utcTimeMillis'] = fix_df['utcTimeMillis'].reindex(desired_ela_index).interpolate('index', limit_area='inside')
    df.set_index('utcTimeMillis', drop=False, inplace=True)
    df.dropna(subset=['utcTimeMillis'], inplace=True)
    df['utcTimeMillis'] = np.round(df['utcTimeMillis'].values).astype('int64')
    return df

def reindex_and_interpolate(df:pd.DataFrame, desired_index:pd.Index):
    # 标记数据是来自传感器还是插值
    df.loc[:, 'Source'] = 'Sensor'
    # 将两种index合并起来, 这个合并能够保证包含两个index的所有项并且不重复
    all_index = df.index.values
    m = np.zeros_like(all_index, dtype=bool)
    m[np.unique(all_index, return_index=True)[1]] = True
    dup_index = all_index[~m]
    if len(dup_index):
        print(dup_index)
        if DEBUGLEVEL<1:
            pdb.set_trace()
        df.drop_duplicates(['utcTimeMillis'], inplace=True)

    union_index = df.index.union(desired_index).dropna()
    
    # 首先将df上采样到union_index, 然后对空缺的地方插值
    target_df = df.reindex(union_index).interpolate('index')
    # 然后下采样到需要的index
    target_df = target_df.reindex(desired_index)
    target_df.loc[:, 'Source'] = target_df.loc[:, 'Source'].fillna('Inter')

    # 检查重采样结果 这个是历史遗留问题, 能够保证进行的插值是线性插值
    # 保留着更保险一点 检查完就扔掉
    if 'utcTimeMillis' in target_df.columns:
        if all(np.round(target_df['utcTimeMillis']).astype("int64") == target_df.index) == False:
            print("[WARNING] timestamp diff std() != 0")
            # pdb.set_trace()
        target_df.drop(columns='utcTimeMillis', inplace=True)
        
    return target_df

def get_and_save_dataset_indoor(phone_dir, trip_dir, PREPARE_TIME):
    # 读取所有的传感器数据
    acc_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/UncalAccel.csv")
    gys_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/UncalGyro.csv")
    rot_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/GameRot.csv")
    mark_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/Mark.csv")

    # 对 IMU 数据施加校准
    acc_df = apply_IMU_bias(acc_df, "Acc")
    gys_df = apply_IMU_bias(gys_df, "Gys")

    # acc_df.set_index('utcTimeMillis', drop=False, inplace=True)
    # gys_df.set_index('utcTimeMillis', drop=False, inplace=True)
    # rot_df.set_index('utcTimeMillis', drop=False, inplace=True)

    rot_df.drop_duplicates(subset=['utcTimeMillis'], keep='last', inplace=True)

    all_time_df = pd.concat([rot_df[['elapsedRealtimeNanos','utcTimeMillis']], acc_df[['elapsedRealtimeNanos','utcTimeMillis']], gys_df[['elapsedRealtimeNanos','utcTimeMillis']]])
    all_time_df.reset_index(names=['index'], inplace=True)
    all_time_df.sort_values('elapsedRealtimeNanos', inplace=True)
    all_time_df.set_index('index', inplace=True)

    acc_df = align_utc_ela(acc_df, all_time_df)
    gys_df = align_utc_ela(gys_df, all_time_df)
    rot_df = align_utc_ela(rot_df, all_time_df)
    if len(mark_df):
        mark_df = align_utc_ela(mark_df, all_time_df)

    # 选择这几个传感器 第一个时间戳最晚的那个 和 最后一个时间戳最早的那个 作为整体的起始和结束时间戳, 并对齐到 10ms
    time_start = max([i.index.min() for i in [acc_df, gys_df, rot_df]])
    time_end   = min([i.index.max() for i in [acc_df, gys_df, rot_df]])
    time_start = np.int64(np.ceil(time_start/10)*10)
    time_end   = np.int64(np.floor(time_end/10)*10)

    print(f"Total: {(time_end-time_start)/60000:.2f} min Data")

    # 时间过少就停下来看一看
    if time_end-time_start < 100000:
        print([i.index.min() for i in [acc_df, gys_df, rot_df]]) # dop_enu_df
        print([i.index.max() for i in [acc_df, gys_df, rot_df]]) # dop_enu_df
        if time_end-time_start < 0:
            print('time_end-time_start < 0')
            return
        

    # 同步后的采样时间戳 以10ms为间隔, 即100Hz
    desired_index = pd.RangeIndex(time_start, time_end, 10, name="timestamp")
    # desired_index = pd.date_range(time_start, time_end, freq='10L')

    # 只保留需要的列, 然后进行插值
    acc_df = reindex_and_interpolate(acc_df, desired_index)
    gys_df = reindex_and_interpolate(gys_df, desired_index)
    rot_df = reindex_and_interpolate(rot_df, desired_index)

    # 施加旋转矩阵
    rot = R.from_quat(rot_df[['quaternionX','quaternionY','quaternionZ','quaternionW']])
    acc_enu = rot.apply(acc_df[['AccX', 'AccY', 'AccZ']]) # , inverse=True
    gys_enu = rot.apply(gys_df[['GysX', 'GysY', 'GysZ']]) # , inverse=True

    acc_df = acc_df.assign(AccE=acc_enu[:,0], AccN=acc_enu[:,1], AccU=acc_enu[:,2])
    gys_df = gys_df.assign(GysE=gys_enu[:,0], GysN=gys_enu[:,1], GysU=gys_enu[:,2])

    dfs = (acc_df, gys_df, rot_df, mark_df)
    dfs_names = ("acc", "gys", "rot", "mark")
    save_data_h5(phone_dir, trip_dir, dfs, dfs_names)
    return (time_end-time_start)/1000

def get_and_save_dataset(phone_dir, trip_dir, PREPARE_TIME):
    # 读取所有的传感器数据
    acc_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/UncalAccel.csv")
    gys_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/UncalGyro.csv")
    mag_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/UncalMag.csv")
    ori_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/OrientationDeg.csv")
    fix_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/Fix.csv")
    gngga_df = load_GNSSLogger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/GNGGA.csv")
    dop_ecef_df = load_Doppler_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/doppler.csv", fix_df)

    fix_df = fix_df.query(f"Provider == 'GPS'").drop("Provider", axis=1)
    gngga_df.drop_duplicates(['utcTimeMillis'], inplace=True)
    
    # 需要对 rollDeg 进行取反, 这样才是真正的旋转矩阵
    if phone_dir not in not_neg_rollDeg_devices:
        ori_df.loc[:, 'rollDeg'] = -ori_df.loc[:, 'rollDeg']

    # 将 LLA 转换为 ENU
    pos_e, pos_n, pos_u = pm.geodetic2enu(fix_df['LatitudeDegrees'], fix_df['LongitudeDegrees'], fix_df['AltitudeMeters'], ENU_BASE[0], ENU_BASE[1], ENU_BASE[2])
    fix_df = fix_df.assign(PosE=pos_e, PosN=pos_n, PosU=pos_u)

    pos_e, pos_n, pos_u = pm.geodetic2enu(gngga_df['LatitudeDegrees'], gngga_df['LongitudeDegrees'], gngga_df['AltitudeMeters'], ENU_BASE[0], ENU_BASE[1], ENU_BASE[2])
    gngga_df = gngga_df.assign(PosE=pos_e, PosN=pos_n, PosU=pos_u)

    # 对 IMU 数据施加校准
    acc_df = apply_IMU_bias(acc_df, "Acc")
    gys_df = apply_IMU_bias(gys_df, "Gys")
    mag_df = apply_IMU_bias(mag_df, "Mag")

    # 删除 Fix 前 18s 的数据 以及不是GPS给出的定位结果
    try:
        true_time_start = fix_df.iloc[0].utcTimeMillis + PREPARE_TIME*1000
    except:
        pdb.set_trace()
    fix_df = fix_df.query(f"utcTimeMillis >= {true_time_start}")

    # # 删除 GNGGA 中第一个 Quality != 4 和 5 的所有数据
    # if gngga_df['Quality'].ne(4).any():
    #     invalid_start = gngga_df.query("Quality != 4").iloc[0].utcTimeMillis
    #     gngga_valid_df = gngga_df.query(f"utcTimeMillis < {invalid_start}")
    #     print(f"[WARNING] not all RTKLite Quality is 4 so Deleting {len(gngga_df)-len(gngga_valid_df)} rows of RTKLite Data")
    #     gngga_df = gngga_valid_df
    
    # 将 ECEF 测速结果转换为 ENU
    dop_enu_df = get_dop_enu_df(dop_ecef_df)
    
    # 生成同步对齐时间戳
    acc_df = align_utc_ela(acc_df, fix_df)
    gys_df = align_utc_ela(gys_df, fix_df)
    mag_df = align_utc_ela(mag_df, fix_df)
    ori_df = align_utc_ela(ori_df, fix_df)

    fix_df.set_index('utcTimeMillis', drop=False, inplace=True)
    gngga_df.set_index('utcTimeMillis', drop=False, inplace=True)

    # 选择这几个传感器 第一个时间戳最晚的那个 和 最后一个时间戳最早的那个 作为整体的起始和结束时间戳, 并对齐到 10ms
    time_start = max([i.index.min() for i in [acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df]]) # dop_enu_df
    time_end = min([i.index.max() for i in [acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df]]) # dop_enu_df
    time_start = np.int64(np.ceil(time_start/10)*10)
    time_end = np.int64(np.floor(time_end/10)*10)

    print(f"Total: {(time_end-time_start)/60000:.2f} min Data")

    # 时间过少就停下来看一看
    if time_end-time_start < 100000:
        print([i.index.min() for i in [acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df]]) # dop_enu_df
        print([i.index.max() for i in [acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df]]) # dop_enu_df
        if DEBUGLEVEL<1:
            pdb.set_trace()
        if time_end-time_start < 0:
            print('time_end-time_start < 0')
            return
        

    # 同步后的采样时间戳 以10ms为间隔, 即100Hz
    desired_index = pd.RangeIndex(time_start, time_end, 10, name="timestamp")
    # desired_index = pd.date_range(time_start, time_end, freq='10L')

    # 只保留需要的列, 然后进行插值
    acc_df = reindex_and_interpolate(acc_df, desired_index)
    gys_df = reindex_and_interpolate(gys_df, desired_index)
    mag_df = reindex_and_interpolate(mag_df, desired_index)
    ori_df = reindex_and_interpolate(ori_df, desired_index)
    fix_df = reindex_and_interpolate(fix_df, desired_index)
    gngga_df = reindex_and_interpolate(gngga_df, desired_index)
    dop_enu_df = reindex_and_interpolate(dop_enu_df, desired_index)

    # 修正磁偏角
    ori_df.loc[:, 'yawDeg'] = ori_df.loc[:, 'yawDeg'] + MAG_DECLINATION

    # 施加旋转矩阵
    rot = R.from_euler('zxy', ori_df[['yawDeg', 'pitchDeg', 'rollDeg']], degrees=True)
    acc_enu = rot.apply(acc_df[['AccX', 'AccY', 'AccZ']], inverse=True)
    gys_enu = rot.apply(gys_df[['GysX', 'GysY', 'GysZ']], inverse=True)

    acc_df = acc_df.assign(AccE=acc_enu[:,0], AccN=acc_enu[:,1], AccU=acc_enu[:,2])
    gys_df = gys_df.assign(GysE=gys_enu[:,0], GysN=gys_enu[:,1], GysU=gys_enu[:,2])

    dfs = (acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df, dop_enu_df)
    dfs_names = ("acc", "gys", "mag", "ori", "fix", "gngga", "dop")
    save_data_h5(phone_dir, trip_dir, dfs, dfs_names)
    return (time_end-time_start)/1000

def save_data_h5(phone_dir, trip_dir, dfs, dfs_names):
    h5f_path = f"{DATA_DIR}/{phone_dir}/{trip_dir}/data.h5"
    if os.path.exists(h5f_path):
        print(f"Overriding {h5f_path}")
        os.remove(h5f_path)

    store = pd.HDFStore(f"{DATA_DIR}/{phone_dir}/{trip_dir}/data.h5")
    for df, df_name in zip(dfs, dfs_names):
        if df.index.name not in df.columns.tolist():
            df.reset_index(inplace=True)
        else:
            df.reset_index(inplace=True, names=['index'])
        store.put(df_name, df, index=False)
    store.close()

    print("\n")

def generate_all_h5(OVERRIDE_FLAG, PREPARE_TIME, INDOOR):
    phone_dirs = sorted(read_file('devices.txt'))
    for phone_dir in phone_dirs:
        if not os.path.isdir(f"{DATA_DIR}/{phone_dir}"): 
            continue
        for trip_dir in sorted(os.listdir(f"{DATA_DIR}/{phone_dir}")):
            if not os.path.isdir(f"{DATA_DIR}/{phone_dir}/{trip_dir}"):
                continue
            if os.path.exists(f"{DATA_DIR}/{phone_dir}/{trip_dir}/data.h5") and OVERRIDE_FLAG == 0:
                continue
            else:
                print(f"#### {phone_dir}/{trip_dir} ####")
                if INDOOR:
                    get_and_save_dataset_indoor(phone_dir, trip_dir, PREPARE_TIME)
                else:
                    get_and_save_dataset(phone_dir, trip_dir, PREPARE_TIME)

def main():
    generate_all_h5(OVERRIDE_FLAG, PREPARE_TIME, INDOOR)
    
if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    IGR_DIR = arguments.data_path
    OVERRIDE_FLAG = int(arguments.overwrite)
    PREPARE_TIME = int(arguments.preparetime)
    INDOOR = int(arguments.indoor)
    DEBUGLEVEL = int(arguments.debuglevel)
    os.chdir(os.path.join('IGRData', IGR_DIR))
    main()

# python train_data.py IGR_indoor_test -i 1


    # 画 rtk轨迹 并在上面画加速度方向
    # draw_rtk_df = gngga_df.query('Source=="Sensor"')[['PosE', 'PosN']]
    # draw_acc_df = acc_df.loc[draw_rtk_df.index, ['AccE', 'AccN']]
    # draw_rtk_np = draw_rtk_df.values
    # draw_acc_np = draw_acc_df.values
    # plt.figure(figsize=(8,6))
    # plt.plot(draw_rtk_np[:, 0], draw_rtk_np[:, 1], label="rtk")
    # for (pos, acc) in zip(draw_rtk_np, draw_acc_np):
    #     plt.annotate("", xy=(pos+acc), xytext=(pos), arrowprops=dict(arrowstyle="->", color="r")) # xytext（坐标） 指向 xy
    # plt.show()
    # exit()

'''
def apply_ori(imu_df):
    euler = imu_df[['yawDeg', 'rollDeg', 'pitchDeg']].values

    # # 应用磁偏角
    # euler[:, 0] += MAG_DECLINATION

    # 这个很难解释... 总之安卓给出的getOrientation方法坐标系不是相对于ENU的, 而是 -E -N -U 所以旋转需要反向一下
    # euler = -euler

    acc = imu_df[['UncalAccelXMps2', 'UncalAccelYMps2', 'UncalAccelZMps2']].values
    gys = imu_df[['UncalGyroXRadPerSec', 'UncalGyroYRadPerSec', 'UncalGyroZRadPerSec']].values

    rot = R.from_euler('zxy', euler, degrees=True)
    acc_enu = rot.apply(acc)
    gys_enu = rot.apply(gys)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(acc[:, 0], label='X')
    axes[0].plot(acc[:, 1], label='Y')
    axes[0].plot(acc[:, 2], label='Z')
    axes[0].legend()
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Acc (m/s2)")

    axes[1].plot(acc_enu[:, 0], label='E')
    axes[1].plot(acc_enu[:, 1], label='N')
    axes[1].plot(acc_enu[:, 2], label='U')
    axes[1].legend()
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Acc (m/s2)")


    fig.tight_layout()
    plt.show()
    plt.close(fig)
'''