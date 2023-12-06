# -*- coding: UTF-8 -*-
#根据提取出的各种csv整理形成数据集

import os, argparse
import ipdb as pdb

import numpy as np
np.set_printoptions(precision=10, suppress=True, formatter={'float_kind':'{:f}'.format})

import pandas as pd
pd.set_option('display.float_format','{:.6f}'.format)

from scipy.spatial.transform import Rotation as R
import pymap3d as pm
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12.0, 3.0]

from mtools import read_file, str2bool
from tools.load_tools import *
from tools.tools import magnetic_calibration, do_lowpass, get_mag_rot, rerange_deg, moving_average_filter

DATA_DIR = "processed"
not_neg_rollDeg_devices = []

def _drop_duplicates(df):
    subset = []
    if 'utcTimeMillis' in df.columns:
        subset.append('utcTimeMillis')
    if 'elapsedRealtimeNanos' in df.columns:
        subset.append('elapsedRealtimeNanos')
    return df.drop_duplicates(subset=subset, keep='last')

def _load_Logger_csv(name, phone_dir, trip_dir):
    return load_Logger_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/{name}.csv")

def _check_total_time(dfs):
    # 选择这几个传感器 第一个时间戳最晚的那个 和 最后一个时间戳最早的那个 作为整体的起始和结束时间戳, 并对齐到 10ms
    time_start = max([i.index.min() for i in dfs]) # dop_enu_df
    time_end = min([i.index.max() for i in dfs]) # dop_enu_df
    time_start = np.int64(np.ceil(time_start/10)*10)
    time_end = np.int64(np.floor(time_end/10)*10)

    print(f"Total: {(time_end-time_start)/60000:.2f} min Data")
    # 时间过少就停下来看一看
    if time_end-time_start < 100000:
        print([i.index.min() for i in dfs]) # dop_enu_df
        print([i.index.max() for i in dfs]) # dop_enu_df
        if DEBUGLEVEL<1:
            pdb.set_trace()
        if time_end-time_start < 0:
            print('time_end-time_start < 0')
            pdb.set_trace()
    return time_start, time_end

def _reindex_and_interpolate(df, desired_index):
    return reindex_and_interpolate(df, desired_index, DEBUGLEVEL)

def get_and_save_dataset_indoor(phone_dir, trip_dir):
    # 读取所有的传感器数据
    log_names = ['UncalAccel', 'UncalGyro', 'UncalMag', 'Rot', 'GameRot', 'Mark']
    acc_df, gys_df, mag_df, rot_df, game_df, mark_df = [_load_Logger_csv(name, phone_dir, trip_dir) for name in log_names]

    # 对 IMU 数据施加校准
    data_names = ("Acc", "Gys", "Mag")
    acc_df, gys_df, mag_df = [apply_IMU_bias(df, name) for df, name in zip((acc_df, gys_df, mag_df), data_names)]

    # 去除可能存在的重复utcTimeMillis
    acc_df, gys_df, mag_df, rot_df, game_df = tuple(map(_drop_duplicates, (acc_df, gys_df, mag_df, rot_df, game_df)))
    
    # 合并所有elapsedRealtimeNanos，utcTimeMillis，用于时间戳对齐
    all_time_df = pd.concat([df[['elapsedRealtimeNanos','utcTimeMillis']] for df in (acc_df, gys_df, mag_df, rot_df, game_df)])
    all_time_df.reset_index(names=['index'], inplace=True)
    all_time_df.sort_values('elapsedRealtimeNanos', inplace=True)
    all_time_df.set_index('index', inplace=True)

    # 对齐时间戳
    acc_df, gys_df, mag_df, rot_df, game_df, mark_df = [align_utc_ela(df, all_time_df) for df in (acc_df, gys_df, mag_df, rot_df, game_df, mark_df)]

    time_start, time_end = _check_total_time([acc_df, gys_df, mag_df, rot_df, game_df])

    # 同步后的采样时间戳 以10ms为间隔, 即100Hz
    desired_index = pd.RangeIndex(time_start, time_end, 10, name="timestamp")

    # 只保留需要的列, 然后进行插值
    (acc_df, gys_df, mag_df, rot_df, game_df) = [_reindex_and_interpolate(df, desired_index) for df in (acc_df, gys_df, mag_df, rot_df, game_df)]

    # 计算重力
    gra = do_lowpass(acc_df[['AccX', 'AccY', 'AccZ']].values, fc=1)
    acc_df[['GraX', 'GraY', 'GraZ']] = gra

    # 使用地磁计算rot
    mag_cali = magnetic_calibration(mag_df[['MagX', 'MagY', 'MagZ', 'UMagX', 'UMagY', 'UMagZ']].values)
    mag_df[['CMagX', 'CMagY', 'CMagZ']] = mag_cali

    # mag_rot = get_mag_gra_rot(mag_cali, gra)
    mag_rot = get_mag_rot(mag_cali, rot_df[['rollDeg', 'pitchDeg', 'yawDeg']].values)

    mag_euler = mag_rot.as_euler('yxz', degrees=True)
    mag_euler[:, 2] -= MAG_DECLINATION
    mag_df[['quaternionX','quaternionY','quaternionZ','quaternionW']] = R.from_euler('yxz', mag_euler, degrees=True).as_quat()
    mag_df[['rollDeg', 'pitchDeg', 'yawDeg']] = mag_euler    

    rot_euler = R.from_quat(rot_df[['quaternionX','quaternionY','quaternionZ','quaternionW']]).as_euler('yxz', degrees=True)
    rot_euler[:, 2] -= MAG_DECLINATION
    rot_df[['quaternionX','quaternionY','quaternionZ','quaternionW']] = R.from_euler('yxz', rot_euler, degrees=True).as_quat()
    rot_df[['rollDeg', 'pitchDeg', 'yawDeg']] = rot_euler

    game_euler = R.from_quat(game_df[['quaternionX','quaternionY','quaternionZ','quaternionW']]).as_euler('yxz', degrees=True)
    game_df[['rollDeg', 'pitchDeg', 'yawDeg']] = game_euler
    game_df['yawDeg'] += np.mean(rerange_deg(rot_df['yawDeg'][:100] - game_df['yawDeg'][:100]))

    OriSource = 'Game'

    # 施加旋转矩阵
    rot = R.from_quat(game_df[['quaternionX','quaternionY','quaternionZ','quaternionW']])
    acc_enu = rot.apply(acc_df[['AccX', 'AccY', 'AccZ']])
    gra_enu = rot.apply(acc_df[['GraX', 'GraY', 'GraZ']])
    gys_enu = rot.apply(gys_df[['GysX', 'GysY', 'GysZ']])
    mag_enu = rot.apply(mag_df[['MagX', 'MagY', 'MagZ']])
    acc_df[['AccE', 'AccN', 'AccU']] = acc_enu
    acc_df[['GraE', 'GraN', 'GraU']] = gra_enu
    gys_df[['GysE', 'GysN', 'GysU']] = gys_enu
    mag_df[['MagE', 'MagN', 'MagU']] = mag_enu
    acc_df['OriSource'] = OriSource
    gys_df['OriSource'] = OriSource
    mag_df['OriSource'] = OriSource
    plot_acc(acc_df, phone_dir, trip_dir)
    
    dfs = (acc_df, gys_df, mag_df, rot_df, game_df, mark_df)
    dfs_names = ("acc", "gys", "mag", "rot", "gamerot", "mark")
    save_data_h5(DATA_DIR, phone_dir, trip_dir, dfs, dfs_names)

    yaws = np.column_stack((game_df['yawDeg'], rot_df['yawDeg'], mag_df['yawDeg']))
    plot_yaw(yaws, ['GameRot', 'Rot', 'Mag'], phone_dir, trip_dir, slice(1, None, None))
    return (time_end-time_start)/1000

def get_and_save_dataset(phone_dir, trip_dir):
    # 读取所有的传感器数据
    log_names = ['UncalAccel', 'UncalGyro', 'UncalMag', 'OrientationDeg', 'Fix', 'GNGGA']
    acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df = [_load_Logger_csv(name, phone_dir, trip_dir) for name in log_names]
    if ALLSENSOR:
        rot_df, game_df = [_load_Logger_csv(name, phone_dir, trip_dir) for name in ['Rot', 'GameRot']]
    dop_ecef_df = load_Doppler_csv(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/doppler.csv", fix_df)

    # 对 IMU 数据施加校准
    data_names = ("Acc", "Gys", "Mag")
    acc_df, gys_df, mag_df = [apply_IMU_bias(df, name) for df, name in zip((acc_df, gys_df, mag_df), data_names)]

    # 去除可能存在的重复utcTimeMillis
    acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df = tuple(map(_drop_duplicates, (acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df)))
    if ALLSENSOR:
        rot_df, game_df = tuple(map(_drop_duplicates, (rot_df, game_df)))
    
    # 筛选fix来源为GPS
    fix_df = fix_df.query(f"Provider == 'GPS'").drop("Provider", axis=1)

    # 将 LLA 转换为 ENU
    pos_e, pos_n, pos_u = pm.geodetic2enu(fix_df['LatitudeDegrees'], fix_df['LongitudeDegrees'], fix_df['AltitudeMeters'], ENU_BASE[0], ENU_BASE[1], ENU_BASE[2])
    fix_df = fix_df.assign(PosE=pos_e, PosN=pos_n, PosU=pos_u)

    pos_e, pos_n, pos_u = pm.geodetic2enu(gngga_df['LatitudeDegrees'], gngga_df['LongitudeDegrees'], gngga_df['AltitudeMeters'], ENU_BASE[0], ENU_BASE[1], ENU_BASE[2])
    gngga_df = gngga_df.assign(PosE=pos_e, PosN=pos_n, PosU=pos_u)

    # 删除 Fix 前 18s 的数据 以及不是GPS给出的定位结果
    try:
        true_time_start = fix_df.iloc[0].utcTimeMillis + PREPARE_TIME*1000
    except:
        pdb.set_trace()
        if len(fix_df)==0:
            return
    fix_df = fix_df.query(f"utcTimeMillis >= {true_time_start}")

    if (gngga_df['Quality'].ne(4) & gngga_df['Quality'].ne(5)).any():
        gngga_valid_df = gngga_df.query(f"Quality == 4 or Quality == 5")
        print(f"[WARNING] not all RTKLite Quality is 4 or 5 so Deleting {len(gngga_df)-len(gngga_valid_df)} rows of RTKLite Data")
        gngga_df = gngga_valid_df
    
    # 将 ECEF 测速结果转换为 ENU
    dop_enu_df = get_dop_enu_df(dop_ecef_df)
    
    # 对齐时间戳
    acc_df, gys_df, mag_df, ori_df = [align_utc_ela(df, fix_df) for df in (acc_df, gys_df, mag_df, ori_df)]
    if ALLSENSOR:
        rot_df, game_df = [align_utc_ela(df, fix_df) for df in (rot_df, game_df)]
    
    fix_df['utcTimeMillis'] = (np.round(fix_df['utcTimeMillis']/10)*10).astype('int64')
    fix_df.set_index('utcTimeMillis', drop=False, inplace=True)
    gngga_df.set_index('utcTimeMillis', drop=False, inplace=True)

    if ALLSENSOR:
        time_start, time_end = _check_total_time([acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df, rot_df, game_df])
    else:
        time_start, time_end = _check_total_time([acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df])

    # 同步后的采样时间戳 以10ms为间隔, 即100Hz
    desired_index = pd.RangeIndex(time_start, time_end, 10, name="timestamp")

    # 只保留需要的列, 然后进行插值
    acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df, dop_enu_df = [_reindex_and_interpolate(df, desired_index) for df in \
        (acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df, dop_enu_df)]

    if np.sum(fix_df['Source']=='Sensor') == 0:
        pdb.set_trace()
    if ALLSENSOR:
        rot_df, game_df = [_reindex_and_interpolate(df, desired_index) for df in (rot_df, game_df)]

    # 需要对 rollDeg 进行取反, 这样才是真正的旋转矩阵
    if phone_dir not in not_neg_rollDeg_devices:
        ori_df.loc[:, 'rollDeg'] = -ori_df.loc[:, 'rollDeg']

    # 修正磁偏角
    ori_df.loc[:, 'yawDeg'] = ori_df.loc[:, 'yawDeg'] + MAG_DECLINATION
    ori_df[['rollDeg', 'pitchDeg', 'yawDeg']] = - rerange_deg(ori_df[['rollDeg', 'pitchDeg', 'yawDeg']])
    ori_df[['quaternionX','quaternionY','quaternionZ','quaternionW']] = R.from_euler('yxz', ori_df[['rollDeg', 'pitchDeg', 'yawDeg']], degrees=True).as_quat()
    
    # 计算重力
    gra = do_lowpass(acc_df[['AccX', 'AccY', 'AccZ']].values, fc=1)
    acc_df[['GraX', 'GraY', 'GraZ']] = gra

    # 使用地磁计算rot
    mag_cali = magnetic_calibration(mag_df[['MagX', 'MagY', 'MagZ', 'UMagX', 'UMagY', 'UMagZ']].values)
    mag_df[['CMagX', 'CMagY', 'CMagZ']] = mag_cali

    # mag_rot = get_mag_gra_rot(mag_cali, gra)
    mag_rot = get_mag_rot(mag_cali, ori_df[['rollDeg', 'pitchDeg', 'yawDeg']].values)

    mag_euler = mag_rot.as_euler('yxz', degrees=True)
    mag_euler[:, 2] -= MAG_DECLINATION
    mag_df[['quaternionX','quaternionY','quaternionZ','quaternionW']] = R.from_euler('yxz', mag_euler, degrees=True).as_quat()
    mag_df[['rollDeg', 'pitchDeg', 'yawDeg']] = mag_euler    

    if ALLSENSOR:
        rot_euler = R.from_quat(rot_df[['quaternionX','quaternionY','quaternionZ','quaternionW']]).as_euler('yxz', degrees=True)
        rot_euler[:, 2] -= MAG_DECLINATION
        rot_df[['quaternionX','quaternionY','quaternionZ','quaternionW']] = R.from_euler('yxz', rot_euler, degrees=True).as_quat()
        rot_df[['rollDeg', 'pitchDeg', 'yawDeg']] = rot_euler

        game_euler = R.from_quat(game_df[['quaternionX','quaternionY','quaternionZ','quaternionW']]).as_euler('yxz', degrees=True)
        game_df[['rollDeg', 'pitchDeg', 'yawDeg']] = game_euler
        game_df['yawDeg'] += np.mean(rerange_deg(rot_df['yawDeg'][:100] - game_df['yawDeg'][:100]))

    enu_pos = np.column_stack((moving_average_filter(gngga_df['PosE'], 10), moving_average_filter(gngga_df['PosN'], 10)))
    gngga_yaws = rerange_deg(np.rad2deg(np.arctan2(enu_pos[1:, 1]-enu_pos[:-1, 1], enu_pos[1:, 0]-enu_pos[:-1, 0]))-90)
    gngga_yaws = np.concatenate((gngga_yaws, (gngga_yaws[0],)))
    gngga_df['yawDeg'] = gngga_yaws

    if ALLSENSOR:
        yaw_srcs = ['GameRot', 'Ori', 'Rot', 'Mag', 'RTKLite']
        yaws = np.column_stack((game_df['yawDeg'], ori_df['yawDeg'], rot_df['yawDeg'], mag_df['yawDeg'], gngga_df['yawDeg']))
        yaw_err = plot_yaw(yaws, yaw_srcs, phone_dir, trip_dir, slice(1, None, None))
    else:
        # if phone_dir == 'Mi8' and trip_dir == '01_12_12_22':
        #     pdb.set_trace()
        yaw_srcs = ['Ori', 'Mag', 'RTKLite']
        yaws = np.column_stack((ori_df['yawDeg'], mag_df['yawDeg'], gngga_df['yawDeg']))
        yaw_err = plot_yaw(yaws, yaw_srcs, phone_dir, trip_dir, slice(None, None, None))

    # 施加旋转矩阵
    if ORI_SOURCE == 0:
        OriSource = yaw_srcs[np.argmin(yaw_err)]
        if OriSource == 'GameRot':
            rot_source_df = game_df
        elif OriSource == 'Rot':
            rot_source_df = rot_df
        elif OriSource == 'Mag':
            rot_source_df = mag_df
        elif OriSource == 'Ori':
            rot_source_df = ori_df
    elif ORI_SOURCE == 1:
        rot_source_df = rot_df; OriSource = 'Rot'
    elif ORI_SOURCE == 2:
        rot_source_df = mag_df; OriSource = 'Mag'
    elif ORI_SOURCE == 3:
        rot_source_df = ori_df; OriSource = 'Ori'
    print(f'Select OriSource: {OriSource}')

    rot_source_quat = rot_source_df[['quaternionX','quaternionY','quaternionZ','quaternionW']]
    # rot_source_euler = R.from_quat(rot_source_quat).as_euler('yxz', degrees=True)
    rot = R.from_quat(rot_source_quat)
    acc_enu = rot.apply(acc_df[['AccX', 'AccY', 'AccZ']])
    gra_enu = rot.apply(acc_df[['GraX', 'GraY', 'GraZ']])
    gys_enu = rot.apply(gys_df[['GysX', 'GysY', 'GysZ']])
    mag_enu = rot.apply(mag_df[['MagX', 'MagY', 'MagZ']])
    acc_df[['AccE', 'AccN', 'AccU']] = acc_enu
    acc_df[['GraE', 'GraN', 'GraU']] = gra_enu
    gys_df[['GysE', 'GysN', 'GysU']] = gys_enu
    mag_df[['MagE', 'MagN', 'MagU']] = mag_enu
    acc_df['OriSource'] = OriSource
    gys_df['OriSource'] = OriSource
    mag_df['OriSource'] = OriSource
    plot_acc(acc_df, phone_dir, trip_dir)
    
    if ALLSENSOR:
        dfs = (acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df, dop_enu_df, rot_df, game_df)
        dfs_names = ("acc", "gys", "mag", "ori", "fix", "gngga", "dop", "rot", "gamerot")
    else:
        dfs = (acc_df, gys_df, mag_df, ori_df, fix_df, gngga_df, dop_enu_df)
        dfs_names = ("acc", "gys", "mag", "ori", "fix", "gngga", "dop")
    
    plt.close('all')
    save_data_h5(DATA_DIR, phone_dir, trip_dir, dfs, dfs_names)
    return (time_end-time_start)/1000

def plot_yaw(yaws, columns, phone_dir, trip_dir, stable_slice=None, freq=100):
    yaws = rerange_deg(yaws)
    yaws[yaws<0] = yaws[yaws<0] + 360
    yaw_df = pd.DataFrame(yaws, columns=columns)
    g = yaw_df[::freq].plot()
    g.grid()
    plt.savefig(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/yaw.png")
    yaw_err = None
    if not INDOOR:
        yaws_diff = rerange_deg(yaws[:, :-1]-yaws[:, -1:])
        stable_mask = np.any(np.abs(yaws_diff[:, stable_slice])<50, axis=-1)
        yaw_diff_df = pd.DataFrame(yaws_diff, columns=columns[:-1])
        g = yaw_diff_df[::freq].plot()
        g.grid()
        g.set_ylim([-50, 50])
        plt.savefig(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/yaw_diff.png")
        
        # yaws_diff_cali = yaws_diff - np.mean(yaws_diff[stable_mask], axis=0)
        # yaw_diff_cali_df = pd.DataFrame(yaws_diff_cali, columns=columns[:-1])
        # g = yaw_diff_cali_df[::freq].plot()
        # g.grid()
        # g.set_ylim([-50, 50])
        # plt.savefig(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/yaws_diff_cali.png")

        yaw_diff = np.mean(yaws_diff[stable_mask], axis=0)
        yaw_err = np.mean(np.abs(yaws_diff[stable_mask]), axis=0)
        yaw_std = np.std(yaws_diff[stable_mask], axis=0)
        data = np.row_stack((yaw_diff, yaw_err, yaw_std))
        np.savetxt(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/yaw_diff_stat.csv", data, fmt='%.2f', delimiter=',')
    return yaw_err

def plot_acc(acc_df, phone_dir, trip_dir, plot_time=60, freq=100):
    acc_df[:plot_time*freq][['AccX', 'AccY', 'AccZ']].plot()
    plt.savefig(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/accXYZ.png")
    acc_df[:plot_time*freq][['AccE', 'AccN', 'AccU']].plot()
    plt.savefig(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/accENU.png")
    acc_df[:plot_time*freq][['GraX', 'GraY', 'GraZ']].plot()
    plt.savefig(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/graXYZ.png")
    acc_df[:plot_time*freq][['GraE', 'GraN', 'GraU']].plot()
    plt.savefig(f"{DATA_DIR}/{phone_dir}/{trip_dir}/supplementary/graENU.png")

def generate_all_h5():
    phone_dirs = sorted(read_file('devices.txt'))
    phone_dirs = list(filter(lambda x: not x.startswith('#'), phone_dirs))
    for phone_dir in phone_dirs:
        if not os.path.isdir(f"{DATA_DIR}/{phone_dir}"): 
            continue
        for trip_dir in sorted(os.listdir(f"{DATA_DIR}/{phone_dir}")):
            if not os.path.isdir(f"{DATA_DIR}/{phone_dir}/{trip_dir}"):
                continue
            if os.path.exists(f"{DATA_DIR}/{phone_dir}/{trip_dir}/data.h5") and not OVERRIDE_FLAG:
                continue
            else:
                print(f"#### {phone_dir}/{trip_dir} ####")
                try:
                    if INDOOR:
                        get_and_save_dataset_indoor(phone_dir, trip_dir)
                    else:
                        get_and_save_dataset(phone_dir, trip_dir)
                except:
                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path',    type=str)
    parser.add_argument('-o', '--overwrite',    type=str2bool, default=True)
    parser.add_argument('-t', '--preparetime',  type=int,      default=18)
    parser.add_argument('-i', '--indoor',       type=int,      default=0)
    parser.add_argument('-s', '--allsensor',    type=int,      default=1)
    parser.add_argument('-os', '--ori_source',  type=int,      default=0) # 0: auto min, 1: rot, 2: mag, 3: ori
    parser.add_argument('-dl', '--debuglevel',  type=int,      default=0)
    args = parser.parse_args()
    IGR_DIR = args.data_path
    OVERRIDE_FLAG = args.overwrite
    PREPARE_TIME = args.preparetime
    INDOOR = args.indoor
    ALLSENSOR = args.allsensor
    DEBUGLEVEL = args.debuglevel
    ORI_SOURCE = args.ori_source
    os.chdir(os.path.join('IGRData', IGR_DIR))
    generate_all_h5()
