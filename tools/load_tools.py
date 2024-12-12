import os
import ipdb as pdb
import pymap3d as pm
import numpy as np
np.set_printoptions(precision=10, suppress=True, formatter={'float_kind':'{:f}'.format})

import pandas as pd
pd.set_option('display.float_format','{:.6f}'.format)

USELESS_COLUMNS = set((
    'UncalAccel',
    'UncalGyro',
    'UncalMag',
    'OrientationDeg',
    'Fix',
    'GameRot',
    'Raw',
    'Rot',
    'Loc'
))

# 天津大学磁偏角
MAG_DECLINATION = -7.516667

# ENU 基准坐标点
ENU_BASE = [38.9961, 117.3050, 2]

'''
1. 载入csv文件
2. UnixTimeMillis->utcTimeMillis
3. 如果有elapsedRealtimeNanos，设为index
4. 保证index没有重复
'''
def load_Logger_csv(csv_path:str) -> pd.DataFrame:
    # 读取csv, 删除无用列
    df = pd.read_csv(csv_path)
    df = df.drop(columns=set(df.columns).intersection(USELESS_COLUMNS), axis=1)
    # 不同的数据类型时间戳命名不一样 只有 Fix 时间戳那一列的名称是 UnixTimeMillis 其余的均为 utcTimeMillis
    if 'UnixTimeMillis' in df.columns.to_list():
        df.rename(columns={"UnixTimeMillis": "utcTimeMillis"}, inplace=True)
    if 'elapsedRealtimeNanos' in df.columns.to_list():    
        df = df.set_index('elapsedRealtimeNanos', drop=False)
    if len(df):
        assert all(df.index.duplicated())==False
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
        assert all(df.index.duplicated())==False
    else:
        df = get_empty_dop_df(fix_df['utcTimeMillis'], len(fix_df))
    return df

def get_empty_dop_df(index, df_len):
    return pd.DataFrame({
            "timestamp": index.values,
            "utcTimeMillis": index.values,
            "vx":np.zeros((df_len,)),
            "vy":np.zeros((df_len,)),
            "vz":np.zeros((df_len,)) 
        })


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
    # 因此 只要把后三列 减去 中间三列上就行了
    uncalibrated_data = df.iloc[:, -6:-3].values
    bias_data = df.iloc[:, -3:].values
    calibrated_data = uncalibrated_data - bias_data

    calibrated_df = pd.DataFrame(
        data = {
            'utcTimeMillis': df['utcTimeMillis'],
            'elapsedRealtimeNanos': df['elapsedRealtimeNanos'],
            f"{name}X": calibrated_data[:, 0],
            f"{name}Y": calibrated_data[:, 1],
            f"{name}Z": calibrated_data[:, 2],
            f"U{name}X": uncalibrated_data[:, 0],
            f"U{name}Y": uncalibrated_data[:, 1],
            f"U{name}Z": uncalibrated_data[:, 2],
        }, 
        index=df.index
    )
    
    return calibrated_df

def align_utc_ela(df, fix_df):
    if len(df):
        desired_ela_index = fix_df.index.union(df['elapsedRealtimeNanos']).unique()
        df.reindex(desired_ela_index)
        df['utcTimeMillis'] = fix_df['utcTimeMillis'].reindex(desired_ela_index).interpolate('index', limit_area='inside')
        df.set_index('utcTimeMillis', drop=False, inplace=True)
        df.dropna(subset=['utcTimeMillis'], inplace=True)
        df['utcTimeMillis'] = np.round(df['utcTimeMillis'].values).astype('int64')
    return df

def save_data_h5(DATA_DIR, phone_dir, trip_dir, dfs, dfs_names):
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

def reindex_and_interpolate(df:pd.DataFrame, desired_index:pd.Index, DEBUGLEVEL):
    # 标记数据是来自传感器还是插值
    df = pd.DataFrame(df)
    df['Source'] = 'Sensor'

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

    # 检查重采样结果 这个是历史遗留问题, 能够保证进行的插值是线性插值, 保留着更保险一点
    # 只会在对多普勒速度插值时出现，原因时desired_index的范围超出对多普勒速度的范围
    if 'utcTimeMillis' in target_df.columns:
        try:
            if all(np.round(target_df['utcTimeMillis']).astype("int64") == target_df.index) == False:
                print("[WARNING] timestamp diff std() != 0")
        except:
            pdb.set_trace()
    return target_df