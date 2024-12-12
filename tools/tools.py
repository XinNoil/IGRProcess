import os, re, json, sys
import os.path as osp
from os.path import join as join_path
from mtools import list_con
import numpy as np
import ipdb as pdb
import datetime
from datetime import timezone
import pynmea2
from scipy import signal
from scipy.spatial.transform import Rotation as R

HEADER_DEF = {
    "Raw": "Raw,utcTimeMillis,TimeNanos,LeapSecond,TimeUncertaintyNanos,FullBiasNanos,BiasNanos,BiasUncertaintyNanos,DriftNanosPerSecond,DriftUncertaintyNanosPerSecond,HardwareClockDiscontinuityCount,Svid,TimeOffsetNanos,State,ReceivedSvTimeNanos,ReceivedSvTimeUncertaintyNanos,Cn0DbHz,PseudorangeRateMetersPerSecond,PseudorangeRateUncertaintyMetersPerSecond,AccumulatedDeltaRangeState,AccumulatedDeltaRangeMeters,AccumulatedDeltaRangeUncertaintyMeters,CarrierFrequencyHz,CarrierCycles,CarrierPhase,CarrierPhaseUncertainty,MultipathIndicator,SnrInDb,ConstellationType,AgcDb,BasebandCn0DbHz,FullInterSignalBiasNanos,FullInterSignalBiasUncertaintyNanos,SatelliteInterSignalBiasNanos,SatelliteInterSignalBiasUncertaintyNanos,CodeType,ChipsetElapsedRealtimeNanos",
    "UncalAccel": "UncalAccel,utcTimeMillis,elapsedRealtimeNanos,UncalAccelXMps2,UncalAccelYMps2,UncalAccelZMps2,BiasXMps2,BiasYMps2,BiasZMps2",
    "UncalGyro": "UncalGyro,utcTimeMillis,elapsedRealtimeNanos,UncalGyroXRadPerSec,UncalGyroYRadPerSec,UncalGyroZRadPerSec,DriftXRadPerSec,DriftYRadPerSec,DriftZRadPerSec",
    "UncalMag": "UncalMag,utcTimeMillis,elapsedRealtimeNanos,UncalMagXMicroT,UncalMagYMicroT,UncalMagZMicroT,BiasXMicroT,BiasYMicroT,BiasZMicroT",
    "OrientationDeg": "OrientationDeg,utcTimeMillis,elapsedRealtimeNanos,yawDeg,rollDeg,pitchDeg",
    "Fix": "Fix,Provider,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SpeedMps,AccuracyMeters,BearingDegrees,UnixTimeMillis,SpeedAccuracyMps,BearingAccuracyDegrees,elapsedRealtimeNanos,VerticalAccuracyMeters",
    "GNGGA": "utcTimeMillis,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SatNum,Quality,hdop",
    "GameRot": "GameRot,elapsedRealtimeNanos,utcTimeMillis,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Rot": "Rot,elapsedRealtimeNanos,utcTimeMillis,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Mark":"Loc,elapsedRealtimeNanos,utcTimeMillis,LID", 
}

ALLSENSOR_SHORT_HEADER_DEF = {
    "UncalAccel": "UncalAccel,elapsedRealtimeNanos,UncalAccelXMps2,UncalAccelYMps2,UncalAccelZMps2,BiasXMps2,BiasYMps2,BiasZMps2",
    "UncalGyro": "UncalGyro,elapsedRealtimeNanos,UncalGyroXRadPerSec,UncalGyroYRadPerSec,UncalGyroZRadPerSec,DriftXRadPerSec,DriftYRadPerSec,DriftZRadPerSec",
    "UncalMag": "UncalMag,elapsedRealtimeNanos,UncalMagXMicroT,UncalMagYMicroT,UncalMagZMicroT,BiasXMicroT,BiasYMicroT,BiasZMicroT",
    "GameRot": "GameRot,elapsedRealtimeNanos,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Rot": "Rot,elapsedRealtimeNanos,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Mark":"Loc,elapsedRealtimeNanos,LID", 
}

ALLSENSOR_HEADER_DEF = {
    "UncalAccel": "UncalAccel,elapsedRealtimeNanos,utcTimeMillis,UncalAccelXMps2,UncalAccelYMps2,UncalAccelZMps2,BiasXMps2,BiasYMps2,BiasZMps2",
    "UncalGyro": "UncalGyro,elapsedRealtimeNanos,utcTimeMillis,UncalGyroXRadPerSec,UncalGyroYRadPerSec,UncalGyroZRadPerSec,DriftXRadPerSec,DriftYRadPerSec,DriftZRadPerSec",
    "UncalMag": "UncalMag,elapsedRealtimeNanos,utcTimeMillis,UncalMagXMicroT,UncalMagYMicroT,UncalMagZMicroT,BiasXMicroT,BiasYMicroT,BiasZMicroT",
    "GameRot": "GameRot,elapsedRealtimeNanos,utcTimeMillis,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Rot": "Rot,elapsedRealtimeNanos,utcTimeMillis,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Mark":"Loc,elapsedRealtimeNanos,utcTimeMillis,LID", 
}

def load_paths(filename):
    lines = read_file(filename)
    lines = list(filter(lambda x: not x.startswith('#'), lines))
    paths = [line.split(',') for line in lines]
    return paths

def get_info(folder):
    infos = read_file(osp.join(folder, 'info.yaml'))
    info = {}
    for _ in infos:
        key = _[:_.index(':')]
        val = _[_.index(':')+1:].strip()
        info[key] = val
    return info

def _print(*args, is_print=True):
    if is_print:
        print(*args)
        
def check_dir(path, is_file=False, is_print=True):
    if is_file:
        sub_paths = path.split(os.path.sep)
        path = os.path.sep.join(sub_paths[:-1])
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            _print('mkdir: %s'%path, is_print=is_print)
        except:
            _print('mkdir fail: %s'%path, is_print=is_print)
    else:
        _print('mkdir exist: %s'%path, is_print=is_print)
    return path

def list_ind(l, ind):
    return [l[i] for i in ind]

def tojson(o, ensure_ascii=True):
    return json.dumps(o, default=lambda obj: obj.__dict__, sort_keys=True,ensure_ascii=ensure_ascii)

def toobj(strjson):
    json.loads(strjson)

def load_json(filename, encoding=None):
    json_file=open(filename, 'r', encoding=encoding)
    json_strings=json_file.readlines()
    json_string=''.join(json_strings)
    json_file.close()
    return json.loads(json_string)

def save_json(filename, obj, ensure_ascii=True, encoding=None):
    str_json=tojson(obj, ensure_ascii)
    with open(filename, 'w', encoding=encoding) as f:
        f.write(str_json)
        f.close()
    
def write_file(file_name, str_list, encoding=None, mode='w'):
    file_=open(file_name, mode, encoding=encoding)
    file_.writelines(['%s\n'%s for s in str_list])
    file_.close()

def read_file(file_name, encoding=None):
    file_=open(file_name, 'r', encoding=encoding)
    str_list = file_.read().splitlines()
    file_.close()
    return str_list

def pd2csv(df, file):
    df.to_csv(file,index=False)

def is_args_set(arg_name, option_strings_dict):
    if '-%s'%arg_name in option_strings_dict:
        option_strings = option_strings_dict['-%s'%arg_name]
    elif '--%s'%arg_name in option_strings_dict:
        option_strings = option_strings_dict['--%s'%arg_name]
    else:
        return False
    for option_string in option_strings:
        if (option_string in sys.argv) or (option_string in sys.argv):
            return True 
    return False

def get_option_strings_dict(option_strings_list):
    option_strings_dict = {}
    for option_strings in option_strings_list:
        for option_string in option_strings:
            option_strings_dict[option_string] = option_strings
    return option_strings_dict

def _set_args_config(args, parser, path=join_path('configs', 'train_configs')):
    if hasattr(args, 'config') and (args.config is not None) and len(args.config):
        option_strings_list = [action.option_strings for action in parser._actions]
        option_strings_dict = get_option_strings_dict(option_strings_list)
        for config_name in args.config:
            config = load_json(join_path(path,'%s.json'%config_name))
            for _name in config:
                if not is_args_set(_name, option_strings_dict):
                    setattr(args, _name, config[_name])

def set_args_config(parser, path=join_path('configs', 'train_configs')):
    # args > json > default
    args = parser.parse_args()
    _set_args_config(args, parser, path)
    # print('>> %s\n' % str(args))
    return args

def get_header_def(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        while (line := f.readline()):
            if line.startswith("UAcc"):
                data_len = len(line.split(','))
                if data_len == 8:
                    return ALLSENSOR_SHORT_HEADER_DEF
                elif data_len == 9:
                    return ALLSENSOR_HEADER_DEF
                else:
                    raise Exception(f'Unexpected data len {data_len}: {line}')

def convert_AllSenosr_log(txt_file, output_path, overwrite=False, imu_overwrite=False):
    uncal_accel, uncal_gyro, uncal_mag, game_orientation, orientation, mark = False, False, False, False, False, False
    _HEADER_DEF = get_header_def(txt_file)

    if not osp.exists(f"{output_path}/UncalAccel.csv") or imu_overwrite:
        uncal_accel_f = open(f"{output_path}/UncalAccel.csv", "w")
        uncal_accel_f.write(_HEADER_DEF["UncalAccel"])
        uncal_accel_f.write("\n")
        uncal_accel = True

    if not osp.exists(f"{output_path}/UncalGyro.csv") or imu_overwrite:
        uncal_gyro_f = open(f"{output_path}/UncalGyro.csv", "w")
        uncal_gyro_f.write(_HEADER_DEF["UncalGyro"])
        uncal_gyro_f.write("\n")
        uncal_gyro = True
    
    if not osp.exists(f"{output_path}/UncalMag.csv") or imu_overwrite:
        uncal_mag_f = open(f"{output_path}/UncalMag.csv", "w")
        uncal_mag_f.write(_HEADER_DEF["UncalMag"])
        uncal_mag_f.write("\n")
        uncal_mag = True

    if not osp.exists(f"{output_path}/GameRot.csv") or overwrite:
        game_orientation_deg_f = open(f"{output_path}/GameRot.csv", "w")
        game_orientation_deg_f.write(_HEADER_DEF["GameRot"])
        game_orientation_deg_f.write("\n")
        game_orientation = True

    if not osp.exists(f"{output_path}/Rot.csv") or overwrite:
        orientation_deg_f = open(f"{output_path}/Rot.csv", "w")
        orientation_deg_f.write(_HEADER_DEF["Rot"])
        orientation_deg_f.write("\n")
        orientation = True

    if not osp.exists(f"{output_path}/Mark.csv") or overwrite:
        mark_f = open(f"{output_path}/Mark.csv", "w")
        mark_f.write(_HEADER_DEF["Mark"])
        mark_f.write("\n")
        mark = True

    with open(txt_file, 'r', encoding='utf-8') as f:
        while (line := f.readline()):
            if line.startswith("#"):
                continue
            elif line.startswith("UAcc") and uncal_accel:
                uncal_accel_f.write(line)
            elif line.startswith("UGys") and uncal_gyro:
                uncal_gyro_f.write(line)
            elif line.startswith("UMag") and uncal_mag:
                uncal_mag_f.write(line)
            elif line.startswith("GameRot") and game_orientation:
                game_orientation_deg_f.write(line)
            elif line.startswith("Rot") and orientation:
                orientation_deg_f.write(line)
            elif line.startswith("Loc") and mark:
                mark_f.write(line)

    if uncal_accel:
        uncal_accel_f.close()
    if uncal_gyro:
        uncal_gyro_f.close()
    if uncal_mag:
        uncal_mag_f.close()
    if game_orientation:
        game_orientation_deg_f.close()
    if orientation:
        orientation_deg_f.close()
    if mark:
        mark_f.close()

def get_path_o(origin_path, *path):
    return osp.join(origin_path, *path)

def get_path_p(processed_path, *path):
    return osp.join(processed_path, *path)

def get_gnss_files(files, subtype=None, origin_path=None, device=None):
    if subtype is not None:
        files = sorted(list(filter(lambda x: osp.isdir(get_path_o(origin_path, device, x)), files)))
        return list_con([sorted(get_gnss_files(os.listdir(get_path_o(origin_path, device, file)))) for file in files])
    else:
        return sorted(list(filter(lambda x: re.match(r'gnss_log_[0-9_]*.txt', x), files)))

def get_rnx_files(files, subtype=None, origin_path=None, device=None):
    if subtype is not None:
        files = sorted(list(filter(lambda x: osp.isdir(get_path_o(origin_path, device, x)), files)))
        return list_con([sorted(get_rnx_files(os.listdir(get_path_o(origin_path, device, file)))) for file in files])
    else:
        return sorted(list(filter(lambda x: ('24o' in x) or ('23o' in x) or ('22o' in x), files)))
    
def get_sensor_files(files, subtype=None, origin_path=None, device=None):
    if subtype is not None:
        files = sorted(list(filter(lambda x: osp.isdir(get_path_o(origin_path, device, x)), files)))
        return list_con([sorted(get_sensor_files(os.listdir(get_path_o(origin_path, device, file)))) for file in files])
    else:
        return sorted(list(filter(lambda x: re.match(r'[0-9.]*-[0-9_]*.csv', x), files)))

def get_rtkfiles(files, subtype=None, origin_path=None, device=None):
    if subtype is not None:
        files = sorted(list(filter(lambda x: osp.isdir(get_path_o(origin_path, device, x)), files)))
        return list_con([sorted(get_rtkfiles(os.listdir(get_path_o(origin_path, device, file)))) for file in files])
    else:
        return sorted(list(filter(lambda x: re.match(r'[0-9]*gngga.txt', osp.basename(x)), files)))
    
def get_subtypes(origin_path, device, files):
    files = sorted(list(filter(lambda x: osp.isdir(get_path_o(origin_path, device, x)), files)))
    return [len(get_sensor_files(os.listdir(get_path_o(origin_path, device, file))))*[file] for file in files]

def get_trace_names(gnss_files, sensor_files):
    if len(gnss_files):
        return [_[11:-4] for _ in gnss_files]
    else:
        return [_[:-4].replace('.', '-').replace('_', '-') for _ in sensor_files]

def get_rtkfiles_from_all(all_rtkfiles, trace_names):
    all_rtkfiles_nums = [int(_[2:-9]) for _ in all_rtkfiles]
    rtkfiles = []
    for trace_name in trace_names:
        datenum = trace_name.replace('_', '')
        assert len(datenum)==12
        datenum = int(datenum)
        ind = np.argmin(np.abs(datenum-np.array(all_rtkfiles_nums)))
        if np.abs(all_rtkfiles_nums[ind] - datenum) > 60:
            print(f"[WARNING] closest rtkfile to {trace_name} is {all_rtkfiles[ind]}")
            # pdb.set_trace()
        rtkfiles.append(all_rtkfiles[ind])
    return rtkfiles

def scan_devices(origin_path):
    devices = os.listdir(origin_path)
    if 'rtklite' in devices:
        devices.remove('rtklite')
    return devices

def scan_rtkfiles(origin_path):
    if osp.exists(get_path_o(origin_path, 'rtklite')):
        files = os.listdir(get_path_o(origin_path, 'rtklite'))
        return get_rtkfiles(files)
    else:
        return []

def match_rtkfiles(trace_names, all_rtkfiles, files, subtype, origin_path, device):
    if len(all_rtkfiles):
        rtkfiles = get_rtkfiles_from_all(all_rtkfiles, trace_names)
    else:
        rtkfiles = get_rtkfiles(files, subtype, origin_path, device)
        rtkfiles = get_rtkfiles_from_all(rtkfiles, trace_names)
    return rtkfiles

def get_attribute(trace_num, attribute, attributes, subtype, subtypes):
    if attribute == subtype:
        attributes = list_con(subtypes)
    elif len(attributes)==1:
        attributes = attributes * trace_num
    elif subtype is not None and subtype != attribute:
        attributes = attributes * len(subtypes)
    return attributes

def convert_GNSS_log(trip_dir, gnss_file_name):
    # gnss_log_2022_12_30_12_31_01.txt -> 12_30_12_31
    # date_time = os.path.splitext(gnss_file_name)[0][14:-3]

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
    with open(gnss_file_name, 'r', encoding='utf-8') as f:
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

def convert_RTKLite_log(rtkfile, csv_name):
    gngga_f = open(csv_name, "w")
    gngga_f.write(HEADER_DEF["GNGGA"])
    gngga_f.write("\n")
    
    with open(rtkfile, 'r', encoding='utf-8') as f:
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

def rerange_deg(raw_deg):
    # raw_deg = gt_df[bearing_key].values
    raw_deg = raw_deg%360 # 所有角度限制到[0,360]
    raw_deg[raw_deg>180] = raw_deg[raw_deg>180]-360
    return raw_deg

def moving_average_filter(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def magnetic_calibration(mag, N=100):
    cali_mag = mag[:, :3]
    uncali_mag = mag[:, 3:]
    freq = max(int(mag.shape[0]//N), 1)
    samp_mag = uncali_mag[::freq, :]

    mag_square = np.square(samp_mag)
    mag_square_diff = mag_square[:, np.newaxis] - mag_square[np.newaxis, :]
    double_mag_diff = 2*(samp_mag[:, np.newaxis] - samp_mag[np.newaxis, :])
    A = double_mag_diff.reshape((-1,3))
    b = np.sum(-mag_square_diff.reshape((-1,3)), axis=-1, keepdims=True)
    dx, dy, dz = np.linalg.lstsq(A, b, rcond=None)[0] # 最小二乘求解

    mag_cali = uncali_mag + np.concatenate((dx, dy, dz))
    mag_cali_norm_mean = np.mean(np.linalg.norm(mag_cali, axis=1))
    cali_mag_norm_mean = np.mean(np.linalg.norm(cali_mag, axis=1))
    if mag_cali_norm_mean<20 and cali_mag_norm_mean<70:
        print(f'norm of mag_cali ({mag_cali_norm_mean}) is too small, use cali_mag ({cali_mag_norm_mean}) instead')
        # pdb.set_trace()
        mag_cali = cali_mag
    return mag_cali

def do_lowpass(imu,fs=100,fc=40):
    '''
        低通滤波
        fs = 100 # 采样率 (赫兹)
        fc = 20 # 截止频率 （赫兹）
    '''
    b, a = signal.butter(4, 2.0*fc/fs, 'lowpass')
    filtered_x = signal.filtfilt(b, a, imu[:,0])
    filtered_y = signal.filtfilt(b, a, imu[:,1])
    filtered_z = signal.filtfilt(b, a, imu[:,2])
    filtered_imu = np.column_stack((filtered_x, filtered_y, filtered_z))
    return filtered_imu

def rotation_matrix_row(a, b):
    """
    计算将向量 a 旋转至向量 b 所需的旋转矩阵
    """
    a = a / np.linalg.norm(a, axis=-1)[:, np.newaxis]  # 将向量 a 归一化
    b = b / np.linalg.norm(b, axis=-1)[:, np.newaxis]  # 将向量 b 归一化
    v = np.cross(a, b, axis=-1)         # 计算向量 a 和 b 的外积
    s = np.linalg.norm(v, axis=-1)      # 计算向量 a 和 b 的外积的模
    c = np.sum(a*b, axis=-1)           # 计算向量 a 和 b 的内积
    vx = np.zeros((a.shape[0], 3, 3))
    vx[:, 0, 1] = -v[:, 2]
    vx[:, 0, 2] = v[:, 1]
    vx[:, 1, 0] = v[:, 2]
    vx[:, 1, 2] = -v[:, 0]
    vx[:, 2, 0] = -v[:, 1]
    vx[:, 2, 1] = v[:, 0]
    vxdot = -np.matmul(vx[:, np.newaxis], vx[:, :, :, np.newaxis])[:, :, :, 0]
    return np.identity(3) + vx + vxdot * ((1 - c) / (s ** 2)).reshape(-1,1,1)

def get_mag_gra_rot(mag_cali, gra, _gra=9.81):
    gra_norm = np.repeat([[0,0, _gra]],len(gra),axis=0)
    Rots = R.from_matrix(rotation_matrix_row(gra, gra_norm))
    euler = Rots.as_euler('xyz', degrees=True)
    euler[:,2] = 0
    Rots = R.from_euler('xyz', euler, degrees=True)
    mag_cali_enu = Rots.apply(mag_cali)
    euler[:,2] = rerange_deg(90-np.rad2deg(np.arctan2(mag_cali_enu[:,1], mag_cali_enu[:,0])))
    Rots = R.from_euler('xyz', euler, degrees=True)
    return Rots

def get_mag_rot(mag_cali, euler):
    euler = euler.copy()
    euler[:,2] = 0
    Rots = R.from_euler('yxz', euler, degrees=True)
    mag_cali_enu = Rots.apply(mag_cali)
    euler[:,2] = rerange_deg(90-np.rad2deg(np.arctan2(mag_cali_enu[:,1], mag_cali_enu[:,0])))
    Rots = R.from_euler('yxz', euler, degrees=True)
    return Rots

def read_data_list(dataset, data_list):
    infos = read_file(data_list)
    infos = list(filter(lambda x: not x.startswith('#'), infos))
    trips = list(map(lambda x: x.split(',')[0], infos))
    persons = list(map(lambda x: x.split(',')[1], infos))
    return infos, trips, persons