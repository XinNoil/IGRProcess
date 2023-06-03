import os
import json
import sys
import os.path as osp
from os.path import join as join_path

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

HEADER_DEF = {
    "UncalAccel": "UncalAccel,elapsedRealtimeNanos,utcTimeMillis,UncalAccelXMps2,UncalAccelYMps2,UncalAccelZMps2,BiasXMps2,BiasYMps2,BiasZMps2",
    "UncalGyro": "UncalGyro,elapsedRealtimeNanos,utcTimeMillis,UncalGyroXRadPerSec,UncalGyroYRadPerSec,UncalGyroZRadPerSec,DriftXRadPerSec,DriftYRadPerSec,DriftZRadPerSec",
    "GameRot": "GameRot,elapsedRealtimeNanos,utcTimeMillis,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Rot": "Rot,elapsedRealtimeNanos,utcTimeMillis,quaternionX,quaternionY,quaternionZ,quaternionW",
    "Mark":"Loc,elapsedRealtimeNanos,utcTimeMillis,LID", 
}

def convert_AllSenosr_log(txt_file, output_path, uncal_accel=True, uncal_gyro=True, game_orientation=True, orientation=True, mark=True):
    if uncal_accel:
        uncal_accel_f = open(f"{output_path}/UncalAccel.csv", "w")
        uncal_accel_f.write(HEADER_DEF["UncalAccel"])
        uncal_accel_f.write("\n")

    if uncal_gyro:
        uncal_gyro_f = open(f"{output_path}/UncalGyro.csv", "w")
        uncal_gyro_f.write(HEADER_DEF["UncalGyro"])
        uncal_gyro_f.write("\n")

    if game_orientation:
        game_orientation_deg_f = open(f"{output_path}/GameRot.csv", "w")
        game_orientation_deg_f.write(HEADER_DEF["GameRot"])
        game_orientation_deg_f.write("\n")

    if orientation:
        orientation_deg_f = open(f"{output_path}/Rot.csv", "w")
        orientation_deg_f.write(HEADER_DEF["Rot"])
        orientation_deg_f.write("\n")

    if mark:
        mark_f = open(f"{output_path}/Mark.csv", "w")
        mark_f.write(HEADER_DEF["Mark"])
        mark_f.write("\n")

    with open(txt_file, 'r', encoding='utf-8') as f:
        while (line := f.readline()):
            if line.startswith("#"):
                continue
            elif line.startswith("UAcc") and uncal_accel:
                uncal_accel_f.write(line)
            elif line.startswith("UGys") and uncal_gyro:
                uncal_gyro_f.write(line)
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
    if game_orientation:
        game_orientation_deg_f.close()
    if orientation:
        orientation_deg_f.close()
    if mark:
        mark_f.close()