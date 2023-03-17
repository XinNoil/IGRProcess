from operator import concat
from turtle import goto
import pandas as pd
import numpy as np
import time, os, shutil, sys, argparse, math
from os.path import join, split, isdir, isfile, realpath
from pyparsing import with_attribute
from utm import from_latlon as ll2en
import pymap3d as pm
from itertools import dropwhile
import re, datetime
from tools.tools import set_args_config, pd2csv, check_dir

pd.set_option('display.float_format',lambda x : '%.5f' % x)
np.set_printoptions(precision=5) 
GPS_TO_UTC = 315964782

parser = argparse.ArgumentParser(description='')
parser.add_argument('-snr','--snr',type=int,default=0)   # snr筛选
parser.add_argument('-ele','--ele',type=int,default=0)   # 高度角筛选
parser.add_argument('-GPSa','--GPSa',type=float,default=0.0)
parser.add_argument('-GPSb','--GPSb',type=float,default=0.0)   
parser.add_argument('-GLOa','--GLOa',type=float,default=0.0)
parser.add_argument('-GLOb','--GLOb',type=float,default=0.0) 
parser.add_argument('-GALa','--GALa',type=float,default=0.0)
parser.add_argument('-GALb','--GALb',type=float,default=0.0)   
parser.add_argument('-BDSa','--BDSa',type=float,default=0.0)
parser.add_argument('-BDSb','--BDSb',type=float,default=0.0)  
parser.add_argument('-exe_file','--exe_file',type=str)   # doppler的可执行文件
parser.add_argument('-obs_file','--obs_file',type=str)   # 观测文件
parser.add_argument('-beph_file','--beph_file',type=str) # 广播星历
parser.add_argument('-out_path','--out_path',type=str,default=None)   # 输出路径
args = set_args_config(parser)

if args.out_path is None:
    args.out_path = os.path.dirname(args.obs_file)

pos_file = os.path.join(args.out_path, "doppler.pos") 
csv_file = os.path.join(args.out_path, "doppler.csv") 

terminal = f"{args.exe_file} -p 0 -e -ele {args.ele} -snr {args.snr} "\
           f"-GPSa {args.GPSa} -GPSb {args.GPSb} -GLOa {args.GLOa} -GLOb {args.GLOb} "\
           f"-GALa {args.GALa} -GALb {args.GALb} -BDSa {args.BDSa} -BDSb {args.BDSb} "\
           f"-o {pos_file} {args.obs_file} {args.beph_file}" # -p 0: mode=single;   -e: output ecef

print("")
print("running doppler .................................................... ")
if len(args.out_path): check_dir(args.out_path)
print(terminal)
os.system(terminal)

print("extract .csv from .pos .................................................... ")
pos_fp = open(pos_file, 'r')
str_list = pos_fp.read()

with open(pos_file) as f:
    lines = list(dropwhile(lambda line: line.startswith('%'), f)) # 跳过开头为 % 的行
    lines = [re.split(r"[ ]+", line.replace('\n','')) for line in lines]
    df = pd.DataFrame(lines)
    if len(df):
        df.columns = ['week', 'tow', 'x-ecef', 'y-ecef', 'z-ecef', 'Q', 'ns', 'sdx', 'sdy', 'sdz', 'sdxy', 'sdyz', 'sdzx', 'age', 'ratio', 'vx', 'vy', 'vz', 'sdvx', 'sdvy', 'sdvz', 'sdvxy', 'sdvyz', 'sdvzx']
        df['week'] = pd.to_numeric(df['week'], downcast='integer')
        df['tow'] = [float(i) for i in df['tow']]
        df['vx'] = [float(i) for i in df['vx']]
        df['vy'] = [float(i) for i in df['vy']]
        df['vz'] = [float(i) for i in df['vz']]
        df['timestampUTC'] = df.apply(lambda x: (int(1000*(x['week']*7*24*3600 + x['tow']))) + GPS_TO_UTC*1000, axis=1) # UTC时间
        df['timestamp'] = df.apply(lambda x: int((x['week']*7*24*3600 + x['tow'] + pd.Timestamp(datetime.datetime(1980, 1, 6, 0, 0, 0), tz='utc').timestamp())*1000), axis=1) # GPS时间（用unix在线时间转化器转化的结果和rinex文件的时间一致）
        pd2csv(df[['timestamp','timestampUTC','vx','vy','vz']], csv_file) 
    else:
        print(f"[WARNING] {pos_file} is empty\n")


