from operator import concat
from turtle import goto
import pandas as pd
import numpy as np
import time, os, shutil, sys, argparse, math
import os.path as osp
from pyparsing import with_attribute
from utm import from_latlon as ll2en
import pymap3d as pm
from itertools import dropwhile
import re, datetime
from tools.tools import set_args_config, pd2csv, check_dir
from mtools import str2bool

pd.set_option('display.float_format',lambda x : '%.5f' % x)
np.set_printoptions(precision=5) 

parser = argparse.ArgumentParser(description='')
# file and path
parser.add_argument('-exe_file', '--exe_file', type=str)    # tdcp的可执行文件
parser.add_argument('-obs_file', '--obs_file', type=str)    # 观测文件
parser.add_argument('-beph_file','--beph_file',type=str)    # 广播星历
parser.add_argument('-peph_file','--peph_file',type=str)    # 精密星历
parser.add_argument('-clk_file', '--clk_file', type=str)    # 钟差文件
parser.add_argument('-out_path', '--out_path', type=str)    # 输出路径
parser.add_argument('-gt_file',  '--gt_file',  type=str)    # ground truth文件    
# param 
parser.add_argument('-sys','--sys',type=str,default='GREC') # 使用的卫星系统 GREC 
parser.add_argument('-eph','--eph',type=int,default=0)      # 0:广播星历 1:精密星历
parser.add_argument('-snr','--snr',type=int,default=20)     # snr阈值
parser.add_argument('-ele','--ele',type=int,default=15)     # 高度角阈值
parser.add_argument('-pntpos_validate','--pntpos_validate',type=int,default=1)  # 0:单点定位不进行valsol验证过滤 1:单点定位进行valsol验证过滤
parser.add_argument('-get_satinfo','--get_satinfo',type=int,default=0)  # 0:不保存卫星信息 1:保存卫星信息到文件
args = set_args_config(parser)

if args.out_path is None:
    args.out_path = osp.dirname(args.obs_file)
else:
    check_dir(args.out_path)

sol_file = os.path.join(args.out_path, "pntpos.csv") 
sat_file = os.path.join(args.out_path, "satinfo.csv") 
pos_file = os.path.join(args.out_path, "pntpos.pos")  

terminal = f"{args.exe_file} "+\
           f"-sys {args.sys} -eph {args.eph} -snr {args.snr} -ele {args.ele} "+\
           f"-pntpos_validate {args.pntpos_validate} -get_satinfo {args.get_satinfo} "+\
           f"-Fobs {args.obs_file} -Feph {args.beph_file} -Fout {pos_file} -Pout {args.out_path}\\ %s %s "\
           %((f"-Fsp3 {args.peph_file} -Fclk {args.clk_file} "if args.eph else ""), (f"-Fgt {args.gt_file} "if args.gt_file else ""))

print("RUNNING PNTPOS ....................................................... ")
file_list = ['pntpos.csv','satinfo.csv','pntpos.pos']
for file in file_list:
    if osp.isfile(osp.join(args.out_path, file)):
        os.system(f"rm -rf {osp.join(args.out_path, file)}")

print(terminal)
os.system(terminal)






