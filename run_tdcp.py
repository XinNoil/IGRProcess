
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
parser.add_argument('-gt_file',  '--gt_file',  type=str)    # ground truth文件    # 是否运行tdcp
# param 
parser.add_argument('-alg',  '--alg',  type=int,default=1)         # 0:tdpr 1:tdcp 
parser.add_argument('-sys',  '--sys',  type=str,default='GREC')    # 使用的卫星系统 GREC 
parser.add_argument('-etype','--etype',type=int,default=2)         # tdcp的四种方法之一
parser.add_argument('-eph',  '--eph',  type=int,default=0)         # 0:广播星历 1:精密星历
parser.add_argument('-dts',  '--dts',  type=int,default=0)         # 卫星钟差的处理方案 0:不处理 1:后减前 -1:前减后
parser.add_argument('-snr',  '--snr',  type=int,default=20)        # snr阈值
parser.add_argument('-ele',  '--ele',  type=int,default=15)        # 高度角阈值
parser.add_argument('-lli',  '--lli',  type=int,default=0)         # 0:无lli筛选 1:lli筛选
parser.add_argument('-doppler','--doppler',type=int,default=0)     # 多普勒筛选 0:不开启 其它:?
parser.add_argument('-max_speed','--max_speed',type=int,default=0) # 0:无速度限制 其它：限制的速度
# param-seldom use
parser.add_argument('-cs','--cs',type=int,default=0)                            # 0:无周跳检测 1:四次差
parser.add_argument('-cs_repair','--cs_repair',type=int,default=0)              # 0:不进行周跳修复 1:用ground truth修复 2:用pre_N_pesudorange修复 3:用mean_pre_N_pesudorange修复 4:用mean_pre_N_pntpos修复 
parser.add_argument('-get_gt_N','--get_gt_N',type=int,default=0)                # 0:不获取周跳的值  1:获取
parser.add_argument('-get_LPDSNRLLI','--get_LPDSNRLLI',type=int,default=0)      # 0:不获取LS.csv等 1:获取
parser.add_argument('-get_mid','--get_mid',type=int,default=0)                  # 0:不获取中间计算结果 1:获取
parser.add_argument('-pntpos_validate','--pntpos_validate',type=int,default=1)  # 0:单点定位不进行valsol验证过滤 1:单点定位进行valsol验证过滤
args = set_args_config(parser)

if args.out_path is None:
    args.out_path = osp.dirname(args.obs_file)
else:
    check_dir(args.out_path)

sol_file = osp.join(args.out_path, "tdcp.csv") 
pos_file = osp.join(args.out_path, "tdcp.pos")  

terminal = f"{args.exe_file} "+\
           f"-alg {args.alg} -sys {args.sys} -etype {args.etype} -eph {args.eph} -dts {args.dts} -snr {args.snr} -ele {args.ele} -doppler {args.doppler} -lli {args.lli} "+\
           f"-cs {args.cs} -get_LPDSNRLLI {args.get_LPDSNRLLI} -get_gt_N {args.get_gt_N} -cs_repair {args.cs_repair} -get_mid {args.get_mid} -max_speed {args.max_speed} -pntpos_validate {args.pntpos_validate} "+\
           f"-Fobs {args.obs_file} -Feph {args.beph_file} -Fout {pos_file} -Pout {args.out_path}\\ %s %s "\
           %((f"-Fsp3 {args.peph_file} -Fclk {args.clk_file} " if args.eph else ""), (f"-Fgt {args.gt_file}"if args.gt_file else ""))

print("RUNNING TDCP ....................................................... ")
file_list = ['Ds.csv','LLIs.csv','Ls.csv','Ps.csv','SNRs.csv','tdcp.csv','tdcp_mid_data.csv','tdcp.pos']
for file in file_list:
    if osp.isfile(osp.join(args.out_path, file)):
        os.system(f"rm -rf {osp.join(args.out_path, file)}")
print(terminal)
os.system(terminal)

sol_df = pd.read_csv(sol_file)
if len(sol_df):
    sol_df['epoch0_timeUTC'] = sol_df.apply(lambda x: int(x['epoch0_time'] - 18000), axis=1)
    sol_df['epoch1_timeUTC'] = sol_df.apply(lambda x: int(x['epoch1_time'] - 18000), axis=1)
    save_col = ['epoch0','epoch1','epoch0_time','epoch1_time','epoch0_timeUTC','epoch1_timeUTC','x','y','z','dt_gps','dt_glo','dt_gal','dt_cmp','nv','gdop_ecef','pdop_ecef','hdop_ecef','vdop_ecef','gdop_enu','pdop_enu','hdop_enu','vdop_enu']
    pd2csv(sol_df[save_col], sol_file)
else:
    print("[WARNING] No successful TDCP solution")




