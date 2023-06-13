#
# 转化rinex脚本 (H:\workspace-oml\RTK\data\OurGNSSLoggerDataset\RTK_GNSSLogger)
#

# if 'rtklib-py/src' not in sys.path:
#     sys.path.append('rtklib-py/src')
# if 'android_rinex/src' not in sys.path:
#     sys.path.append('android_rinex/src')

import os, argparse
import re
import shutil
import os.path as osp
from multiprocessing import Pool
import tools.gnsslogger_to_rnx as rnx
from time import time
from tools.tools import read_file, get_info
from mtools import str2bool
import ipdb as pdb

# set run parameters
maxepoch = None # max number of epochs, used for debug, None = no limit

# Set solution choices
DEBUG = True            # True 使用串行执行方式

# specify location of input folder and files

# input structure for rinex conversion
class Args:
    def __init__(self):
        # Input parameters for conversion to rinex
        self.slip_mask = 0 # overwritten below
        self.fix_bias = True
        self.timeadj = 1e-7
        self.pseudorange_bias = 0
        self.filter_mode = 'sync'
        # Optional hader values for rinex files
        self.marker_name = ''
        self.observer = ''
        self.agency = ''
        self.receiver_number = ''
        self.receiver_type = ''
        self.receiver_version = ''
        self.antenna_number = ''
        self.antenna_type = ''

# function to convert single rinex file
def convert_rnx(rawFile, rovFile, slipMask):
    argsIn = Args()
    argsIn.input_log = rawFile
    argsIn.output = os.path.basename(rovFile) # path: /a/b/c/ return c
    argsIn.slip_mask = slipMask
    rnx.convert2rnx(argsIn)

def main(data_path):
    datadir = os.path.join('IGRData', data_path, 'processed')
    devices = read_file(os.path.join('IGRData', data_path, 'devices.txt'))
    devices = list(filter(lambda x: not x.startswith('#'), devices))
    rinexIn = []
    for phone in devices:
        times = os.listdir(os.path.realpath(os.path.join(datadir,phone)))
        times.sort()
        for time in times:
            # skip if no folder for this phone
            folder = osp.realpath(osp.join(datadir, phone, time))
            # 获取raw data的路径
            rawfiles = os.listdir(osp.join(folder, 'supplementary'))
            try:
                rawFile = list(filter(lambda _: re.search(r'gnss_log_[0-9_]*.txt', _) is not None, rawfiles))[0]
            except:
                pdb.set_trace()
            rawFile = osp.join(folder, 'supplementary', rawFile)
            rovFile = osp.join(folder, 'supplementary', 'gnss_log.obs')

            # check if need rinex conversion
            if OVERWRITE_RINEX or not osp.isfile(rovFile):
                # generate list of input parameters for each rinex conversion
                slipMask = 0 
                rinexIn.append((rawFile, rovFile, slipMask))
                print(phone, time, rawFile, '->', rovFile)    
    
    if len(rinexIn) > 0:
        print('\nConvert rinex files...')
        # 并发运行
        # generate rinx obs files in parallel, does not give error messages
        if not DEBUG:
            with Pool() as pool: # defaults to using cpu_count for number of procceses
                res = pool.starmap(convert_rnx, rinexIn)
        # 串行运行
        # run sequentially, use for debug
        if DEBUG:
            for input in rinexIn:
                print(input[1].replace(osp.realpath(datadir)+os.sep, ''))
                try:
                    convert_rnx(*input)
                except:
                    info = get_info(osp.dirname(input[0]))
                    rnx_file = info['rnx_file']
                    shutil.copyfile(osp.join(osp.dirname(input[0]), rnx_file), input[1])
                    print(f'convert failed, use {rnx_file} directly')


"""
Usage:
  get_rinex.py <data_path> [options]

Options:
  -o <overwrite>, --overwrite <overwrite>  室内 [default: 1]
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str)
    parser.add_argument('-o', '--overwrite', type=str2bool, default=False)
    args = parser.parse_args()
    data_path = args.data_path
    OVERWRITE_RINEX = args.overwrite

    t0 = time()
    main(data_path)
    print('Runtime=%.1f' % (time() - t0))

