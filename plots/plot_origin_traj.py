import os, argparse, re
import os.path as osp
from mtools import list_con, read_file
import pandas as pd
import ipdb as pdb
import datetime
from datetime import timezone
import pynmea2
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str)
parser.add_argument('-st', '--subtype', type=str, default=None)
args = parser.parse_args()

os.chdir(osp.join('IGRData', args.data_dir))

def get_rtkfiles(files):
    return sorted(list(filter(lambda x: re.match(r'.*[0-9]*gngga.txt', x), files)))

def convert_RTKLite_log(rtkfile, csv_name):
    gngga_f = open(csv_name, "w")
    gngga_f.write("utcTimeMillis,LatitudeDegrees,LongitudeDegrees,AltitudeMeters,SatNum,Quality,hdop")
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

def plot_traj(csv_name, latlon):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(latlon[:, 0], latlon[:, 1], label="Ground Truth")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # ax.axis('equal')
    ax.set_title(csv_name.replace('.csv', ''))
    fig.tight_layout()
    fig_name = csv_name.replace('.csv', '.png')
    fig.savefig(fig_name, dpi=300)
    print(f'save png to {fig_name}')
    plt.close(fig)

if args.subtype is not None:
    devices = read_file('devices.txt')
    for device in devices:
        files = sorted(os.listdir(os.path.join('origin', device)))
        files = sorted(list(filter(lambda x: osp.isdir(os.path.join('origin', device, x)), files)))
        files = list_con([[osp.join(file, _) for _ in sorted(os.listdir(os.path.join('origin', device, file)))] for file in files])
        files = get_rtkfiles(files)
        print(files)
        for file in files:
            rtk_name = osp.join('origin', device, file)
            csv_name = rtk_name.replace('.txt', '.csv')
            if not osp.exists(csv_name):
                convert_RTKLite_log(rtk_name, csv_name)
            gngga_df = pd.read_csv(csv_name)
            latlon = gngga_df[['LatitudeDegrees', 'LongitudeDegrees']].values
            plot_traj(csv_name, latlon)
else:
    files = sorted(os.listdir(os.path.join('origin', 'rtklite')))
    files = get_rtkfiles(files)
    for file in files:
        rtk_name = osp.join('origin', 'rtklite', file)
        csv_name = rtk_name.replace('.txt', '.csv')
        if not osp.exists(csv_name):
            convert_RTKLite_log(rtk_name, csv_name)
        gngga_df = pd.read_csv(csv_name)
        latlon = gngga_df[['LatitudeDegrees', 'LongitudeDegrees']].values
        plot_traj(csv_name, latlon)
