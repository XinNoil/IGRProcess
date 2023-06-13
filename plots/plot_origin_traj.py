import os, argparse, re
import os.path as osp
from mtools import list_con, read_file
import pandas as pd
import ipdb as pdb
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('tools')
from tools.tools import get_rtkfiles, convert_RTKLite_log

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str)
args = parser.parse_args()

origin_path = osp.join('IGRData', args.data_dir, 'origin')
os.chdir(osp.join('IGRData', args.data_dir))

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

if not os.path.exists(os.path.join('origin', 'rtklite')):
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
