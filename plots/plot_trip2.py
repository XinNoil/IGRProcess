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

def main():
    trip_dir = sys.argv[1]
    if len(sys.argv)>2:
        suffix = sys.argv[2]
        rtk_name = os.path.join(trip_dir, suffix)
        csv_name = rtk_name.replace('.txt', '.csv')
        if not osp.exists(csv_name):
            convert_RTKLite_log(rtk_name, csv_name)
        gngga_df = pd.read_csv(csv_name)
        latlon = gngga_df[['LatitudeDegrees', 'LongitudeDegrees']].values
        plot_traj(csv_name, latlon)
    else:
        files = sorted(os.listdir(trip_dir))
        files = get_rtkfiles(files)
        for file in files:
            rtk_name = osp.join(trip_dir, file)
            csv_name = rtk_name.replace('.txt', '.csv')
            if not osp.exists(csv_name):
                convert_RTKLite_log(rtk_name, csv_name)
            gngga_df = pd.read_csv(csv_name)
            latlon = gngga_df[['LatitudeDegrees', 'LongitudeDegrees']].values
            plot_traj(csv_name, latlon)

if __name__ == "__main__":
    main()

'''
python plot_trip.py IGR230307/origin/rtklite -s 20230307161946gngga.csv
python plots/plot_trip2.py IGRData/IGR241012/origin/rtklite
'''