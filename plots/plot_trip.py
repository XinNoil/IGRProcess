# -*- coding: UTF-8 -*-
"""
Usage:
  plot_trip.py <trip_dir> [options]

Options:
  -s <suffix>, --suffix <suffix>  后缀
"""
import os

from docopt import docopt
import ipdb as pdb

import numpy as np
np.set_printoptions(precision=4, suppress=True, formatter={'float_kind':'{:f}'.format})

import pandas as pd
pd.set_option('display.float_format','{:.4f}'.format)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman']

import plotly.express as px

def plot_traj(
    df,
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    center=None,
    # color_col="phone",
    # label_col="tripId",
    zoom=9,
    opacity=1,
):
    if center is None:
        center = {
            "lat": df[lat_col].mean(),
            "lon": df[lon_col].mean(),
        }
    fig = px.scatter_mapbox(
        df,
        # Here, plotly gets, (x,y) coordinates
        lat=lat_col,
        lon=lon_col,
        # Here, plotly detects color of series
        # color=color_col,
        # labels=label_col,
        zoom=zoom,
        center=center,
        height=600,
        width=800,
        opacity=0.5,
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()

def plot_trip(gngga_path):
    gngga_df = pd.read_csv(gngga_path)
    plot_traj(gngga_df, zoom=16)

def main():
    arguments = docopt(__doc__)
    trip_dir = arguments.trip_dir
    suffix = arguments.suffix

    if suffix is None:
        pdb.set_trace()
        suffixs = list(filter(lambda x: x.endswith('csv'), os.listdir(trip_dir)))
        for suffix in suffixs:
            print(f'python plot_trip.py {arguments.trip_dir} -s {suffix}')
    else:
        plot_trip(os.path.join(trip_dir, suffix))

if __name__ == "__main__":
    main()

'''
python plot_trip.py IGR230307/origin/rtklite -s 20230307161946gngga.csv
python plot_trip.py IGR230307/origin/rtklite -s 20230307160401gngga.csv
python plot_trip.py IGR230307/origin/rtklite -s 20230307144938gngga.csv
python plot_trip.py IGR230307/origin/rtklite -s 20230307155133gngga.csv
python plot_trip.py IGR230307/origin/rtklite -s 20230307151408gngga.csv
python plot_trip.py IGR230307/origin/rtklite -s 20230307153521gngga.csv
python plot_trip.py IGR230307/origin/rtklite -s 20230307164016gngga.csv
python plot_trip.py IGR230307/origin/rtklite -s 20230307152423gngga.csv

python plot_trip.py IGR230312/origin/rtklite -s 20230312171823gngga.csv
python plot_trip.py IGR230312/origin/rtklite -s 20230312165422gngga.csv
python plot_trip.py IGR230312/origin/rtklite -s 20230312172747gngga.csv
python plot_trip.py IGR230312/origin/rtklite -s 20230312153817gngga.csv
python plot_trip.py IGR230312/origin/rtklite -s 20230312170545gngga.csv
python plot_trip.py IGR230312/origin/rtklite -s 20230312160511gngga.csv
python plot_trip.py IGR230312/origin/rtklite -s 20230312164447gngga.csv
python plot_trip.py IGR230312/origin/rtklite -s 20230312173738gngga.csv

python plot_trip.py IGR230415/origin/rtklite -s 20230415145239gngga.csv
python plot_trip.py IGR230415/origin/rtklite -s 20230415150333gngga.csv
python plot_trip.py IGR230415/origin/rtklite -s 20230415152021gngga.csv
python plot_trip.py IGR230415/origin/rtklite -s 20230415152949gngga.csv
python plot_trip.py IGR230415/origin/rtklite -s 20230416181855gngga.csv
python plot_trip.py IGR230415/origin/rtklite -s 20230416183648gngga.csv


python plot_trip.py IGRData/IGR230419/origin/rtklite -s 20230419105342gngga.csv
'''