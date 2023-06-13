# -*- coding: UTF-8 -*-

import os
import sys

import ipdb as pdb

import numpy as np
np.set_printoptions(precision=4, suppress=True, formatter={'float_kind':'{:f}'.format})

import pandas as pd
pd.set_option('display.float_format','{:.4f}'.format)
pd.set_option('display.width', 500)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['figure.titlesize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 16

import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

DATA_DIR = fr"/home/wjk/Workspace/Datasets/IGR/IGRProcessed"

def plot_traj(
    df,
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    opacity=1,
    style="open-street-map",
    save_path="./temp.png",
    color="red",
):
    def zoom_center(lons: tuple=None, lats: tuple=None, lonlats: tuple=None,
        format: str='lonlat', projection: str='mercator',
        width_to_height: float=2.0):
        """Finds optimal zoom and centering for a plotly mapbox.
        Must be passed (lons & lats) or lonlats.
        Temporary solution awaiting official implementation, see:
        https://github.com/plotly/plotly.js/issues/3434
        
        Parameters
        --------
        lons: tuple, optional, longitude component of each location
        lats: tuple, optional, latitude component of each location
        lonlats: tuple, optional, gps locations
        format: str, specifying the order of longitud and latitude dimensions,
            expected values: 'lonlat' or 'latlon', only used if passed lonlats
        projection: str, only accepting 'mercator' at the moment,
            raises `NotImplementedError` if other is passed
        width_to_height: float, expected ratio of final graph's with to height,
            used to select the constrained axis.
        
        Returns
        --------
        zoom: float, from 1 to 20
        center: dict, gps position with 'lon' and 'lat' keys

        >>> print(zoom_center((-109.031387, -103.385460),
        ...     (25.587101, 31.784620)))
        (5.75, {'lon': -106.208423, 'lat': 28.685861})
        """
        if lons is None and lats is None:
            if isinstance(lonlats, tuple):
                lons, lats = zip(*lonlats)
            else:
                raise ValueError(
                    'Must pass lons & lats or lonlats'
                )
        
        maxlon, minlon = max(lons), min(lons)
        maxlat, minlat = max(lats), min(lats)
        center = {
            'lon': round((maxlon + minlon) / 2, 6),
            'lat': round((maxlat + minlat) / 2, 6)
        }
        
        # longitudinal range by zoom level (20 to 1)
        # in degrees, if centered at equator
        lon_zoom_range = np.array([
            0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
            0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
            47.5136, 98.304, 190.0544, 360.0
        ])
        
        if projection == 'mercator':
            margin = 1.2
            height = (maxlat - minlat) * margin * width_to_height
            width = (maxlon - minlon) * margin
            lon_zoom = np.interp(width , lon_zoom_range, range(20, 0, -1))
            lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
            zoom = round(min(lon_zoom, lat_zoom), 2)
            zoom -= 0.5
        else:
            raise NotImplementedError(
                f'{projection} projection is not implemented'
            )
        
        return zoom, center
    
    mapbox_access_token = "pk.eyJ1IjoiZHllcmNveDMwOSIsImEiOiJjbGhyYXR3dHEwM3ozM3RtczI5bXl4NDJvIn0.zkfyk_i7aonj5j7ntmMD-g"
    zoom, center = zoom_center(lons=df[lon_col], lats=df[lat_col])

    if type(color)==str:
        fig = go.Figure(go.Scattermapbox(
                lat=df[lat_col],
                lon=df[lon_col],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=6,
                    color="red",
                ),
            ))
    else:
        fig = go.Figure(go.Scattermapbox(
            lat=df[lat_col],
            lon=df[lon_col],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=6,
                color=color,
            ),
        ))

    fig.update_layout(
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=center,
            zoom=zoom,
            style=style
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title_text="LiteRTK Trajectory",
        height=768,
        width=1024,
    )

    fig.write_image(save_path)

def plot_trip(phone, trip):
    print(phone, trip)

    h5path = f"{DATA_DIR}/{phone}/{trip}/data.h5"
    if not os.path.exists(h5path):
        return
    
    store = pd.HDFStore(h5path, mode='r')
    if "/gngga" not in store.keys():
        store.close()
        return

    fix_df = store.get('fix')
    gngga_df = store.get('gngga')
    store.close()

    real_fix_df = fix_df.query("Source == 'Sensor'")
    real_gngga_df = gngga_df.query("Source == 'Sensor'")

    
    #  "stamen- terrain" "open-street-map"
    plot_traj(real_gngga_df, style="satellite-streets", save_path=f"{DATA_DIR}/{phone}/{trip}/[Map]-[{phone}]-[{trip}].png")


def main():
    phone_list = [
        "Mi8",
        "Mi8",
        "Mi8",
        "Mi8",
        "Mi8",
        "Mi8",
    ]
    trip_list = [
        "01_12_12_52",
        "01_12_12_22",
        "01_12_12_31",
        "01_12_12_41",
        "01_12_13_04",
        "01_12_12_11",
    ]

    for phone, trip in zip(phone_list, trip_list):
        plot_trip(phone, trip)


if __name__ == "__main__":
    main()