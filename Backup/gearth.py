# -*- coding: UTF-8 -*-
"""
Usage:
  gearth.py [options]

Options:
  -g <gngga_file> --gngga <gngga_file>  GNGGA文件
  -f <fix_file> --fix <fix_file>  Fix文件
  -o <outdir>, --outdir <outdir>  输出目录 [default: ./]
  -s <suffix>, --suffix <suffix>  后缀 [default: ]
"""
from docopt import docopt
import ipdb as pdb
import simplekml
import pandas as pd
import os
pd.set_option('display.float_format','{:.4f}'.format)

'''
point_list: [timestamp, lat, lon]
'''
def draw_points(kml, point_list, color='bf00ff00'):
    # count = 0
    for timestamp, lat, lon in point_list:
        pnt = kml.newpoint()
        pnt.name = ""
        pnt.description = timestamp
        pnt.coords = [(lon, lat)]
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png'
        pnt.style.iconstyle.scale = 0.6
        pnt.style.iconstyle.color = color

        # count += 1
        # if count > 2:
        #     break


def main():
    # arguments = docopt(__doc__)

    # gngga_file = arguments.gngga
    # fix_file = arguments.fix
    # out_dir = arguments.outdir
    # suffix = arguments.suffix

    BaseDir = fr"/home/wjk/Workspace/Datasets/IGR/IGR230312/processed/Oneplus9pro"
    for trip in os.listdir(BaseDir):
        gngga_file = f"{BaseDir}/{trip}/GNGGA.csv"
        kml = simplekml.Kml(open=1)
        gngga_df = pd.read_csv(gngga_file)
        point_list = gngga_df[['utcTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']].values
        draw_points(kml, point_list, 'ff5053ef')
        kml.save(f"{BaseDir}/{trip}/Traj.kml")

if __name__ == "__main__":
    main()


