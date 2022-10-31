import math
import numpy as np
import pandas as pd
import geopandas as gpd


def load_transect_file(fname='./Transect_20220302_Fabian.csv'):
    return pd.read_csv(fname)


def read_station_display_info(station_info_file='station_info_file.csv'):
    '''reads the station_info_file indexed with first column with 'Station ID' (same as dfs stations)
    adds in that info with defaults (angle=180, everything else 0) and returns a data frame
    '''
    dfinfo = pd.read_csv(station_info_file, index_col=0)
    dfinfo.dtype = float
    dfinfo['angle'] = dfinfo['angle'].fillna(180)
    dfinfo['angle'] = dfinfo['angle'] * math.pi / 180
    dfinfo = dfinfo.fillna(0.)
    return dfinfo


def read_barriers_info(file='barriers.csv'):
    dfb = pd.read_csv(file)
    dfb = dfb.astype({'UTMx': 'float', 'UTMy': 'float',
                     'datein': np.datetime64, 'dateout': np.datetime64})
    gdfb = gpd.GeoDataFrame(dfb, geometry=gpd.points_from_xy(
        dfb['UTMx'], dfb['UTMy'], crs='+init=epsg:32610'))
    gdfb = gdfb.to_crs('EPSG:4326')
    dfb['Longitude'] = gdfb.geometry.x
    dfb['Latitude'] = gdfb.geometry.y
    return dfb.drop(columns=['geometry'])  # gdfb


def read_geojson(file, crs='epsg:32610'):
    '''
    read geojson file and return it in default crs of UTM Zone 10N, i.e. epsg=32610'''
    gdf = gpd.read_file(file)
    return gdf.to_crs(crs)


def to_geojson(gdf, file):
    '''write out geo dataframe to file using GeoJSON driver'''
    gdf.to_file(file, driver='GeoJSON')


def convert_geojson_line_to_pts(line_file, pts_file, delx=25):
    df = read_geojson(line_file)
    from . import transect_generator
    dfp = transect_generator.create_points_along_line(df.iloc[0].geometry, delx=delx)
    dfp = transect_generator.add_lon_lat(dfp)
    to_geojson(dfp, pts_file)


def read_stations_csv_file(stations_csv_file):
    return pd.read_csv(stations_csv_file)


def read_data_csv_file(data_csv_file):
    df = pd.read_csv(data_csv_file, index_col=0, parse_dates=True)
    return df.asfreq(pd.infer_freq(df.index))
