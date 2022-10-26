import hvplot.pandas
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.geometry import LinearRing
from shapely.ops import nearest_points


def read_geojson(file):
    '''
    read geojson file and return it in crs of UTM Zone 10N, i.e. epsg=32610'''
    gdf = gpd.read_file(file)
    return gdf.to_crs('epsg:32610')


def to_geojson(gdf, file):
    gdf.to_file(file, driver='GeoJSON')


def create_points_along_line(line, delx=25):
    '''
    line is a shapely line string
    create equidistant points (delx (25) apart) along line by projection 
    return a geodata frame in crs of epsg 32610 (UTM Zone 10N)
    '''
    return gpd.GeoDataFrame(data={'transect_dist': np.arange(0, line.length, delx)},
                            geometry=[line.interpolate(x) for x in np.arange(0, line.length, delx)], crs='epsg:32610')


def add_lon_lat(gdfp):
    '''add longitude and latitude for geoviews to work'''
    gdfp_ll = gdfp.copy().to_crs('+init=epsg:4326')
    gdfp['Longitude'] = gdfp_ll.geometry.x
    gdfp['Latitude'] = gdfp_ll.geometry.y
    return gdfp


def convert_geojson_line_to_pts(line_file, pts_file, delx=25):
    df = read_geojson(line_file)
    dfp = create_points_along_line(df.iloc[0].geometry, delx=delx)
    dfp = add_lon_lat(dfp)
    to_geojson(dfp, pts_file)


def create_transect_line(gdf):
    return LineString(gdf.geometry)


def to_geoframe(df):
    ''' convert data frame to geopandas data frame
      converts from lon/lat to utm xy coordinates'''
    gdf = gpd.GeoDataFrame(df)
    gdf['geometry'] = gpd.points_from_xy(df['Longitude'], df['Latitude'], crs="EPSG:4326")
    return gdf.to_crs('EPSG:32610')


def add_transect_dist(gdf, transect_line):
    '''adds transect_dist column for linear referencing info'''
    gdf['transect_dist'] = [transect_line.project(row.geometry) for _, row in gdf.iterrows()]
    return gdf.sort_values(by='transect_dist').reset_index(drop=True)


def get_transect_min_max(gdf):
    return gdf.transect_dist.min(), gdf.transect_dist.max()


def create_begin_end_points_dataframe(gdf, columns=['Longitude', 'Latitude', 'EC', 'transect_dist']):
    pgdf = gdf[columns]
    df_end_points = pd.DataFrame(pgdf.iloc[[0, -1]])
    df_end_points['Station ID'] = ['BEG', 'END']
    return df_end_points


def create_linear_refs_dataframe(gdf, df_end_points):
    return pd.concat([df_end_points, gdf]).sort_values('transect_dist').reset_index(drop=True)


def acquire_observed_values(date_value, df_cdec_15):
    df_ec_vals = df_cdec_15.loc[date_value].T
    df_ec_vals = df_ec_vals.to_frame()
    df_ec_vals.columns = ['Obs EC']
    return df_ec_vals


def add_obs_vals(df_stations, df_ec_vals):
    df_stations = df_ec_vals.join(df_stations.set_index('Station ID'),
                                  how='outer').sort_values(by='transect_dist')
    df_stations.loc[df_stations['EC'].isna(), 'EC'] = df_stations['Obs EC']
    df_stations = df_stations.drop(columns='Obs EC')
    return df_stations


def join_with_station_info(df_ec_vals, df_station_linear_refs):
    return df_ec_vals.join(df_station_linear_refs.set_index('Station ID')).sort_values(by='transect_dist')


def create_transect(gdf, gdfs, data_column='EC', distance_column='transect_dist'):
    '''
    create a linearly referenced data frame with select columns
    gdf is a geo dataframe with datetime, lon/lat, geometry and data_column and distance_column (linear distance projected on a line)
    gdfs is a geo dataframe with lon/lat, geometry and Station ID column and distance_column 

    if close_circle then start transect at first point at first non null Station ID and last point at the same.

    return a data frame indexed on distance_column so that interpolations can use 'index' method to interpolate irregular spaced points
    '''
    columns = ['Longitude', 'Latitude', 'geometry']
    if 'DateTime' in gdf.columns:
        columns.append('DateTime')
    gdf = gdf[columns+[data_column, distance_column]]
    gdfs = gdfs[['Longitude', 'Latitude', 'geometry', 'Station ID', distance_column]]
    df_ec_transect = pd.concat([gdf, gdfs]).set_index(distance_column).sort_index()
    return df_ec_transect


def close_transect(dft, distance_column='transect_dist'):
    '''
    close an input transect data frame indexed by distance_column and sorted by it
    The first non-null Station ID is used to roll the data frame and reset the transect_dist starting from 
    that station id as distance 0 to the final point which is the same as the first with the distances
    '''
    dft = dft.reset_index()
    df1 = dft.reset_index().dropna(subset=['Station ID'])
    dfafter = dft.loc[df1.iloc[0]['index']:].copy()
    dfbefore = dft.loc[:df1.iloc[0]['index']].copy()
    min_td = df1[distance_column].min()
    max_td = dft[distance_column].max()
    dfafter.loc[:, distance_column] -= min_td
    dfbefore.loc[:, distance_column] += max_td
    return pd.concat([dfafter, dfbefore]).reset_index().set_index(distance_column)


def interpolate_transect(dft, df_vals, data_column='EC'):
    ''' interpolate with df_vals defining values indexed at Station ID which is one of the columns on dft
    dft is indexed by which it is then interpolated (usually distance along the transect)'''
    df_vals.columns = ['values']
    dfj = dft.join(df_vals, on='Station ID')
    # run backward and forward fills after interpolation 
    dft[data_column] = dfj['values'].interpolate('index').fillna(method='bfill').fillna(method='ffill').values
    return dft
