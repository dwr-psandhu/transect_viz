import math
from operator import index
from matplotlib.pyplot import xlabel
from vtools.functions.filter import godin
import pandas as pd
import numpy as np
#
import geoviews as gv
from .nice_scale import NiceScale
from . import transect_data
#
import hvplot.pandas
import holoviews as hv
from holoviews import dim, opts
#
import panel as pn
pn.extension()


def points_transect_by_point_size(df, data_column='EC'):
    return df.hvplot.points('Longitude', 'Latitude', geo=True, s=data_column, hover=True, framewise=False)


def points_transect_by_color(df, cmap='rainbow4', tiles='CartoLight', data_column='EC'):
    pts = gv.Points(df, kdims=['Longitude', 'Latitude'], vdims=[
                    data_column]).opts(opts.Points(color=data_column))
    return pts.opts(cmap=cmap, colorbar=True)


def get_tile_layer(tiles='CartoLight'):
    return hv.element.tile_sources.get(tiles)()


def create_nice_scale(df, data_column='EC', value_range=None):
    if value_range is None:
        value_range = (df[data_column].min(), df[data_column].max())
    ns = NiceScale(*value_range)
    return ns


def points_transect_by_size_and_color(df, data_column='EC', cmap='rainbow4',
                                      value_range=None, max_point_size=25, min_point_size=1):
    ns = create_nice_scale(df, data_column=data_column, value_range=value_range)
    points_dim = dim(data_column).norm(limits=(ns.niceMin, ns.niceMax)).clip(0, 1) * \
        (max_point_size - min_point_size) + min_point_size
    vdims = [data_column]
    if 'DateTime' in df.columns:
        vdims.append('DateTime')
    scpts = gv.Points(df, kdims=['Longitude', 'Latitude'], vdims=vdims).opts(
        opts.Points(tools=['hover'], framewise=False, color=data_column, size=points_dim, alpha=0.5, cmap=cmap, clim=(ns.niceMin, ns.niceMax)))
    return scpts


def create_scaled_points_legend(ns, max_point_size=25, min_point_size=1):
    data = {
        'x': np.zeros(ns.maxTicks),
        'y': np.arange(1, 0, -1. / ns.maxTicks),
        'size': np.array([round((ns.tickSpacing * i) / (ns.niceMax - ns.niceMin) * max_point_size + min_point_size) for i in range(ns.maxTicks)]),
        'labels': [str(round(a * ns.tickSpacing + ns.niceMin)) for a in range(ns.maxTicks)]
    }
    values = np.array(data['size'])
    lbls = hv.Labels(data, kdims=['x', 'y'], vdims='labels').opts(
        opts.Labels(text_align='left', text_baseline='middle', xoffset=0.4))
    pts = hv.Points(data, kdims=['x', 'y'], vdims=['size']).opts(
        size='size', ylim=(0, 1.1)).opts(opts.Points(fill_alpha=0))
    return (lbls * pts).opts(xaxis=None, yaxis=None, toolbar=None)


def map_transect_with_size_and_color(df, data_column='EC', cmap='rainbow4',
                                     value_range=None,
                                     max_point_size=25, min_point_size=1, pts_legend_height=150):
    ns = create_nice_scale(df, data_column=data_column, value_range=value_range)
    scpts = points_transect_by_size_and_color(
        df, data_column, value_range=value_range, cmap=cmap, max_point_size=max_point_size, min_point_size=min_point_size)
    pts_legend = create_scaled_points_legend(ns,
        max_point_size=max_point_size, min_point_size=min_point_size).opts(frame_height=pts_legend_height)
    return scpts, pts_legend


def save_map(scmap, fname):
    hv.save(scmap, fname)


def get_start_end_dates(df):
    sdate, edate = df.DateTime.agg(['min', 'max'])
    return pd.to_datetime(sdate), pd.to_datetime(edate)


def get_buffered_start_end_dates(df, time_buffer='2D'):
    '''
    get rounded times with a buffer
    '''
    sdate, edate = get_start_end_dates(df)
    start_date = sdate.floor('1D') - pd.Timedelta(time_buffer)
    end_date = edate.ceil('1D') + pd.Timedelta(time_buffer)
    return start_date, end_date


def points_for_station(dfs):
    return dfs.hvplot.points(x='Longitude', y='Latitude', geo=True, hover_cols=['Station ID'], framewise=False).opts(frame_width=800)


def labels_for_stations(dfs):
    return gv.Labels(dfs, kdims=['Longitude', 'Latitude'], vdims=['Station ID']).opts(opts.Labels(xoffset=10, yoffset=10, text_align='left', framewise=False))


def timespan(start_date, end_date):
    return hv.VSpan(x1=pd.to_datetime(start_date), x2=pd.to_datetime(end_date)).opts(xlim=(start_date - pd.Timedelta('1D'), end_date + pd.Timedelta('1D')))


def plot_data_around_time(start_date, end_date, dflist, data_type, ylabel, title):
    ovl = hv.Overlay([d.hvplot(label=f'{d.columns[0]}').redim(
        **{d.columns[0]:f'{data_type} {d.columns[0]}'}) for d in dflist])
    return ovl.opts(opts.Curve(ylabel=ylabel)).opts(title=title, xlim=(pd.to_datetime(start_date) - pd.Timedelta('1D'), pd.to_datetime(end_date) + pd.Timedelta('1D')))


def get_tidal_filtered(dflist, resample_interval='1H'):
    return [godin(d.resample(resample_interval).mean()) for d in dflist]


def plot_flow_around_time(start_date, end_date, dflist):
    return plot_data_around_time(start_date, end_date, dflist, data_type='Flow', ylabel='Flow (CFS)', title='Flows at locations')


def plot_tidal_filtered_flow_around_time(start_date, end_date, dflist):
    return plot_data_around_time(start_date, end_date, dflist, data_type='Tidal Flow', ylabel='Tidal Flow (cfs)', title='Tidal Filtered Flow at Locations')


def plot_tidal_filtered_velocity_around_time(start_date, end_date, dflist):
    return plot_data_around_time(start_date, end_date, dflist, data_type='Tidal Velocity', ylabel='Tidal Velocity (ft/sec)', title='Tidal Filtered Velocity at Locations')


def plot_velocity_around_time(start_date, end_date, dflist):
    return plot_data_around_time(start_date, end_date, dflist, data_type='Velocity', ylabel='Velocity (ft/sec)', title='Velocity at locations')


def calculate_travel_distances_in_miles(dflist):
    '''
    Given list of velocities in ft/sec
    '''
    travel_distances = [v.cumsum() * 3600 / 5280.0 for v in dflist]


def mean_velocity_over_time(start_date, end_date, dflist):
    dfvel = pd.DataFrame([v[start_date:end_date].mean()[0]
                          for v in dflist], index=[v.columns[0] for v in dflist])
    dfvel.columns = ['mean velocity']
    return dfvel


def merge_mean_velocity_with_stations(dfs, dfother):
    return merge_with_stations(dfs, dfother)


def merge_with_stations(dfs, dfother):
    dfvector = dfs.set_index('Station ID').join(dfother)
    return dfvector


def plot_velocity_vectors(dfvelgis):
    return plot_vectors(dfvelgis, angle_column='angle', mag_column='mean velocity')


def plot_vectors(dfvector, angle_column='angle', mag_column='mag', line_width=10):
    dfvector = dfvector.copy()
    dfvector['Longitude'] += dfvector['arrow_xoffset']
    dfvector['Latitude'] += dfvector['arrow_yoffset']
    vectors = gv.VectorField(dfvector, kdims=['Longitude', 'Latitude'],
        vdims=[angle_column, mag_column]).opts(framewise=False, magnitude=mag_column)  # hover doesn't work with Vectorfields
    vectors = vectors.opts(opts.VectorField(
        alpha=0.85, color='blue', pivot='mid', line_width=line_width, line_cap='round'))
    return vectors


def plot_velocity_labels(dfvelgis):
    return plot_vector_labels(dfvelgis, mag_column='mean velocity', units='ft/s')


def plot_vector_labels(dfvector, mag_column='mag', units='units', format_str='.02f'):
    dfvector = dfvector.copy()
    dfvector['label'] = dfvector[mag_column].map(('{:,' + format_str + '} ' + units).format)
    dfvector['Longitude'] += dfvector['arrow_xoffset'] + dfvector['value_xoffset']
    dfvector['Latitude'] += dfvector['arrow_yoffset'] + dfvector['value_yoffset']
    # .opts(opts.Labels(text_align='left'))  # , xoffset=40, yoffset=40))
    vec_labels = dfvector.hvplot.labels(
        x='Longitude', y='Latitude', text='label', geo=True, hover=False, framewise=False)  # doesn't respect framewise ?
    return vec_labels.opts(framewise=False)


def create_vector_field(date_value, df, dfs, mag_column='mag'):
    dfvector = df.loc[date_value, :].to_frame()
    dfvector.columns = [mag_column]
    dfvectors = merge_with_stations(dfs, dfvector).copy()
    return dfvectors


def get_mag_dim(mag_column):  # not working yet.  # try lognorm() or norm() with abs() and sgn()
    max_vec_length = 1000
    min_vec_length = 100
    vec_dim = dim(mag_column).lognorm(limits=(0, 1000)).clip(0, 1) * \
        (max_vec_length - min_vec_length) + min_vec_length
    return vec_dim


def create_vector_field_map(dfv, angle_column='angle', mag_column='mag', mag_units='', mag_factor=0.1, format_str='.02f', line_width=4):
    dfv = dfv.dropna()
    vecs = plot_vectors(dfv, angle_column=angle_column, mag_column=mag_column)
    vec_dim = dim(mag_column) * mag_factor
    vecs = vecs.opts(magnitude=vec_dim,
                     line_width=line_width, rescale_lengths=False)
    labels = plot_vector_labels(dfv, mag_column=mag_column, units=mag_units, format_str=format_str)
    return vecs.opts(framewise=False)  # FIXME: labels causing issues with framewise=False


def add_in_station_info(dfs, station_display_info):
    return dfs.set_index('Station ID').join(station_display_info).reset_index().fillna(0)


def create_barrier_marks(dfb, date_value):
    dfbnow = dfb[(dfb.datein <= pd.to_datetime(date_value)) &
                 (dfb.dateout >= pd.to_datetime(date_value))]
    return gv.Points(dfbnow, kdims=['Longitude', 'Latitude']).opts(
        marker='square', size=20, color='black', framewise=False)


def show_map(dfresult, value_range=None):
    map, legend = map_transect_with_size_and_color(dfresult, value_range=value_range)
    carto_light_tiles = get_tile_layer()
    return carto_light_tiles * map.opts(frame_width=800, colorbar=True) + legend


def create_transect_profile(df):
    from . import transect_generator
    gdf = transect_generator.to_geoframe(df)
    transect_line = transect_generator.create_transect_line(gdf)
    gdf = transect_generator.add_transect_dist(gdf, transect_line)
    return gdf.hvplot.scatter(x='transect_dist', y='EC').opts(
        title='Profile of Specific Conductivity along Transect ', xlabel='Transect Distance (ft)', ylabel='Specific Conductivity')

def show_begin_end_pts_with_labels(df):
    begin_end = df.iloc[[0, -1]][['Longitude', 'Latitude', 'DateTime']].copy()
    begin_end['Label'] = ['Start', 'End']
    print(begin_end)
    plt = begin_end.iloc[0:1, :].hvplot.labels(
        x='Longitude', y='Latitude', text='Label', geo=True).opts(opts.Labels(yoffset=300))
    plt = plt * begin_end.iloc[-1:, :].hvplot.labels(
        x='Longitude', y='Latitude', text='Label', geo=True).opts(opts.Labels(yoffset=-300))
    plt = plt * begin_end.hvplot.points(geo=True)
    return plt

def make_list(df):
    return [df[[c]] for c in df.columns]

def generate_transect_vizualization(transect_file, transect_viz_outfile, transect_viz_title, 
        stations_csv_file, station_info_file,
        flow_data_file, velocity_data_file, stage_data_file, ec_data_file, data_column='Sonde1_SpCond'):
    #cdec_stations=['ORM', 'ODM', 'OLD', 'GLE', 'GLC']
    df = transect_data.load_transect_file(transect_file)
    df['EC'] = df[data_column]  # 'EC' is mapped from Sonde1_SpCond
    dfs = transect_data.read_stations_csv_file(stations_csv_file=stations_csv_file)
    station_info = transect_data.read_station_display_info(station_info_file)
    dfs = add_in_station_info(dfs, station_info)
    transect_profile_plot = create_transect_profile(df)
    station_map_with_tiles = get_tile_layer('CartoLight') *\
    points_for_station(dfs) *\
    labels_for_stations(dfs).opts(frame_height=350)
    sdate, edate = get_start_end_dates(df)
    bsdate, bedate = get_buffered_start_end_dates(df)

    cdec_flows = transect_data.read_data_csv_file(flow_data_file)
    cdec_flows = make_list(cdec_flows)
    cdec_vels = transect_data.read_data_csv_file(velocity_data_file)
    cdec_vels = make_list(cdec_vels)
    collection_span = timespan(sdate, edate)
    flow_plot = plot_flow_around_time(sdate, edate, cdec_flows) * collection_span
    velocity_plot = plot_velocity_around_time(sdate, edate, cdec_vels) * collection_span
    dftidalvelgis = merge_mean_velocity_with_stations(
        dfs, mean_velocity_over_time(sdate, edate, get_tidal_filtered(cdec_vels)))
    vel_tidal_vectors = plot_velocity_vectors(dftidalvelgis) * plot_velocity_labels(dftidalvelgis)
    vel_tidal_map_with_station = station_map_with_tiles * vel_tidal_vectors
    vel_tidal_map_with_station.opts(
        frame_height=350, title='Mean Tidally filtered Velocity over collection period', xlabel='Longitude', ylabel='Latitude')
    dfvelgis = merge_mean_velocity_with_stations(
        dfs, mean_velocity_over_time(sdate, edate, cdec_vels))
    vel_vectors = plot_velocity_vectors(dfvelgis) * plot_velocity_labels(dfvelgis)
    vel_map_with_station = station_map_with_tiles * vel_vectors
    vel_map_with_station.opts(
        frame_height=350, title='Mean Velocity over collection period', xlabel='Longitude', ylabel='Latitude')

    vel_map_with_station * show_begin_end_pts_with_labels(df)
    map1, legend1 = map_transect_with_size_and_color(df)
    map1 = get_tile_layer('CartoLight') * map1.opts(xlabel='Longitude', ylabel='Latitude')
    map1 = map1.opts(opts.Points(colorbar=True)).opts(title='Transect EC by size and color')
    vel_map2 = pn.Row(vel_map_with_station.opts(frame_width=700, frame_height=300),
                      vel_tidal_map_with_station.opts(frame_width=700, frame_height=300))
    map_panel = pn.Row(pn.Column(pn.Row(map1.opts(frame_width=700, frame_height=300), legend1.opts(title='Point Size Legend')),
                                pn.Row(vel_map_with_station.opts(frame_width=700, frame_height=300),
                                    transect_profile_plot)))
    tidal_plot = plot_tidal_filtered_flow_around_time(
        sdate, edate, get_tidal_filtered(cdec_flows)) * collection_span
    tidal_vel_plot = plot_tidal_filtered_velocity_around_time(
        sdate, edate, get_tidal_filtered(cdec_vels)) * collection_span
    flow_velocity_panel = pn.Column(pn.Row(flow_plot, tidal_plot),
                                    pn.Row(velocity_plot, tidal_vel_plot))
    cdec_stage = transect_data.read_data_csv_file(stage_data_file)
    cdec_stage = make_list(cdec_stage)
    cdec_ec = transect_data.read_data_csv_file(ec_data_file)
    cdec_ec = make_list(cdec_ec)

    stage_plot = plot_data_around_time(
        sdate, edate, cdec_stage, data_type='stage', ylabel='Stage (ft)', title='Stage') * timespan(sdate, edate)

    tidal_stage_plot = plot_data_around_time(sdate, edate, get_tidal_filtered(
        cdec_stage), data_type='stage', ylabel='Stage (ft)', title='Stage tidally filtered') * timespan(sdate, edate)

    ec_plot = plot_data_around_time(
        sdate, edate, cdec_ec, data_type='ec', ylabel='EC (umhos/cm)', title='EC') * timespan(sdate, edate)

    tidal_ec_plot = plot_data_around_time(sdate, edate, get_tidal_filtered(
        cdec_ec), data_type='ec', ylabel='Tidally filtered EC (umhos/cm)', title='EC') * timespan(sdate, edate)
    stage_ec_panel = pn.Column(pn.Row(stage_plot, tidal_stage_plot), pn.Row(ec_plot, tidal_ec_plot))
    full_panel = pn.Column(map_panel, flow_velocity_panel, stage_ec_panel)
    pn.io.save.save(full_panel, transect_viz_outfile, title=transect_viz_title)
