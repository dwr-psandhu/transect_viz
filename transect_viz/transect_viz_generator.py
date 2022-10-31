import warnings
warnings.filterwarnings('ignore')
from . import transect_cdec_data
from . import transect_data
from . import transect_generator
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim
import panel as pn
pn.extension()

def create_transect_profile(df):
    gdf = transect_generator.to_geoframe(df)
    transect_line = transect_generator.create_transect_line(gdf)
    gdf = transect_generator.add_transect_dist(gdf, transect_line)
    return gdf.hvplot.scatter(x='transect_dist', y='EC').opts(
        title='Profile of Specific Conductivity along Transect ', xlabel='Transect Distance (ft)', ylabel='Specific Conductivity')


def generate_transect_vizualization(transect_file, stations_csv_file, station_info_file, data_column='Sonde1_SpCond'):
    df=transect_data.load_transect_file(transect_file)
    df['EC'] = df[data_column] # 'EC' is mapped from Sonde1_SpCond
    #cdec_stations=['ORM', 'ODM', 'OLD', 'GLE', 'GLC']
    #station_info_file='../data/station_info_file.csv'
    dfs = transect_data.read_stations_csv_file(stations_csv_file=stations_csv_file)
    station_info = transect_data.read_station_display_info(station_info_file)
    dfs = add_in_station_info(dfs, station_info)
    transect_profile_plot = create_transect_profile(df)
    station_map_with_tiles = transect_viz.get_tile_layer('CartoLight') *\
    transect_viz.points_for_station(dfs) *\
    transect_viz.labels_for_stations(dfs).opts(frame_height=350)
    sdate, edate = transect_viz.get_start_end_dates(df)
    bsdate, bedate = transect_viz.get_buffered_start_end_dates(df)
    def make_list(df):
        return [df[[c]] for c in df.columns]
    cdec_flows = transect_cdec_data.get_cdec_data_cached(bsdate, bedate, cdec_stations, data_type='flow')
    cdec_flows = make_list(cdec_flows)
    cdec_vels = transect_cdec_data.get_cdec_data_cached(bsdate, bedate, cdec_stations, data_type='vel')
    cdec_vels = make_list(cdec_vels)
    collection_span = transect_viz.timespan(sdate, edate)
    flow_plot = transect_viz.plot_flow_around_time(sdate,edate,cdec_flows)*collection_span
    velocity_plot = transect_viz.plot_velocity_around_time(sdate,edate,cdec_vels)*collection_span
dftidalvelgis = transect_viz.merge_mean_velocity_with_stations(dfs, transect_viz.mean_velocity_over_time(sdate,edate,transect_viz.get_tidal_filtered(cdec_vels)))
vel_tidal_vectors = transect_viz.plot_velocity_vectors(dftidalvelgis)*transect_viz.plot_velocity_labels(dftidalvelgis)
vel_tidal_map_with_station = station_map_with_tiles*vel_tidal_vectors
vel_tidal_map_with_station.opts(frame_height=350, title='Mean Tidally filtered Velocity over collection period', xlabel='Longitude', ylabel='Latitude')
dfvelgis = transect_viz.merge_mean_velocity_with_stations(dfs, transect_viz.mean_velocity_over_time(sdate,edate,cdec_vels))
vel_vectors = transect_viz.plot_velocity_vectors(dfvelgis)*transect_viz.plot_velocity_labels(dfvelgis)
vel_map_with_station = station_map_with_tiles*vel_vectors
vel_map_with_station.opts(frame_height=350, title='Mean Velocity over collection period', xlabel='Longitude', ylabel='Latitude')
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
vel_map_with_station*show_begin_end_pts_with_labels(df)
map1, legend1 = map_with_legend_by_color_size(df)
map1 = transect_viz.get_tile_layer('CartoLight')*map1.opts(xlabel='Longitude', ylabel='Latitude')
map1 = map1.opts(opts.Points(colorbar=True)).opts(title='Transect EC by size and color')
vel_map2=pn.Row(vel_map_with_station.opts(frame_width=700, frame_height=300), vel_tidal_map_with_station.opts(frame_width=700, frame_height=300))
map_panel = pn.Row(pn.Column(pn.Row(map1.opts(frame_width=700, frame_height=300),legend1.opts(title='Point Size Legend')),
                             pn.Row(vel_map_with_station.opts(frame_width=700, frame_height=300),
                                   transect_profile_plot)))
tidal_plot = transect_viz.plot_tidal_filtered_flow_around_time(sdate, edate, transect_viz.get_tidal_filtered(cdec_flows))*collection_span
tidal_vel_plot = transect_viz.plot_tidal_filtered_velocity_around_time(sdate, edate, transect_viz.get_tidal_filtered(cdec_vels))*collection_span
flow_velocity_panel = pn.Column(pn.Row(flow_plot, tidal_plot), pn.Row(velocity_plot, tidal_vel_plot))
cdec_stage = transect_cdec_data.get_cdec_data_cached(
    bsdate, bedate, cdec_stations, data_type='stage')
cdec_stage = make_list(cdec_stage)
cdec_ec = transect_cdec_data.get_cdec_data_cached(
    bsdate, bedate, cdec_stations, data_type='ec')
cdec_ec = make_list(cdec_ec)

stage_plot = transect_viz.plot_data_around_time(
    sdate, edate, cdec_stage, data_type='stage', ylabel='Stage (ft)', title='Stage')*transect_viz.timespan(sdate, edate)

tidal_stage_plot = transect_viz.plot_data_around_time(sdate, edate, transect_viz.get_tidal_filtered(
    cdec_stage), data_type='stage', ylabel='Stage (ft)', title='Stage tidally filtered')*transect_viz.timespan(sdate, edate)

ec_plot = transect_viz.plot_data_around_time(
    sdate, edate, cdec_ec, data_type='ec', ylabel='EC (umhos/cm)', title='EC')*transect_viz.timespan(sdate, edate)

tidal_ec_plot = transect_viz.plot_data_around_time(sdate, edate, transect_viz.get_tidal_filtered(
    cdec_ec), data_type='ec', ylabel='Tidally filtered EC (umhos/cm)', title='EC')*transect_viz.timespan(sdate, edate)
stage_ec_panel = pn.Column(pn.Row(stage_plot, tidal_stage_plot), pn.Row(ec_plot, tidal_ec_plot))
full_panel = pn.Column(map_panel, flow_velocity_panel, stage_ec_panel)
pn.io.save.save(full_panel,'../transect_reports/Transect_Viz_20220302_Fabian.html',title='Fabian Tract 2022-03-02 EC Transect Visualization Panel')



