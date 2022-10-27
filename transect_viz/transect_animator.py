from . import transect_viz
from . import transect_generator
from vtools.functions.filter import godin
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim
import panel as pn
# SIZING_MODE='scale_both'
# pn.extension(sizing_mode='stretch_width')


def filter_quantile(dfdata, column):
    dfg = dfdata[column]
    dfg[dfg > dfg.quantile(0.99) * 1.2] = np.nan
    dfg[dfg < dfg.quantile(0.01) * 0.8] = np.nan


def clean_data(dfdata):
    for c in dfdata.columns:
        filter_quantile(dfdata, c)


class TransectMap:

    def __init__(self, points_geojson_file, station_csv_file, data_csv_file, sdate, edate, value_range, close_transect=False, data_column='EC'):
        # store for later use
        self.sdate = sdate
        self.edate = edate
        self.value_range = value_range
        self.data_csv_file = data_csv_file
        self.data_column = data_column
        # load transect points
        self.gdf = transect_generator.read_geojson(points_geojson_file)
        self.gdf[self.data_column] = np.nan  # change data column to data_values
        # load stations
        self.dfs = pd.read_csv(station_csv_file)
        self.gdfs = transect_generator.to_geoframe(self.dfs)
        # calculate transect_dist for both points and station points using the transect line created from transect points
        transect_line = transect_generator.create_transect_line(self.gdf)
        self.gdf = transect_generator.add_transect_dist(self.gdf, transect_line)
        self.gdfs = transect_generator.add_transect_dist(self.gdfs, transect_line)
        # create the combined transect from points and stations
        self.dft = transect_generator.create_transect(
            self.gdf, self.gdfs, data_column=self.data_column, distance_column='transect_dist')
        if close_transect:
            self.dft = transect_generator.close_transect(
                self.dft)  # this is special case for Fabian tract
        # station labels for view
        self.station_labels = transect_viz.labels_for_stations(self.gdfs)
        self.show_station_labels = True
        self.tidally_filter = False  # show tidally_filter filtered values

    def setup(self):
        self.df_data = pd.read_csv(self.data_csv_file, index_col=0, parse_dates=True)
        self.df_data = self.df_data.asfreq(pd.infer_freq(self.df_data.index))
        self.df_data_filtered = godin(self.df_data.interpolate('linear', limit=100))

    def view(self, date_value):
        '''return the holoviews view object'''
        df = self.df_data_filtered if self.tidally_filter else self.df_data
        dfresult = transect_generator.interpolate_transect(
            self.dft, df.loc[date_value].to_frame(), data_column=self.data_column)
        map, legend = transect_viz.map_transect_with_size_and_color(
            dfresult, value_range=self.value_range)
        overlay = map.opts(framewise=False, colorbar=True)
        if self.show_station_labels:
            overlay = overlay * self.station_labels.opts(framewise=False)
        return overlay

    def ec_plot(self, date_value):
        dval = pd.to_datetime(date_value)
        tdval = pd.Timedelta('1D')
        df = self.df_data_filtered if self.tidally_filter else self.df_data
        return df[dval - tdval:dval + tdval].hvplot.line(ylabel='EC',
                                                     color=list(
                                                         hv.Cycle.default_cycles['Colorblind'])
                                                         ).opts(title='Measured EC with line at current time')


class TransectMapComposite:
    def __init__(self, transect_map_list):
        self.transect_map_list = transect_map_list
        for tmap in self.transect_map_list:
            tmap.setup()
        self.gdfs = pd.concat([tmap.gdfs for tmap in self.transect_map_list])
        # station labels for view
        self.station_labels = transect_viz.labels_for_stations(self.gdfs)
        self.df_data = pd.concat([tmap.df_data for tmap in self.transect_map_list], axis=1)
        # remove duplicate columns
        self.df_data = self.df_data.loc[:, ~self.df_data.columns.duplicated()].copy()
        self.df_data_filtered = godin(self.df_data.interpolate('linear', limit=100))
        self.show_station_labels = True
        self.set_tidally_filter(False)

    def set_tidally_filter(self, do_tidally_filter):
        self.tidally_filter = do_tidally_filter
        for tmap in self.transect_map_list:
            tmap.tidally_filter = self.tidally_filter

    def view(self, date_value, value_range):
        dfresults = []
        for tmap in self.transect_map_list:
            df = tmap.df_data_filtered if self.tidally_filter else tmap.df_data
            dfresult = transect_generator.interpolate_transect(
                tmap.dft, df.loc[date_value].to_frame(), data_column=tmap.data_column)
            dfresults.append(dfresult)
        dfresult = pd.concat(dfresults)
        ptsmap, legend = transect_viz.map_transect_with_size_and_color(
            dfresult, value_range=value_range)
        overlay = ptsmap.opts(framewise=False, colorbar=True)
        if self.show_station_labels:
            overlay = overlay * self.station_labels.opts(framewise=False)
        return overlay

    def ec_plot(self, date_value):
        dval = pd.to_datetime(date_value)
        tdval = pd.Timedelta('10D') if self.tidally_filter else pd.Timedelta('1D')
        df = self.df_data_filtered if self.tidally_filter else self.df_data
        return df[dval - tdval:dval + tdval].hvplot.line(ylabel='EC',
                                                     color=list(
                                                         hv.Cycle.default_cycles['Colorblind'])
                                                         ).opts(title=f'Measured EC @ {date_value}'
                                                                ).opts(opts.Curve(show_legend=False))


class GeneratedECMapAnimator:

    def __init__(self, transect_map_list, sdate, edate, flow_stations_csv_file, flow_data_csv_file, barrier_file, station_info_file):
        #
        self.overlay = None
        #
        self.transect_map_list = transect_map_list
        self.tmapc = TransectMapComposite(transect_map_list)
        self.sdate = pd.to_datetime(sdate)
        self.edate = pd.to_datetime(edate)
        self.flow_data_csv_file = flow_data_csv_file
        # load stations
        self.dfs = pd.read_csv(flow_stations_csv_file)
        self.dfs = transect_viz.add_in_station_info(self.dfs, station_info_file=station_info_file)
        self.gdfs = transect_generator.to_geoframe(self.dfs)
        self.tiles = hv.element.tiles.CartoLight().redim(x='Longitude', y='Latitude')
        self.barrier_file = barrier_file
        self.load_data()

    def load_data(self):
        self.flow_data = pd.read_csv(self.flow_data_csv_file, index_col=0, parse_dates=True)
        self.flow_data = self.flow_data.asfreq(pd.infer_freq(self.flow_data.index))
        self.flow_data = self.flow_data.interpolate()  # fill nans
        self.flow_data_filtered = godin(self.flow_data.interpolate('linear', limit=100))
        self.load_barriers()

    def load_barriers(self):
        self.dfbarrier = transect_viz.read_barriers_info(self.barrier_file)

    def create_transect_map(self, date_value, value_range):
        return self.tmapc.view(date_value, value_range)

    def create_vectorfield_map(self, date_value, dfflow, gdfs, mag_factor):
        vfmap = transect_viz.create_vector_field_map(transect_viz.create_vector_field(date_value, dfflow, gdfs, mag_column='flow'),
                                                     angle_column='angle', mag_column='flow',
                                                     mag_factor=mag_factor, line_width=6, format_str='.0f')
        return vfmap

    def show_transect_map(self, date_value, show_station_labels=True, value_range=None, tidal_filter=False, mag_factor=1):
        self.tmapc.show_station_labels = show_station_labels
        self.tmapc.set_tidally_filter(tidal_filter)
        tmap = self.create_transect_map(date_value, value_range)
        dfflow = self.flow_data_filtered if tidal_filter else self.flow_data
        fvdmap = self.show_flow_vectors_map(date_value, dfflow, self.gdfs, mag_factor=mag_factor)
        bpts = self.show_barrier_pts_map(date_value, self.dfbarrier)
        overlay = (self.tiles * tmap * fvdmap * bpts).opts(title=date_value,
                   frame_width=1000, framewise=False)
        return overlay

    def show_flow_vectors_map(self, date_value, dfflow, gdfs, mag_factor):
        return self.create_vectorfield_map(date_value, dfflow, gdfs, mag_factor=mag_factor)

    def show_barrier_pts_map(self, date_value, dfbarrier):
        return transect_viz.create_barrier_marks(dfbarrier, date_value)

    def ec_plot(self, date_value, tidal_filter=False):
        self.tmapc.set_tidally_filter(tidal_filter)
        return self.tmapc.ec_plot(date_value)

    def flow_plot(self, date_value, tidal_filter=False):
        self.tmapc.set_tidally_filter(tidal_filter)
        dval = pd.to_datetime(date_value)
        tdval = pd.Timedelta('10D') if tidal_filter else pd.Timedelta('1D')
        df = self.flow_data_filtered if tidal_filter else self.flow_data
        return df[dval - tdval:dval + tdval].hvplot.line(ylabel='Flow',
                                                     color=list(
                                                         hv.Cycle.default_cycles['Colorblind'])
                                                         ).opts(title=f'Measured Flow @ {date_value}'
                                                                ).opts(opts.Curve(show_legend=False)).redim(Variable='Flow', value='flow')

    def date_line(self, date_value):
        dval = pd.to_datetime(date_value)
        return hv.VLine(x=pd.to_datetime(dval))

    def setup_main_panel(self):
        time_array = [x.strftime('%Y-%m-%d %H:%M')
                      for x in pd.date_range(start=self.sdate, end=self.edate, freq='15T')]
        self.time_array = time_array
        # assuming 15 min data, so step should be about 1 tidal cycle
        date_player = pn.widgets.DiscretePlayer(
            name='Date Player', value=time_array[len(time_array) // 2], options=time_array, interval=1500, step=1, width=400)
        self.date_player = date_player  # keep this reference to change its settings later
        date_slider = pn.widgets.DateSlider(name='Date Slider', start=pd.to_datetime(
            time_array[0]), end=pd.to_datetime(time_array[-1]))

        date_time_selector = pn.widgets.Select(
            name='Datetime Selector', options=time_array, width=400)

        date_player.link(date_time_selector, value='value',
                         bidirectional=True)  # jslink doesn't work?

        def sync_player(target, event):
            hhmm = str(target.value).split(' ')[1]
            try:
                target.value = event.new.strftime('%Y-%m-%d ' + hhmm)
            except:
                target.value = time_array[-1]  # set to end of time array

        def sync_slider(target, event):
            target.value = pd.to_datetime(event.new)

        _ = date_slider.link(date_player, callbacks={'value': sync_player})
        _ = date_player.link(date_slider, callbacks={'value': sync_slider})

        value_range_slider = pn.widgets.RangeSlider(
            name='Value Range Selector', start=0, end=2000, value=(300, 800), width=400)

        vector_mag_factor_slider = pn.widgets.FloatSlider(
            name='Vector Magnitude Factor', start=0, end=10, value=1.0, step=0.1)

        show_station_labels_box = pn.widgets.Checkbox(name='Show Stations', value=True)

        tidal_filter_box = pn.widgets.Checkbox(name='Tidally Filter', value=False)

        def set_player_stepsize(target, event):
            if event.new:  # if tidal filter box is checked
                target.step = 96  # 1 day step given 15 min data
            else:
                target.step = 1
        tidal_filter_box.link(date_player, callbacks={'value': set_player_stepsize})

        dmap = hv.DynamicMap(pn.bind(self.show_transect_map),
                             streams=dict(date_value=date_player,
                             value_range=value_range_slider,
                             show_station_labels=show_station_labels_box,
                             tidal_filter=tidal_filter_box,
                             mag_factor=vector_mag_factor_slider))
        dmap = dmap.opts(framewise=False)

        tsmap = hv.DynamicMap(pn.bind(self.ec_plot), streams=dict(
            date_value=date_player, tidal_filter=tidal_filter_box)).opts(frame_width=300)

        tsflowmap = hv.DynamicMap(pn.bind(self.flow_plot), streams=dict(
                date_value=date_player, tidal_filter=tidal_filter_box)).opts(frame_width=300)

        vlinemap = hv.DynamicMap(pn.bind(self.date_line), streams=dict(date_value=date_player))

        self.widget_area = pn.Column(date_player, date_time_selector, date_slider,
                                     value_range_slider, vector_mag_factor_slider,
                                     show_station_labels_box, tidal_filter_box)
        self.ts_area = pn.Column(tsmap * vlinemap, tsflowmap * vlinemap)

        self.main_panel.objects = [pn.Row(dmap, width=1000)]
        self.side_panel.objects = [self.widget_area, self.ts_area]

        self.main_panel.loading = False
        # main_panel

    def create_main_panel(self):
        self.main_panel = pn.Column()  # sizing mode flexibility causes issue in geoviews map
        explanation = pn.pane.Markdown('''
        # Generated EC at points along the channel
         EC values generated from observed data stations and linear interpolation. The station locations are roughly at the labels. 
         The flow arrows are depicted with direction and magnitude indicated by their length.

        The points are colored by the EC values and the point size is also scaled as shown by the value range (larger point size means higher values)
        ''')
        self.main_panel.loading = True
        return self.main_panel

    def create_app(self):
        self.create_main_panel()
        self.app = pn.template.BootstrapTemplate(
            title='Visualization of EC in South Delta', sidebar_width=500)
        self.app.main.append(self.main_panel)
        self.side_panel = pn.Column(width=500)
        self.app.sidebar.append(self.side_panel)
        #self.app.sidebar.append(pn.pane.Markdown("""[Generated EC from Obs. Stations](/)"""))
        # self.app.sidebar.append(pn.pane.Markdown(
        #    """[Measured Transect Visualization](/transect_data/Transect_Viz_20220302_Fabian.html)"""))
        return self.app.servable(title='Generated EC Animator Map')

    def frame2png(self, date_value, dir='images'):
        fname = dir + '/' + date_value.replace(' ', '_').replace(':', '_') + '.png'
        self.main_panel.save(fname)

    def save2png(self, sdate, edate, dir='images'):
        import os
        if not os.path.exists(dir):
            os.mkdir(dir)
        for date_value in self.time_array[1000:1002]:
            print('Saving for ', date_value)
            self.frame2png(date_value, dir=dir)
