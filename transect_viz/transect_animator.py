from turtle import width

from panel.widgets import slider
from . import transect_viz
from . import transect_generator
from . import transect_data
from vtools.functions.filter import godin
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim
import panel as pn


class TransectMap:

    def __init__(self, transect_pts, stations, df_data, close_transect=False, data_column='EC'):
        # store for later use
        self.df_data = df_data
        self.data_column = data_column
        # load transect points
        self.transect_pts = transect_pts
        self.transect_pts[self.data_column] = np.nan  # change data column to data_values
        self.stations = stations
        self.stations = transect_generator.to_geoframe(self.stations)
        # calculate transect_dist for both points and station points using the transect line created from transect points
        transect_line = transect_generator.create_transect_line(self.transect_pts)
        self.transect_pts = transect_generator.add_transect_dist(self.transect_pts, transect_line)
        self.stations = transect_generator.add_transect_dist(self.stations, transect_line)
        # create the combined transect from points and stations
        self.dft = transect_generator.create_transect(
            self.transect_pts, self.stations, data_column=self.data_column, distance_column='transect_dist')
        if close_transect:
            self.dft = transect_generator.close_transect(
                self.dft)  # this is special case for Fabian tract
        # station labels for view
        self.station_labels = transect_viz.labels_for_stations(self.stations)
        self.show_station_labels = True
        self.tidally_filter = False  # show tidally_filter filtered values
        self.df_data_filtered = godin(self.df_data.interpolate('linear', limit=100))

    def view(self, date_value, value_range):
        '''return the holoviews view object'''
        df = self.df_data_filtered if self.tidally_filter else self.df_data
        dfresult = transect_generator.interpolate_transect(
            self.dft, df.loc[date_value].to_frame(), data_column=self.data_column)
        map, legend = transect_viz.map_transect_with_size_and_color(
            dfresult, value_range=value_range)
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


def create_transect_map(transect_pts_file, stations_csv_file, data_csv_file, close_transect=False, var_column_name='EC'):
    # read in data for fabian tract
    transect_pts = transect_data.read_geojson(transect_pts_file)
    stations = transect_data.read_stations_csv_file(stations_csv_file)
    data = transect_data.read_data_csv_file(data_csv_file)

    # configure and return
    return TransectMap(transect_pts, stations, data, close_transect=close_transect, data_column=var_column_name)


class TransectMapComposite:
    def __init__(self, transect_map_list):
        self.transect_map_list = transect_map_list
        self.stations = pd.concat([tmap.stations for tmap in self.transect_map_list])
        # station labels for view
        self.station_labels = transect_viz.labels_for_stations(self.stations)
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

    def __init__(self, transect_map_list, sdate, edate, stations, station_display_info, flow_data, barrier_info):
        #
        self.overlay = None
        #
        self.transect_map_list = transect_map_list
        self.tmapc = TransectMapComposite(transect_map_list)
        self.sdate = pd.to_datetime(sdate)
        self.edate = pd.to_datetime(edate)
        self.flow_data = flow_data
        # load stations
        self.stations = transect_viz.add_in_station_info(stations, station_display_info)
        self.stations = transect_generator.to_geoframe(self.stations)
        self.tiles = hv.element.tiles.CartoLight().redim(x='Longitude', y='Latitude')
        self.dfbarrier = barrier_info
        self.setup()

    def setup(self):
        self.flow_data = self.flow_data.interpolate(limit=10)  # fill nans
        self.flow_data_filtered = godin(self.flow_data.interpolate('linear', limit=100))

    def create_transect_map(self, date_value, value_range):
        return self.tmapc.view(date_value, value_range)

    def create_vectorfield_map(self, date_value, mag_factor, tidal_filter):
        dfflow = self.flow_data_filtered if tidal_filter else self.flow_data
        dfv = transect_viz.create_vector_field(
            date_value, dfflow, self.stations, mag_column='flow')
        vfmap = transect_viz.create_vector_field_map(dfv,
                                                     angle_column='angle', mag_column='flow',
                                                     mag_factor=mag_factor, line_width=6, format_str='.0f')
        return vfmap

    def show_transect_map(self, date_value, show_station_labels=True, value_range=None, tidal_filter=False, mag_factor=1):
        self.tmapc.show_station_labels = show_station_labels
        self.tmapc.set_tidally_filter(tidal_filter)
        tmap = self.create_transect_map(date_value, value_range)
        dfflow = self.flow_data_filtered if tidal_filter else self.flow_data
        fvdmap = self.create_vectorfield_map(date_value, mag_factor=mag_factor, tidal_filter=tidal_filter)
        bpts = self.show_barrier_pts_map(date_value)
        overlay = (self.tiles * tmap * fvdmap * bpts).opts(title=date_value, framewise=False)
        return overlay

    def show_barrier_pts_map(self, date_value):
        return transect_viz.create_barrier_marks(self.dfbarrier, date_value)

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
        slider_width = 300
        # assuming 15 min data, so step should be about 1 tidal cycle
        date_player = pn.widgets.DiscretePlayer(
            name='Date Player', value=time_array[len(time_array) // 2], options=time_array,
            interval=1500, step=1, width=slider_width)
        self.date_player = date_player  # keep this reference to change its settings later
        date_slider = pn.widgets.DateSlider(name='Date Slider', start=pd.to_datetime(
            time_array[0]), end=pd.to_datetime(time_array[-1]), width=slider_width)

        date_time_selector = pn.widgets.Select(
            name='Datetime Selector', options=time_array)

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
            name='Value Range Selector', start=0, end=2000, value=(300, 800), width=slider_width)

        vector_mag_factor_slider = pn.widgets.FloatSlider(
            name='Vector Magnitude Factor', start=0, end=10, value=1.0, step=0.1, width=slider_width)

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
        dmap = dmap.opts(framewise=False, responsive=True)

        tsmap = hv.DynamicMap(pn.bind(self.ec_plot), streams=dict(
            date_value=date_player, tidal_filter=tidal_filter_box))

        tsflowmap = hv.DynamicMap(pn.bind(self.flow_plot), streams=dict(
                date_value=date_player, tidal_filter=tidal_filter_box))

        vlinemap = hv.DynamicMap(pn.bind(self.date_line), streams=dict(date_value=date_player))

        self.widget_area = pn.Column(date_player, date_time_selector,  # date_slider,
                                     value_range_slider, vector_mag_factor_slider,
                                     pn.Row(show_station_labels_box, tidal_filter_box))

        #self.ts_area = pn.Column((tsmap * vlinemap), (tsflowmap * vlinemap))

        #side_panel = pn.Column(self.widget_area, self.ts_area)#, sizing_mode='stretch_both')
        side_panel = pn.GridSpec(sizing_mode='stretch_height', mode='error')
        side_panel[0:1, 0:8] = date_player
        side_panel[1:2, 0:8] = date_time_selector
        side_panel[2:3, 0:8] = value_range_slider
        side_panel[3:4, 0:8] = vector_mag_factor_slider
        side_panel[4:5, 0:4] = pn.Row(show_station_labels_box, tidal_filter_box)
        side_panel[6:10, 0:7] = tsmap*vlinemap
        side_panel[11:15, 0:7] = tsflowmap*vlinemap
        return pn.Row(dmap, background='green'), side_panel
        #self.main_panel.objects = [pn.Row(dmap, background='green')]
        #self.app.sidebar.objects = [side_panel]

    def create_app(self):
        self.app = pn.template.BootstrapTemplate(
            title='Visualization of EC in South Delta', sidebar_width=600)
        main_panel, side_panel = self.setup_main_panel()
        self.app.main.append(main_panel)
        self.app.sidebar.objects=[side_panel]
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
