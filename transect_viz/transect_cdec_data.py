from warnings import warn
from . import cdec
import pandas as pd
import os

cache_dir = 'cache'


def ensure_dir(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


def _get_raw_cdec_data(start_date, end_date, cdec_stations, data_type='flow'):
    cr = cdec.Reader()
    if data_type.lower() == 'flow':
        event_type = 20
    elif data_type.lower() == 'vel':
        event_type = 21
    elif data_type.lower() == 'stage':
        event_type = 1
    elif data_type.lower() == 'ec':
        event_type = 100
    else:
        raise Exception(f'Unknown data type: {data_type} requested!')
    print(f' Getting {data_type} data from CDEC for ',
          cdec_stations, ' from ', start_date, ' to ', end_date)
    return [cr.read_station_data(station_id, event_type, 'E', start_date, end_date) for station_id in cdec_stations]


def get_lon_lat(station_id):
    cr = cdec.Reader()
    dfmeta = cr.read_station_meta_info(station_id)[0]
    dfmeta = dfmeta.reset_index().set_index('Value')
    degree_sign = u'\N{DEGREE SIGN}'
    return float(dfmeta.loc['Longitude'][0].replace(degree_sign, '')), float(dfmeta.loc['Latitude'][0].replace(degree_sign, ''))


def get_stations(cdec_stations):
    dfs = pd.DataFrame.from_records({sid: get_lon_lat(sid) for sid in cdec_stations}).T
    dfs = dfs.reset_index()
    dfs.columns = ['Station ID', 'Longitude', 'Latitude']
    return dfs


def get_cdec_data(start_date, end_date, cdec_stations, data_type):
    raw_cdec_data = _get_raw_cdec_data(
        start_date, end_date, cdec_stations=cdec_stations, data_type=data_type)
    for i, d in enumerate(raw_cdec_data):
        if d.empty:
            warn(f'Empty data set!: {cdec_stations[i]}. Will fail. Remove it from the stations')
    return [pd.DataFrame(d.VALUE).rename(columns={'VALUE': d.STATION_ID.iloc[0]}) for d in raw_cdec_data]


def get_stations_cached(cdec_stations, recache=False):
    pfname = 'stations_'+'_'.join(cdec_stations)+'.pkl'
    ensure_dir(cache_dir)
    pfname = os.path.join(cache_dir, pfname)
    if recache:
        os.path.remove(pfname)
    if os.path.exists(pfname):
        df = pd.read_pickle(pfname)
    else:
        df = get_stations(cdec_stations)
        df.to_pickle(pfname)
    return df


def get_ec_cdec_data(bsdate, bedate, cdec_stations, resample_interval='15T', cache=True):
    return get_cdec_data_cached(bsdate, bedate, cdec_stations, 'EC', resample_interval=resample_interval, recache=(not cache))


def get_flow_cdec_data(bsdate, bedate, cdec_stations, resample_interval='15T', cache=True):
    return get_cdec_data_cached(bsdate, bedate, cdec_stations, 'FLOW', resample_interval=resample_interval, recache=(not cache))

# cache into pickled file


def get_cdec_data_cached(sdate, edate, station_ids, data_type, resample_interval='15T', recache=False):
    ssdate = pd.to_datetime(sdate).strftime('%Y-%m-%d')
    sedate = pd.to_datetime(edate).strftime('%Y-%m-%d')
    pfname = 'dataset_'+'_'.join(station_ids)+'_'.join([ssdate, sedate, data_type])+'.pkl'
    ensure_dir(cache_dir)
    pfname = os.path.join(cache_dir, pfname)
    if recache:
        os.path.remove(pfname)
    if os.path.exists(pfname):
        df = pd.read_pickle(pfname)
    else:
        dflist = get_cdec_data(sdate, edate, station_ids, data_type)
        df = pd.concat([dfi.resample(resample_interval).mean() for dfi in dflist],axis=1)
        df.to_pickle(pfname)
    return df


def clear_cache():
    ensure_dir(cache_dir)
    os.path.remove(cache_dir)
