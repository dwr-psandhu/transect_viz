import sys
import click
import transect_viz
from . import __version__
from . import transect_data
from . import transect_cdec_data
import pandas as pd

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@click.command()
def version():
    '''prints version of this module'''
    click.echo(f'transect-viz version:{__version__}')


@click.command()
@click.argument('geojson_line_file', type=click.Path(exists=True))
@click.argument('geojson_pts_file', type=click.Path(exists=False))
@click.option("--delx", default=25, help="Equidistance between points generated along line")
def line2pts(geojson_line_file, geojson_pts_file, delx=25):
    '''
    Reads a geojson_line_file containing a line string to equidistant points along that line and saves to geojson_pts_file
    '''
    click.echo(
        f'Reading from geojson file: {geojson_line_file} and convert to equidistant points along that line and writing to {geojson_pts_file}')
    transect_data.convert_geojson_line_to_pts(geojson_line_file, geojson_pts_file, delx=delx)


@click.command()
@click.argument('transect_name', type=str)
@click.argument('transect_file', type=click.Path(exists=True))
@click.argument('transect_viz_outfile', type=click.Path(exists=False))
@click.option('--regenerate_csv/--no-regenerate_csv', default=True)
@click.option('--station_ids', '-s', multiple=True, default=['ORM', 'ODM', 'OLD', 'GLE', 'GLC'])
@click.option('--station_info_file', type=click.Path(exists=True), default='../data/station_info_file.csv')
@click.option('--date_column', type=str, default='Sonde1_SpCond')
def generate_transect_viz(transect_name, transect_file, transect_viz_outfile, transect_viz_title,
                                            regenerate_csv=True,
                                            stations_id=[
                                                'ORM', 'ODM', 'OLD', 'GLE', 'GLC'],
                                            station_info_file='../data/station_info_file.csv',
                                            data_column='Sonde1_SpCond'):
    '''
    Generates transect visualization in .html file from the transect file containing the data and downloaded CDEC data for stage/flow/velocity & ec'''
    # all variables for driving the animation
    dft = transect_data.load_transect_file(transect_file)
    dtmin, dtmax = dft.DateTime.agg(['min', 'max'])

    sdate = (pd.to_datetime(dtmin) - pd.Timedelta('2D')
             ).floor('D').strftime('%Y-%m-%d')

    edate = (pd.to_datetime(dtmin) + pd.Timedelta('2D')
             ).ceil('D').strftime('%Y-%m-%d')

    # -- if data is not updated or available in the csv files
    if (regenerate_csv):
        _, flow_data_file = transect_cdec_data.generate_csv_files(transect_name,
                                                                  stations_id,
                                                                  'flow', sdate, edate)
        _, stage_data_file = transect_cdec_data.generate_csv_files(transect_name,
                                                                   stations_id,
                                                                   'stage', sdate, edate)
        _, velocity_data_file = transect_cdec_data.generate_csv_files(transect_name,
                                                                      stations_id,
                                                                      'vel', sdate, edate)
        stations_csv_file, ec_data_file = transect_cdec_data.generate_csv_files(transect_name,
                                                                                stations_id,
                                                                                'ec', sdate, edate)

    print(f'Generating html file for visualization: {transect_viz_outfile}')
    transect_viz.generate_transect_vizualization(transect_file, transect_viz_outfile, transect_viz_title,
                                                 stations_csv_file, station_info_file,
                                                 flow_data_file, velocity_data_file, stage_data_file, ec_data_file, data_column=data_column)


main.add_command(version)
main.add_command(line2pts)
main.add_command(generate_transect_viz)

if __name__ == '__main__':
    sys.exit(main())
