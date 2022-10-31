import sys
import click
import transect_viz
from . import __version__
from . import transect_data


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
    click.echo(f'Reading from geojson file: {geojson_line_file} and convert to equidistant points along that line and writing to {geojson_pts_file}')
    transect_data.convert_geojson_line_to_pts(geojson_line_file, geojson_pts_file, delx=delx)

main.add_command(version)
main.add_command(line2pts)

if __name__ == '__main__':
    sys.exit(main())
