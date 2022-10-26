from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
    'hvplot',
    'panel',
    'geoviews',
    'geopandas',
    'pandas',
    'xarray',
    'vtools3'
]

setup(
    name='transect_viz',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Transect Visualization and Generation",
    license="MIT",
    author="Nicky Sandhu",
    author_email='psandhu@water.ca.gov',
    url='https://github.com/dwr-psandhu/transect_viz',
    packages=['transect_viz'],
    entry_points={
        'console_scripts': [
            'transect_viz=transect_viz.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='transect_viz',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
