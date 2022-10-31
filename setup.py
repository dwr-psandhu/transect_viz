from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
    'Click',
    'hvplot',
    'panel',
    'geoviews',
    'geopandas',
    'pandas',
    'xarray',
    'vtools3'
]

setup_requirements = ['pytest-runner>=5.0', ]

test_requirements = ['pytest>=5.0', ]

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
            'transect_viz=transect_viz.cli:main'
        ]
    },
    install_requires=requirements,
    include_package_data=True,
    keywords='transect_viz',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)
