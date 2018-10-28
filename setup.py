#!/usr/bin/env python

import os
from setuptools import setup

install_requires = [
    'numpy>=0.14.0',
    'lightgbm>=2.2.2',
    'dask>=0.16.0',
    'distributed >= 1.15.2',
    'sparse>=0.5.0'
]

extras_require = {
    "dev": [
        "pytest>=3.9.0",
        "pandas>=0.23.0"
    ],
    "sparse": [
        "sparse>=0.5.0",
        "scipy>=1.0.0"
    ]
}

setup(name='dask-lightgbm',
      version='0.1.0',
      description='Interactions between Dask and LightGBM',
      license='BSD',
      install_requires=install_requires,
      extras_require=extras_require,
      packages=['dask_lightgbm'],
      long_description=(open('README.rst').read()
                        if os.path.exists('README.rst')
                        else ''),
      zip_safe=False)
