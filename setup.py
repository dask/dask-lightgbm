#!/usr/bin/env python

from setuptools import setup

install_requires = [
    'numpy>=1.17.3',
    'lightgbm>=2.3.0',
    'dask>=2.6.0',
    'distributed>=2.6.0',
    'toolz>=0.10.0'
]

extras_require = {
    'dev': [
        'pytest>=5.2.2',
        'pandas>=0.25.3',
        'dask[dataframe]',
        'dask-ml>=1.1.1',
        'requests>=2.22.0',
        'fsspec>=0.5.2',
        'scikit-learn>=0.21.3'
    ],
    'sparse': [
        'sparse==0.5.0',
        'scipy>=1.3.1'
    ]
}


with open('README.md', mode='r', encoding='utf-8') as f:
    readme = f.read()


setup(name='dask-lightgbm',
      version='0.1.0',
      description='Interactions between Dask and LightGBM',
      long_description=readme,
      long_description_content_type='text/markdown',
      license='BSD',
      url='https://github.com/dask/dask-lightgbm',
      install_requires=install_requires,
      extras_require=extras_require,
      packages=['dask_lightgbm'],
      include_package_data=True,
      zip_safe=False)
