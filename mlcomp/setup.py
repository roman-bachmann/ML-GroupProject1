#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='mlcomp',
      url='',
      author='',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version='0.0.1',
      install_requires=[
          'numpy==1.11.0',
          'scipy==0.19.1',
          'pandas==0.20.3',
      ],
      include_package_data=True,
      zip_safe=False)
