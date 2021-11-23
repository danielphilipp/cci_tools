from distutils.core import setup
from Cython.Build import cythonize

setup(name="parse_cal_profiles", include_dirs=['/cmsaf/nfshome/routcm/Modules_CentOS/python/3.7.2/lib/python3.7/site-packages/numpy/core/include'], ext_modules=cythonize('parse_cal_profiles.pyx'))
