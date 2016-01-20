from distutils.core import setup, Extension
import os
import numpy as np
import platform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def local_file(filename):
    return os.path.join(BASE_DIR, filename)

if platform.system() == 'Windows':
    extra_compile_args = ['/Tp']
else:
    extra_compile_args = []

# define the extension module
module = Extension('convnetlib',
                   sources=[
                       local_file('util.cpp'),
                       local_file('conv.cpp'),
                       local_file('pool.cpp'),
                       local_file('convnetlib.cpp'),
                   ],
                   include_dirs=[np.get_include(), local_file('.')],
                   extra_compile_args=extra_compile_args,
                   )

# run the setup
setup(name='convnetlib',
      version='0.1',
      ext_modules=[module]
      )
