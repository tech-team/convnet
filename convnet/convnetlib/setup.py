from distutils.core import setup, Extension
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def local_file(filename):
    return os.path.join(BASE_DIR, filename)

# define the extension module
module = Extension('convnetlib',
                   sources=[local_file('convnetlib.c')],
                   include_dirs=[np.get_include(), local_file('.')])

# run the setup
setup(name='convnetlib',
      version='0.1',
      ext_modules=[module]
      )
