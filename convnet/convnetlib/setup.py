from distutils.core import setup, Extension
import os
import shutil
import numpy as np
import platform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def local_file(filename):
    return os.path.join(BASE_DIR, filename)

print '-----------------'
print 'Platform: ' + platform.system()
print '-----------------'

c_files = [
    'util.c',
    'conv.c',
    'pool.c',
    'convnetlib.c'
]

# Make VS treat VS .c files as .cpp
# (/TP flag being ignored)
if platform.system() == 'Windows':
    dir = './tmp'
    os.mkdir(dir)
    cpp_files = map(lambda f: 'tmp/' + f + 'pp', c_files)
    
    for i in range(len(c_files)):
        shutil.copy(c_files[i], cpp_files[i])    
    
    c_files = cpp_files


# define the extension module
module = Extension('convnetlib',
                   sources=[local_file(f) for f in c_files],
                   include_dirs=[np.get_include(), local_file('.')]
                   )

# run the setup
setup(name='convnetlib',
      version='0.1',
      ext_modules=[module]
      )

if platform.system() == 'Windows':
    shutil.rmtree(dir)
