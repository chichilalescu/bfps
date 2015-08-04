########################################################################
#
#  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: Cristian.Lalescu@ds.mpg.de
#
########################################################################



from machine_settings import include_dirs, library_dirs, extra_compile_args
import pickle


AUTHOR = 'Cristian C Lalescu'
AUTHOR_EMAIL = 'Cristian.Lalescu@ds.mpg.de'

import datetime
import subprocess
from subprocess import CalledProcessError

now = datetime.datetime.now()
date_name = '{0:0>4}{1:0>2}{2:0>2}'.format(now.year, now.month, now.day)

try:
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
except:
    git_revision = ''
pickle.dump(
        {'include_dirs' : include_dirs,
         'library_dirs' : library_dirs,
         'extra_compile_args' : extra_compile_args,
         'install_date' : now,
         'git_revision' : git_revision},
        open('bfps/install_info.pickle', 'wb'),
        protocol = 2)

VERSION = date_name

src_file_list = ['field_descriptor',
                 'fftw_tools',
                 'vector_field',
                 'fluid_solver_base',
                 'fluid_solver',
                 'slab_field_particles',
                 'tracers',
                 'spline_n1',
                 'spline_n2',
                 'spline_n3']

header_list = ['cpp/base.hpp'] + ['cpp/' + fname + '.hpp' for fname in src_file_list]

# not sure we need the MANIFEST.in file, but I might as well
#with open('MANIFEST.in', 'w') as manifest_in_file:
#    for fname in ['bfps/cpp/' + fname + '.cpp' for fname in src_file_list] + header_list:
#        manifest_in_file.write('include {0}\n'.format(fname))

libraries = ['fftw3_mpi',
             'fftw3',
             'fftw3f_mpi',
             'fftw3f']

from setuptools import setup, Extension

libbfps = Extension(
        'libbfps',
        sources = ['bfps/cpp/' + fname + '.cpp' for fname in src_file_list],
        include_dirs = include_dirs,
        libraries = libraries,
        extra_compile_args = extra_compile_args,
        library_dirs = library_dirs)

setup(
        name = 'bfps',
        packages = ['bfps'],
        install_requires = ['numpy>=1.8', 'matplotlib>=1.3'],
        ext_modules = [libbfps],
        package_data = {'bfps': header_list + ['../machine_settings.py',
                                               'install_info.pickle']},
########################################################################
# useless stuff folows
########################################################################
        description = 'Big Fluid and Particle Simulator',
        long_description = open('README.rst', 'r').read(),
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        version = VERSION,
        license = 'Apache Version 2.0')

