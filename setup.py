#######################################################################
#                                                                     #
#  Copyright 2015 Max Planck Institute                                #
#                 for Dynamics and Self-Organization                  #
#                                                                     #
#  This file is part of bfps.                                         #
#                                                                     #
#  bfps is free software: you can redistribute it and/or modify       #
#  it under the terms of the GNU General Public License as published  #
#  by the Free Software Foundation, either version 3 of the License,  #
#  or (at your option) any later version.                             #
#                                                                     #
#  bfps is distributed in the hope that it will be useful,            #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of     #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      #
#  GNU General Public License for more details.                       #
#                                                                     #
#  You should have received a copy of the GNU General Public License  #
#  along with bfps.  If not, see <http://www.gnu.org/licenses/>       #
#                                                                     #
# Contact: Cristian.Lalescu@ds.mpg.de                                 #
#                                                                     #
#######################################################################



from machine_settings import include_dirs, library_dirs, extra_compile_args, extra_libraries
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

VERSION = date_name

src_file_list = ['field_descriptor',
                 'interpolator',
                 'particles',
                 'fftw_tools',
                 'fluid_solver_base',
                 'fluid_solver',
                 'spline_n1',
                 'spline_n2',
                 'spline_n3',
                 'spline_n4',
                 'spline_n5',
                 'spline_n6',
                 'Lagrange_polys']

header_list = ['cpp/base.hpp'] + ['cpp/' + fname + '.hpp' for fname in src_file_list]

# not sure we need the MANIFEST.in file, but I might as well
#with open('MANIFEST.in', 'w') as manifest_in_file:
#    for fname in ['bfps/cpp/' + fname + '.cpp' for fname in src_file_list] + header_list:
#        manifest_in_file.write('include {0}\n'.format(fname))

libraries = ['fftw3_mpi',
             'fftw3',
             'fftw3f_mpi',
             'fftw3f']

libraries += extra_libraries

pickle.dump(
        {'include_dirs' : include_dirs,
         'library_dirs' : library_dirs,
         'extra_compile_args' : extra_compile_args,
         'libraries' : libraries,
         'install_date' : now,
         'git_revision' : git_revision},
        open('bfps/install_info.pickle', 'wb'),
        protocol = 2)

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
        install_requires = ['numpy>=1.8', 'h5py>=2.2.1'],
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
        license = 'GPL version 3.0')

