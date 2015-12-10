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

import os
import datetime
import subprocess
from subprocess import CalledProcessError

now = datetime.datetime.now()

try:
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    git_date = datetime.datetime.fromtimestamp(int(subprocess.check_output(['git', 'log', '-1', '--format=%ct']).strip()))
except:
    git_revision = ''
    git_date = now

VERSION = '{0:0>4}{1:0>2}{2:0>2}.{3:0>2}{4:0>2}{5:0>2}'.format(
            git_date.year, git_date.month, git_date.day,
            git_date.hour, git_date.minute, git_date.second)

src_file_list = ['field_descriptor',
                 'fluid_solver_base',
                 'fluid_solver',
                 'interpolator',
                 'rFFTW_interpolator',
                 'particles',
                 'rFFTW_particles',
                 'fftw_tools',
                 'spline_n1',
                 'spline_n2',
                 'spline_n3',
                 'spline_n4',
                 'spline_n5',
                 'spline_n6',
                 'Lagrange_polys']

header_list = ['cpp/base.hpp'] + ['cpp/' + fname + '.hpp' for fname in src_file_list]

#with open('MANIFEST.in', 'w') as manifest_in_file:
#    manifest_in_file.write('include libbfps.a\n')
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

def compile_bfps_library():
    if not os.path.isdir('obj'):
        os.makedirs('obj')
        need_to_compile = True
    else:
        ofile = 'bfps/libbfps.a'
        libtime = datetime.datetime.fromtimestamp(os.path.getctime(ofile))
        latest = libtime
        for fname in header_list:
            latest = max(latest,
                         datetime.datetime.fromtimestamp(os.path.getctime('bfps/' + fname)))
        need_to_compile = (latest > libtime)
    for fname in src_file_list:
        ifile = 'bfps/cpp/' + fname + '.cpp'
        ofile = 'obj/' + fname + '.o'
        if not os.path.exists(ofile):
            need_to_compile_file = True
        else:
            need_to_compile_file = (need_to_compile or
                                    (datetime.datetime.fromtimestamp(os.path.getctime(ofile)) <
                                     datetime.datetime.fromtimestamp(os.path.getctime(ifile))))
        if need_to_compile_file:
            command_strings = ['g++', '-c']
            command_strings += ['bfps/cpp/' + fname + '.cpp']
            command_strings += ['-o', 'obj/' + fname + '.o']
            command_strings += extra_compile_args
            command_strings += ['-I' + idir for idir in include_dirs]
            command_strings.append('-Ibfps/cpp/')
            print(' '.join(command_strings))
            subprocess.call(command_strings)
    command_strings = ['ar', 'rvs', 'bfps/libbfps.a']
    command_strings += ['obj/' + fname + '.o' for fname in src_file_list]
    print(' '.join(command_strings))
    subprocess.call(command_strings)
    return None

from distutils.command.build import build as DistutilsBuild
from distutils.command.install import install as DistutilsInstall

class CustomBuild(DistutilsBuild):
    def run(self):
        compile_bfps_library()
        DistutilsBuild.run(self)

# this custom install leads to a broken installation. no idea why...
class CustomInstall(DistutilsInstall):
    def run(self):
        compile_bfps_library()
        DistutilsInstall.run(self)

from setuptools import setup

setup(
        name = 'bfps',
        packages = ['bfps'],
        install_requires = ['numpy>=1.8', 'h5py>=2.2.1'],
        cmdclass={'build' : CustomBuild},
        package_data = {'bfps': header_list + ['../machine_settings.py',
                                               'libbfps.a',
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

