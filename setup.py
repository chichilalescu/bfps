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



AUTHOR = 'Cristian C Lalescu'
AUTHOR_EMAIL = 'Cristian.Lalescu@ds.mpg.de'

import os
import shutil
import datetime
import sys
import subprocess
import pickle



### compiler configuration
# check if .config/bfps/machine_settings.py file exists, create it if not
homefolder = os.path.expanduser('~')
bfpsfolder = os.path.join(homefolder, '.config/', 'bfps')
if not os.path.exists(os.path.join(bfpsfolder, 'machine_settings.py')):
    if not os.path.isdir(bfpsfolder):
        os.mkdir(bfpsfolder)
    shutil.copyfile('./machine_settings_py.py', os.path.join(bfpsfolder, 'machine_settings.py'))
# check if .config/bfps/host_information.py file exists, create it if not
if not os.path.exists(os.path.join(bfpsfolder, 'host_information.py')):
    if not os.path.isdir(bfpsfolder):
        os.mkdir(bfpsfolder)
    open(os.path.join(bfpsfolder, 'host_information.py'),
         'w').write('host_info = {\'type\' : \'none\'}\n')
    shutil.copyfile('./machine_settings_py.py', os.path.join(bfpsfolder, 'machine_settings.py'))
sys.path.append(bfpsfolder)
# import stuff required for compilation of static library
from machine_settings import include_dirs, library_dirs, extra_compile_args, extra_libraries



### package versioning
# get current time
now = datetime.datetime.now()
# obtain version
try:
    git_branch = subprocess.check_output(['git',
                                          'rev-parse',
                                          '--abbrev-ref',
                                          'HEAD']).strip().split()[-1].decode()
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    git_date = datetime.datetime.fromtimestamp(int(subprocess.check_output(['git', 'log', '-1', '--format=%ct']).strip()))
except:
    git_revision = ''
    git_branch = ''
    git_date = now
if git_branch == '':
    # there's no git available or something
    VERSION = '{0:0>4}{1:0>2}{2:0>2}.{3:0>2}{4:0>2}{5:0>2}'.format(
                git_date.year, git_date.month, git_date.day,
                git_date.hour, git_date.minute, git_date.second)
else:
    if (('develop' in git_branch) or
        ('feature' in git_branch) or
        ('bugfix'  in git_branch)):
        VERSION = subprocess.check_output(
                ['git', 'describe', '--tags', '--dirty']).strip().decode().replace('-g', '+g').replace('-dirty', '.dirty').replace('-', '.post')
    else:
        VERSION = subprocess.check_output(['git', 'describe', '--tags']).strip().decode().split('-')[0]
print('This is bfps version ' + VERSION)



### lists of files and MANIFEST.in
src_file_list = ['field_descriptor',
                 'interpolator_base',
                 'distributed_particles',
                 'particles_base',
                 'interpolator',
                 'particles',
                 'rFFTW_interpolator',
                 'rFFTW_particles',
                 'fluid_solver_base',
                 'fluid_solver',
                 'fftw_tools',
                 'spline_n1',
                 'spline_n2',
                 'spline_n3',
                 'spline_n4',
                 'spline_n5',
                 'spline_n6',
                 'Lagrange_polys']

header_list = (['cpp/base.hpp'] +
               ['cpp/' + fname + '.hpp'
                for fname in src_file_list])

with open('MANIFEST.in', 'w') as manifest_in_file:
    for fname in ['bfps/cpp/' + fname + '.cpp' for fname in src_file_list] + header_list:
        manifest_in_file.write('include {0}\n'.format(fname))



### libraries
libraries = ['fftw3_mpi',
             'fftw3',
             'fftw3f_mpi',
             'fftw3f']
libraries += extra_libraries



### save compiling information
pickle.dump(
        {'include_dirs' : include_dirs,
         'library_dirs' : library_dirs,
         'extra_compile_args' : extra_compile_args,
         'libraries' : libraries,
         'install_date' : now,
         'VERSION' : VERSION,
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
            assert(subprocess.call(command_strings) == 0)
    command_strings = ['ar', 'rvs', 'bfps/libbfps.a']
    command_strings += ['obj/' + fname + '.o' for fname in src_file_list]
    print(' '.join(command_strings))
    assert(subprocess.call(command_strings) == 0)
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
        package_data = {'bfps': header_list + ['libbfps.a',
                                               'install_info.pickle']},
        entry_points = {
            'console_scripts': [
                'bfps = bfps.__main__:main'],
            },
        version = VERSION,
########################################################################
# useless stuff folows
########################################################################
        description = 'Big Fluid and Particle Simulator',
        long_description = open('README.rst', 'r').read(),
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = 'GPL version 3.0')

