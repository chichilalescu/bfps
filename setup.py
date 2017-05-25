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
bfpsfolder = os.path.join(homefolder, '.config', 'bfps')
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
sys.path.insert(0, bfpsfolder)
# import stuff required for compilation of static library
from machine_settings import compiler, include_dirs, library_dirs, extra_compile_args, extra_libraries


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
src_file_list = ['hdf5_tools',
                 'full_code/get_rfields',
                 'full_code/NSVE_field_stats',
                 'full_code/native_binary_to_hdf5',
                 'full_code/postprocess',
                 'full_code/code_base',
                 'full_code/direct_numerical_simulation',
                 'full_code/NSVE',
                 'full_code/NSVEparticles',
                 'field_binary_IO',
                 'vorticity_equation',
                 'field',
                 'kspace',
                 'field_layout',
                 'field_descriptor',
                 'rFFTW_distributed_particles',
                 'distributed_particles',
                 'particles',
                 'particles_base',
                 'rFFTW_interpolator',
                 'interpolator',
                 'interpolator_base',
                 'fluid_solver',
                 'fluid_solver_base',
                 'fftw_tools',
                 'spline_n1',
                 'spline_n2',
                 'spline_n3',
                 'spline_n4',
                 'spline_n5',
                 'spline_n6',
                 'spline_n7',
                 'spline_n8',
                 'spline_n9',
                 'spline_n10',
                 'Lagrange_polys',
                 'scope_timer']

particle_headers = [
        'cpp/particles/particles_distr_mpi.hpp',
        'cpp/particles/abstract_particles_input.hpp',
        'cpp/particles/abstract_particles_output.hpp',
        'cpp/particles/abstract_particles_system.hpp',
        'cpp/particles/alltoall_exchanger.hpp',
        'cpp/particles/particles_adams_bashforth.hpp',
        'cpp/particles/particles_field_computer.hpp',
        'cpp/particles/particles_input_hdf5.hpp',
        'cpp/particles/particles_generic_interp.hpp',
        'cpp/particles/particles_output_hdf5.hpp',
        'cpp/particles/particles_output_mpiio.hpp',
        'cpp/particles/particles_system_builder.hpp',
        'cpp/particles/particles_system.hpp',
        'cpp/particles/particles_utils.hpp',
        'cpp/particles/particles_output_sampling_hdf5.hpp',
        'cpp/particles/particles_sampling.hpp',
        'cpp/particles/env_utils.hpp']

full_code_headers = ['cpp/full_code/main_code.hpp',
                     'cpp/full_code/codes_with_no_output.hpp',
                     'cpp/full_code/NSVE_no_output.hpp',
                     'cpp/full_code/NSVEparticles_no_output.hpp']

header_list = (['cpp/base.hpp'] +
               ['cpp/fftw_interface.hpp'] +
               ['cpp/bfps_timer.hpp'] +
               ['cpp/omputils.hpp'] +
               ['cpp/shared_array.hpp'] +
               ['cpp/spline.hpp'] +
               ['cpp/' + fname + '.hpp'
                for fname in src_file_list] +
               particle_headers +
               full_code_headers)

with open('MANIFEST.in', 'w') as manifest_in_file:
    for fname in (['bfps/cpp/' + ff + '.cpp' for ff in src_file_list] +
                  ['bfps/' + ff for ff in header_list]):
        manifest_in_file.write('include {0}\n'.format(fname))



### libraries
libraries = extra_libraries


import distutils.cmd

class CompileLibCommand(distutils.cmd.Command):
    description = 'Compile bfps library.'
    user_options = [
            ('timing-output=', None, 'Toggle timing output.'),
            ('fftw-estimate=', None, 'Use FFTW ESTIMATE.'),
            ('disable-fftw-omp=', None, 'Turn Off FFTW OpenMP.'),
            ]
    def initialize_options(self):
        self.timing_output = 0
        self.fftw_estimate = 0
        self.disable_fftw_omp = 0
        return None
    def finalize_options(self):
        self.timing_output = (int(self.timing_output) == 1)
        self.fftw_estimate = (int(self.fftw_estimate) == 1)
        self.disable_fftw_omp = (int(self.disable_fftw_omp) == 1)
        return None
    def run(self):
        if not os.path.isdir('obj'):
            os.makedirs('obj')
            need_to_compile = True
        if not os.path.isdir('obj/full_code'):
            os.makedirs('obj/full_code')
            need_to_compile = True
        if not os.path.isfile('bfps/libbfps.a'):
            need_to_compile = True
        else:
            ofile = 'bfps/libbfps.a'
            libtime = datetime.datetime.fromtimestamp(os.path.getctime(ofile))
            latest = libtime
            for fname in header_list:
                latest = max(latest,
                             datetime.datetime.fromtimestamp(os.path.getctime('bfps/' + fname)))
            need_to_compile = (latest > libtime)
        eca = extra_compile_args
        eca += ['-fPIC']
        if self.timing_output:
            eca += ['-DUSE_TIMINGOUTPUT']
        if self.fftw_estimate:
            eca += ['-DUSE_FFTWESTIMATE']
        if self.disable_fftw_omp:
            eca += ['-DNO_FFTWOMP']
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
                command_strings = [compiler, '-c']
                command_strings += ['bfps/cpp/' + fname + '.cpp']
                command_strings += ['-o', 'obj/' + fname + '.o']
                command_strings += eca
                command_strings += ['-I' + idir for idir in include_dirs]
                command_strings.append('-Ibfps/cpp/')
                print(' '.join(command_strings))
                subprocess.check_call(command_strings)
        command_strings = ['ar', 'rvs', 'bfps/libbfps.a']
        command_strings += ['obj/' + fname + '.o' for fname in src_file_list]
        print(' '.join(command_strings))
        subprocess.check_call(command_strings)

        ### save compiling information
        pickle.dump(
                {'include_dirs' : include_dirs,
                 'library_dirs' : library_dirs,
                 'compiler'     : compiler,
                 'extra_compile_args' : eca,
                 'libraries' : libraries,
                 'install_date' : now,
                 'VERSION' : VERSION,
                 'git_revision' : git_revision},
                open('bfps/install_info.pickle', 'wb'),
                protocol = 2)
        return None

from setuptools import setup

setup(
        name = 'bfps',
        packages = ['bfps', 'bfps/test'],
        install_requires = ['numpy>=1.8', 'h5py>=2.2.1'],
        cmdclass={'compile_library' : CompileLibCommand},
        package_data = {'bfps': header_list +
                                ['libbfps.a',
                                 'install_info.pickle'] +
                                ['test/B32p1e4_checkpoint_0.h5']},
        entry_points = {
            'console_scripts': [
                'bfps = bfps.__main__:main',
                'bfps1 = bfps.__main__:main',
                'bfps.test_NSVEparticles = bfps.test.test_bfps_NSVEparticles:main'],
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

