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



import os

########################################################################
# these lists should be adapted for your different environment(s)
# personally, I have access to setups where my home folder is shared
# between different machines, including cluster and desktop, therefore
# I check the host name when choosing libraries etc.
# feel free to do your own thing to the copy of this file placed in
# ./config/bfps
########################################################################

hostname = os.getenv('HOSTNAME')

compiler = 'g++'
extra_compile_args = ['-Wall', '-O2', '-g', '-mtune=native', '-ffast-math', '-std=c++11']
extra_libraries = ['hdf5']
include_dirs = []
library_dirs = []

if hostname == 'chichi-G':
    include_dirs = ['/usr/local/include',
                    '/usr/include/mpich']
    library_dirs = ['/usr/local/lib',
                    '/usr/lib/mpich']
    extra_libraries += ['mpich']

if hostname in ['tolima', 'misti']:
    local_install_dir = '/scratch.local/chichi/installs'

    include_dirs = ['/usr/lib64/mpi/gcc/openmpi/include',
                    os.path.join(local_install_dir, 'include')]

    library_dirs = ['/usr/lib64/mpi/gcc/openmpi/lib64',
                    os.path.join(local_install_dir, 'lib'),
                    os.path.join(local_install_dir, 'lib64')]
    extra_libraries += ['mpi_cxx', 'mpi']

