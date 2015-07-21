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


import bfps
from bfps.base import base
import subprocess
import os
import shutil


class code(base):
    def __init__(self):
        super(code, self).__init__()
        self.version_message = ('/***********************************************************************\n' +
                                '* this code automatically generated by bfps\n' +
                                '* version {0}\n'.format(bfps.__version__) +
                                '***********************************************************************/\n\n\n')
        self.includes = """
                //begincpp
                #include "base.hpp"
                #include "fluid_solver.hpp"
                #include <iostream>
                #include <fftw3-mpi.h>
                //endcpp
                """
        self.variables = 'int myrank, nprocs;\n'
        self.variables += 'int iter0;\n'
        self.variables += 'char simname[256];\n'
        self.definitions = ''
        self.main_start = """
                //begincpp
                int main(int argc, char *argv[])
                {
                    MPI_Init(&argc, &argv);
                    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
                    if (argc != 3)
                    {
                        std::cerr << "Wrong number of command line arguments. Stopping." << std::endl;
                        MPI_Finalize();
                        return EXIT_SUCCESS;
                    }
                    else
                    {
                        strcpy(simname, argv[1]);
                        iter0 = atoi(argv[2]);
                        std::cerr << "myrank = " << myrank <<
                                     ", simulation name is " << simname <<
                                     " and iter0 is " << iter0 << std::endl;
                    }
                    read_parameters();
                //endcpp
                """
        self.main_start += 'if (myrank == 0) std::cout << "{0}" << std::endl;'.format(self.version_message).replace('\n', '\\n') + '\n'
        self.main_start += 'if (myrank == 0) std::cerr << "{0}" << std::endl;'.format(self.version_message).replace('\n', '\\n') + '\n'
        self.main_end = """
                //begincpp
                    // clean up
                    fftwf_mpi_cleanup();
                    fftw_mpi_cleanup();
                    MPI_Finalize();
                    return EXIT_SUCCESS;
                }
                //endcpp
                """
        return None
    def write_src(self):
        with open(self.name + '.cpp', 'w') as outfile:
            outfile.write(self.version_message)
            outfile.write(self.includes)
            outfile.write(self.variables)
            outfile.write(self.definitions)
            outfile.write(self.main_start)
            outfile.write(self.main)
            outfile.write(self.main_end)
        return None
    def compile_code(self):
        # compile code
        local_install_dir = '/scratch.local/chichi/installs'
        include_dirs = [bfps.header_dir,
                        '/usr/lib64/mpi/gcc/openmpi/include',
                        os.path.join(local_install_dir, 'include')]
        if not os.path.isfile(os.path.join(bfps.header_dir, 'base.hpp')):
            raise IOError('header not there:\n' +
                          '{0}\n'.format(os.path.join(bfps.header_dir, 'base.hpp')) +
                          '{0}\n'.format(bfps.dist_loc))
        libraries = ['fftw3_mpi',
                     'fftw3',
                     'fftw3f_mpi',
                     'fftw3f',
                     'bfps']

        command_strings = ['mpicxx']
        for idir in include_dirs:
            command_strings += ['-I{0}'.format(idir)]
        command_strings += ['-L' + os.path.join(local_install_dir, 'lib')]
        command_strings += ['-L' + os.path.join(local_install_dir, 'lib64')]
        command_strings.append('-L' + bfps.lib_dir)
        for libname in libraries:
            command_strings += ['-l' + libname]
        command_strings += [self.name + '.cpp', '-o', self.name]
        return subprocess.call(command_strings)
    def run(self,
            ncpu = 2,
            simname = 'test',
            iter0 = 0):
        if self.compile_code() == 0:
            current_dir = os.getcwd()
            if not os.path.isdir(self.work_dir):
                os.makedirs(self.work_dir)
            if self.work_dir != './':
                shutil.copy(self.name, self.work_dir)
            os.chdir(self.work_dir)
            with open(self.name + '_version_info.txt', 'w') as outfile:
                outfile.write(self.version_message)
            subprocess.call(['time',
                             'mpirun',
                             '-np',
                             '{0}'.format(ncpu),
                             './' + self.name,
                             simname,
                             '{0}'.format(iter0)])
            os.chdir(current_dir)
        return None

