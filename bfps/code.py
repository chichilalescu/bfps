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
import pickle

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
        self.host_info = {'type'        : 'cluster',
                          'environment' : None,
                          'deltanprocs' : 1,
                          'queue'       : '',
                          'mail_address': '',
                          'mail_events' : None}
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
        if not os.path.isfile(os.path.join(bfps.header_dir, 'base.hpp')):
            raise IOError('header not there:\n' +
                          '{0}\n'.format(os.path.join(bfps.header_dir, 'base.hpp')) +
                          '{0}\n'.format(bfps.dist_loc))
        libraries = ['bfps'] + bfps.install_info['libraries']

        command_strings = ['g++']
        command_strings += [self.name + '.cpp', '-o', self.name]
        command_strings += ['-O2'] + bfps.install_info['extra_compile_args']
        command_strings += ['-I' + idir for idir in bfps.install_info['include_dirs']]
        command_strings.append('-I' + bfps.header_dir)
        command_strings += ['-L' + ldir for ldir in bfps.install_info['library_dirs']]
        command_strings.append('-L' + bfps.lib_dir)
        for libname in libraries:
            command_strings += ['-l' + libname]
        print('compiling code with command\n' + ' '.join(command_strings))
        return subprocess.call(command_strings)
    def set_host_info(
            self,
            host_info = {}):
        self.host_info.update(host_info)
        return None
    def run(self,
            ncpu = 2,
            simname = 'test',
            iter0 = 0,
            out_file = 'out_file',
            err_file = 'err_file',
            hours = 1,
            minutes = 0,
            njobs = 1):
        assert(self.compile_code() == 0)
        current_dir = os.getcwd()
        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
        if self.work_dir != './':
            shutil.copy(self.name, self.work_dir)
        os.chdir(self.work_dir)
        with open(self.name + '_version_info.txt', 'w') as outfile:
            outfile.write(self.version_message)
        os.chdir(current_dir)
        command_atoms = ['mpirun',
                         '-np',
                         '{0}'.format(ncpu),
                         './' + self.name,
                         simname,
                         '{0}'.format(iter0)]
        if self.host_info['type'] == 'cluster':
            job_name_list = []
            for j in range(njobs):
                suffix = simname + '_{0}.sh'.format(iter0 + j*self.parameters['niter_todo'])
                qsub_script_name = 'run_' + suffix
                self.write_sge_file(
                    file_name     = os.path.join(self.work_dir, qsub_script_name),
                    nprocesses    = ncpu,
                    name_of_run   = self.name + '_' + suffix,
                    command_atoms = command_atoms[3:],
                    hours         = hours,
                    minutes       = minutes,
                    out_file      = out_file,
                    err_file      = err_file)
                os.chdir(self.work_dir)
                qsub_atoms = ['qsub']
                if len(job_name_list) >= 1:
                    qsub_atoms += ['-hold_jid', job_name_list[-1]]
                subprocess.call(qsub_atoms + [qsub_script_name])
                os.chdir(current_dir)
                job_name_list.append(self.name + '_' + suffix)
        elif self.host_info['type'] == 'pc':
            os.chdir(self.work_dir)
            os.environ['LD_LIBRARY_PATH'] += ':{0}'.format(bfps.lib_dir)
            subprocess.call(command_atoms,
                            stdout = open(out_file + '_' + simname, 'w'),
                            stderr = open(err_file + '_' + simname, 'w'))
            os.chdir(current_dir)
        return None
    def write_sge_file(
            self,
            file_name = None,
            nprocesses = None,
            name_of_run = None,
            command_atoms = [],
            hours = None,
            minutes = None,
            out_file = None,
            err_file = None):
        script_file = open(file_name, 'w')
        script_file.write('#!/bin/bash\n')
        # export all environment variables
        script_file.write('#$ -V\n')
        # job name
        script_file.write('#$ -N {0}\n'.format(name_of_run))
        # use current working directory
        script_file.write('#$ -cwd\n')
        # error file
        if not type(err_file) == type(None):
            script_file.write('#$ -e ' + err_file + '\n')
        # output file
        if not type(out_file) == type(None):
            script_file.write('#$ -o ' + out_file + '\n')
        if not type(self.host_info['environment']) == type(None):
            envprocs = (nprocesses / self.host_info['deltanprocs'] + 1) * self.host_info['deltanprocs']
            script_file.write('#$ -pe {0} {1}\n'.format(
                    self.host_info['environment'],
                    envprocs))
        script_file.write('echo "got $NSLOTS slots."\n')
        script_file.write('echo "Start time is `date`"\n')
        script_file.write('mpiexec -machinefile $TMPDIR/machines ' +
                          '-genv LD_LIBRARY_PATH ' +
                          '"' +
                          ':'.join([bfps.lib_dir] + bfps.install_info['library_dirs']) +
                          '" ' +
                          '-n {0} {1}\n'.format(nprocesses, ' '.join(command_atoms)))
        script_file.write('echo "End time is `date`"\n')
        script_file.write('exit 0\n')
        script_file.close()
        return None

