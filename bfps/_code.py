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
import sys
import shutil
import subprocess
import argparse
import h5py
from datetime import datetime
import math

import bfps
from ._base import _base

class _code(_base):
    """This class is meant to stitch together the C++ code into a final source file,
    compile it, and handle all job launching.
    """
    def __init__(
            self,
            work_dir = './',
            simname = 'test'):
        _base.__init__(self, work_dir = work_dir, simname = simname)
        self.version_message = ('/***********************************************************************\n' +
                                '* this code automatically generated by bfps\n' +
                                '* version {0}\n'.format(bfps.__version__) +
                                '***********************************************************************/\n\n\n')
        self.includes = """
                //begincpp
                #include "base.hpp"
                #include "fluid_solver.hpp"
                #include <iostream>
                #include <hdf5.h>
                #include <string>
                #include <cstring>
                #include <fftw3-mpi.h>
                //endcpp
                """
        self.variables = 'int myrank, nprocs;\n'
        self.variables += 'int iteration;\n'
        self.variables += 'char simname[256], fname[256];\n'
        self.variables += ('hid_t parameter_file, stat_file, Cdset;\n')
        self.definitions = ''
        self.main_start = """
                //begincpp
                int main(int argc, char *argv[])
                {
                    MPI_Init(&argc, &argv);
                    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
                    fftw_mpi_init();
                    fftwf_mpi_init();
                    if (argc != 2)
                    {
                        std::cerr << "Wrong number of command line arguments. Stopping." << std::endl;
                        MPI_Finalize();
                        return EXIT_SUCCESS;
                    }
                    strcpy(simname, argv[1]);
                    sprintf(fname, "%s.h5", simname);
                    parameter_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
                    Cdset = H5Dopen(parameter_file, "iteration", H5P_DEFAULT);
                    H5Dread(Cdset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &iteration);
                    DEBUG_MSG("simname is %s and iteration is %d\\n", simname, iteration);
                    H5Dclose(Cdset);
                    H5Fclose(parameter_file);
                    read_parameters();
                    if (myrank == 0)
                    {
                        // set caching parameters
                        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
                        herr_t cache_err = H5Pset_cache(fapl, 0, 521, 134217728, 1.0);
                        DEBUG_MSG("when setting stat_file cache I got %d\\n", cache_err);
                        stat_file = H5Fopen(fname, H5F_ACC_RDWR, fapl);
                    }
                //endcpp
                """
        for ostream in ['cout', 'cerr']:
            self.main_start += 'if (myrank == 0) std::{1} << "{0}" << std::endl;'.format(self.version_message, ostream).replace('\n', '\\n') + '\n'
        self.main_end = """
                //begincpp
                    // clean up
                    if (myrank == 0)
                    {
                        Cdset = H5Dopen(stat_file, "iteration", H5P_DEFAULT);
                        H5Dwrite(Cdset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &iteration);
                        H5Dclose(Cdset);
                        H5Fclose(stat_file);
                    }
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
        self.main = ''
        return None
    def write_src(self):
        with open(self.name + '.cpp', 'w') as outfile:
            outfile.write(self.version_message)
            outfile.write(self.includes)
            outfile.write(self.cdef_pars())
            outfile.write(self.variables)
            outfile.write(self.cread_pars())
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
        libraries = ['bfps']
        libraries += bfps.install_info['libraries']

        command_strings = [bfps.install_info['compiler']]
        command_strings += [self.name + '.cpp', '-o', self.name]
        command_strings += bfps.install_info['extra_compile_args']
        command_strings += ['-I' + idir for idir in bfps.install_info['include_dirs']]
        command_strings.append('-I' + bfps.header_dir)
        command_strings += ['-L' + ldir for ldir in bfps.install_info['library_dirs']]
        command_strings.append('-L' + bfps.lib_dir)
        for libname in libraries:
            command_strings += ['-l' + libname]
        self.write_src()
        print('compiling code with command\n' + ' '.join(command_strings))
        return subprocess.call(command_strings)
    def set_host_info(
            self,
            host_info = {}):
        self.host_info.update(host_info)
        return None
    def run(self,
            ncpu = 2,
            out_file = 'out_file',
            err_file = 'err_file',
            hours = 0,
            minutes = 10,
            njobs = 1):
        self.read_parameters()
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r') as data_file:
            iter0 = data_file['iteration'].value
        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
        if not os.path.exists(os.path.join(self.work_dir, self.name)):
            need_to_compile = True
        else:
            need_to_compile = (datetime.fromtimestamp(os.path.getctime(os.path.join(self.work_dir, self.name))) <
                               bfps.install_info['install_date'])
        if need_to_compile:
            assert(self.compile_code() == 0)
            if self.work_dir != os.path.realpath(os.getcwd()):
                shutil.copy(self.name, self.work_dir)
        current_dir = os.getcwd()
        os.chdir(self.work_dir)
        os.chdir(current_dir)
        command_atoms = ['mpirun',
                         '-np',
                         '{0}'.format(ncpu),
                         './' + self.name,
                         self.simname]
        if self.host_info['type'] == 'cluster':
            job_name_list = []
            for j in range(njobs):
                suffix = self.simname + '_{0}'.format(iter0 + j*self.parameters['niter_todo'])
                qsub_script_name = 'run_' + suffix + '.sh'
                self.write_sge_file(
                    file_name     = os.path.join(self.work_dir, qsub_script_name),
                    nprocesses    = ncpu,
                    name_of_run   = suffix,
                    command_atoms = command_atoms[3:],
                    hours         = hours,
                    minutes       = minutes,
                    out_file      = out_file + '_' + suffix,
                    err_file      = err_file + '_' + suffix)
                os.chdir(self.work_dir)
                qsub_atoms = ['qsub']
                if len(job_name_list) >= 1:
                    qsub_atoms += ['-hold_jid', job_name_list[-1]]
                subprocess.call(qsub_atoms + [qsub_script_name])
                os.chdir(current_dir)
                job_name_list.append(suffix)
        elif self.host_info['type'] == 'IBMLoadLeveler':
            job_name_list = []
            for j in range(njobs):
                suffix = self.simname + '_{0}'.format(iter0 + j*self.parameters['niter_todo'])
                job_script_name = 'run_' + suffix + '.sh'
                self.write_IBMLoadLeveler_file(
                    file_name     = os.path.join(self.work_dir, job_script_name),
                    nprocesses    = ncpu,
                    name_of_run   = suffix,
                    command_atoms = command_atoms[3:],
                    hours         = hours,
                    minutes       = minutes,
                    out_file      = out_file + '_' + suffix,
                    err_file      = err_file + '_' + suffix)
                submit_atoms = ['llsubmit']
                #if len(job_name_list) >= 1:
                #    qsub_atoms += ['-hold_jid', job_name_list[-1]]
                subprocess.call(submit_atoms + [os.path.join(self.work_dir, job_script_name)])
                job_name_list.append(suffix)
        elif self.host_info['type'] == 'pc':
            os.chdir(self.work_dir)
            os.environ['LD_LIBRARY_PATH'] += ':{0}'.format(bfps.lib_dir)
            print('added to LD_LIBRARY_PATH the location {0}'.format(bfps.lib_dir))
            for j in range(njobs):
                suffix = self.simname + '_{0}'.format(iter0 + j*self.parameters['niter_todo'])
                print('running code with command\n' + ' '.join(command_atoms))
                subprocess.call(command_atoms,
                                stdout = open(out_file + '_' + suffix, 'w'),
                                stderr = open(err_file + '_' + suffix, 'w'))
            os.chdir(current_dir)
        return None
    def write_IBMLoadLeveler_file(
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
        script_file.write('# @ shell=/bin/bash\n')
        # export all environment variables
        #script_file.write('#$ -V\n')
        # job name
        #script_file.write('#$ -N {0}\n'.format(name_of_run))
        # use current working directory
        #script_file.write('#$ -cwd\n')
        # error file
        if not type(err_file) == type(None):
            script_file.write('# @ error = ' + err_file + '\n')
        else:
            script_file.write('# @ error = err.job.$(jobid)\n')
        # output file
        if not type(out_file) == type(None):
            script_file.write('# @ output = ' + out_file + '\n')
        else:
            script_file.write('# @ output = out.job.$(jobid)\n')
        script_file.write('# @ job_type = parallel\n')
        script_file.write('# @ node_usage = not_shared\n')
        script_file.write('# @ resources = ConsumableCpus(1)\n')
        script_file.write('# @ network.MPI = sn_all,not_shared,us\n')
        script_file.write('# @ wall_clock_limit = {0}:{1:0>2d}:00\n'.format(hours, minutes))
        assert(type(self.host_info['environment']) != type(None))
        if self.host_info['environment'] == '16node':
            tasks_per_node = 16
            number_of_nodes = int(math.ceil((nprocesses*1.0/16)))
        script_file.write('# @ node = {0}\n'.format(number_of_nodes))
        script_file.write('# @ tasks_per_node = {0}\n'.format(tasks_per_node))
        first_node_tasks = nprocesses - (number_of_nodes-1)*tasks_per_node
        if (first_node_tasks > 0):
            script_file.write('# @ first_node_tasks = {0}\n'.format(first_node_tasks))
        script_file.write('# @ queue\n')
        script_file.write('LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:' +
                          ':'.join([bfps.lib_dir] + bfps.install_info['library_dirs']) +
                          '\n')
        script_file.write('echo "Start time is `date`"\n')
        script_file.write('cd ' + self.work_dir + '\n')
        script_file.write('poe ' +
                os.path.join(
                    self.work_dir,
                    command_atoms[0]) +
                ' ' +
                ' '.join(command_atoms[1:]) +
                '\n')
        script_file.write('echo "End time is `date`"\n')
        script_file.write('exit 0\n')
        script_file.close()
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
            envprocs = self.host_info['deltanprocs'] * int(math.ceil((nprocesses *1.0/ self.host_info['deltanprocs'])))
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
    def prepare_launch(
            self,
            args = [],
            **kwargs):
        parser = argparse.ArgumentParser('bfps ' + type(self).__name__)
        self.add_parser_arguments(parser)
        opt = parser.parse_args(args)
        self.set_host_info(bfps.host_info)
        if type(opt.environment) != type(None):
            self.host_info['environment'] = opt.environment
        return opt

