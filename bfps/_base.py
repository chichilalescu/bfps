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
import numpy as np
import h5py
from bfps import install_info
from bfps import __version__

class _base(object):
    """This class contains simulation parameters, and handles parameter related
    functionalities of both python objects and C++ codes.
    """
    def __init__(
            self,
            work_dir = './',
            simname = 'test'):
        self.iorank = 0
        ### simulation parameters
        self.parameters = {'nx' : 32,
                           'ny' : 32,
                           'nz' : 32}
        self.string_length = 512
        self.work_dir = os.path.realpath(work_dir)
        self.simname = simname
        return None
    def cdef_pars(
            self,
            parameters = None):
        if type(parameters) == type(None):
            parameters = self.parameters
        key = sorted(list(parameters.keys()))
        src_txt = ''
        for i in range(len(key)):
            if type(parameters[key[i]]) == int:
                src_txt += 'int ' + key[i] + ';\n'
            elif type(parameters[key[i]]) == str:
                src_txt += 'char ' + key[i] + '[{0}];\n'.format(self.string_length)
            elif type(parameters[key[i]]) == np.ndarray:
                src_txt += 'std::vector<'
                if parameters[key[i]].dtype == np.float64:
                    src_txt += 'double'
                elif parameters[key[i]].dtype == np.int:
                    src_txt += 'int'
                src_txt += '> ' + key[i] + ';\n'
            else:
                src_txt += 'double ' + key[i] + ';\n'
        return src_txt
    def cread_pars(
            self,
            parameters = None,
            function_suffix = '',
            file_group = 'parameters'):
        if type(parameters) == type(None):
            parameters = self.parameters
        key = sorted(list(parameters.keys()))
        src_txt = ('int read_parameters' + function_suffix + '()\n{\n' +
                   'hid_t parameter_file;\n' +
                   'hid_t dset, memtype, space;\n' +
                   'char fname[256];\n' +
                   'hsize_t dims[1];\n' +
                   'char string_data[512];\n' +
                   'sprintf(fname, "%s.h5", simname);\n' +
                   'parameter_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);\n')
        for i in range(len(key)):
            src_txt += 'dset = H5Dopen(parameter_file, "/{0}/{1}", H5P_DEFAULT);\n'.format(
                    file_group, key[i])
            if type(parameters[key[i]]) == int:
                src_txt += 'H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &{0});\n'.format(key[i])
            elif type(parameters[key[i]]) == str:
                src_txt += ('space = H5Dget_space(dset);\n' +
                            'memtype = H5Dget_type(dset);\n' +
                            'H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &string_data);\n' +
                            'sprintf({0}, "%s", string_data);\n'.format(key[i]) +
                            'H5Sclose(space);\n' +
                            'H5Tclose(memtype);\n')
            elif type(parameters[key[i]]) == np.ndarray:
                if parameters[key[i]].dtype in [np.int, np.int64, np.int32]:
                    template_par = 'int'
                elif parameters[key[i]].dtype == np.float64:
                    template_par = 'double'
                src_txt += '{0} = read_vector<{1}>(parameter_file, "/{2}/{0}");\n'.format(
                        key[i], template_par, file_group)
            else:
                src_txt += 'H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &{0});\n'.format(key[i])
            src_txt += 'H5Dclose(dset);\n'
        src_txt += 'H5Fclose(parameter_file);\n'
        src_txt += 'return 0;\n}\n' # finishing read_parameters
        return src_txt
    def cprint_pars(self):
        key = sorted(list(self.parameters.keys()))
        src_txt = ''
        for i in range(len(key)):
            if type(self.parameters[key[i]]) == int:
                src_txt += 'DEBUG_MSG("'+ key[i] + ' = %d\\n", ' + key[i] + ');\n'
            elif type(self.parameters[key[i]]) == str:
                src_txt += 'DEBUG_MSG("'+ key[i] + ' = %s\\n", ' + key[i] + ');\n'
            elif type(self.parameters[key[i]]) == np.ndarray:
                src_txt += ('for (unsigned int array_counter=0; array_counter<' +
                            key[i] +
                            '.size(); array_counter++)\n' +
                            '{\n' +
                            'DEBUG_MSG("' + key[i] + '[%d] = %')
                if self.parameters[key[i]].dtype == np.int:
                    src_txt += 'd'
                elif self.parameters[key[i]].dtype == np.float64:
                    src_txt += 'g'
                src_txt += ('\\n", array_counter, ' +
                            key[i] +
                            '[array_counter]);\n}\n')
            else:
                src_txt += 'DEBUG_MSG("'+ key[i] + ' = %g\\n", ' + key[i] + ');\n'
        return src_txt
    def write_par(
            self,
            iter0 = 0):
        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
        ofile = h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'w-')
        for k in self.parameters.keys():
            if (type(self.parameters[k]) == str) and (sys.version_info[0] == 3):
                ofile['parameters/' + k] = bytes(self.parameters[k], 'ascii')
            else:
                ofile['parameters/' + k] = self.parameters[k]
        ofile['iteration'] = int(iter0)
        ofile['bfps_info/solver_class'] = type(self).__name__
        for k in install_info.keys():
            ofile['bfps_info/' + k] = str(install_info[k])
        ofile.close()
        return None
    def rewrite_par(
            self,
            group = None,
            parameters = None):
        assert(group != 'parameters')
        ofile = h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r+')
        for k in parameters.keys():
            if group not in ofile.keys():
                ofile.create_group(group)
            if k not in ofile[group].keys():
                if (type(parameters[k]) == str) and (sys.version_info[0] == 3):
                    ofile[group + '/' + k] = bytes(parameters[k], 'ascii')
                else:
                    ## this code to be used when h5py is fixed
                    # apparently things do work as expected when
                    # the "chunks" parameters is omitted. however,
                    # the default chunk size is the initial size of
                    # the dataset, which could be disastrous for performance...
                    #if (type(parameters[k]) == np.ndarray):
                    #    #ofile[group].create_dataset(
                    #    #        k,
                    #    #        parameters[k].shape,
                    #    #        chunks = (128,),
                    #    #        maxshape=(None),
                    #    #        dtype = parameters[k].dtype)
                    #    ofile[group + '/' + k][...] = parameters[k]
                    #else:
                    ofile[group + '/' + k] = parameters[k]
            else:
                if (type(parameters[k]) == str) and (sys.version_info[0] == 3):
                    ofile[group + '/' + k][...] = bytes(parameters[k], 'ascii')
                else:
                    if (type(parameters[k]) == np.ndarray):
                        if ofile[group + '/' + k].shape[0] != parameters[k].shape[0]:
                            del ofile[group + '/' + k]
                            ofile[group + '/' + k] = parameters[k]
                            #ofile[group + '/' + k].resize(parameters[k].shape)
                        else:
                            ofile[group + '/' + k][...] = parameters[k]
                    ofile[group + '/' + k][...] = parameters[k]
        ofile.close()
        return None
    def read_parameters(self):
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r') as data_file:
            for k in data_file['parameters'].keys():
                if k in self.parameters.keys():
                    if type(self.parameters[k]) in [int, str, float]:
                        self.parameters[k] = type(self.parameters[k])(data_file['parameters/' + k].value)
                    else:
                        self.parameters[k] = data_file['parameters/' + k].value
        return None
    def pars_from_namespace(
            self,
            opt,
            parameters = None,
            get_sim_info = True):
        if type(parameters) == type(None):
            parameters = self.parameters
        cmd_line_pars = vars(opt)
        for k in ['nx', 'ny', 'nz']:
            if type(cmd_line_pars[k]) == type(None):
                cmd_line_pars[k] = opt.n
        for k in parameters.keys():
            if k in cmd_line_pars.keys():
                if not type(cmd_line_pars[k]) == type(None):
                    parameters[k] = cmd_line_pars[k]
        if get_sim_info:
            self.simname = opt.simname
            self.work_dir = os.path.realpath(opt.work_dir)
        return None
    def get_coord(self, direction):
        assert(direction == 'x' or direction == 'y' or direction == 'z')
        return np.arange(.0, self.parameters['n' + direction])*2*np.pi / self.parameters['n' + direction]
    def add_parser_arguments(
            self,
            parser):
        self.specific_parser_arguments(parser)
        self.parameters_to_parser_arguments(parser)
        return None
    def specific_parser_arguments(
            self,
            parser):
        parser.add_argument(
                '-v', '--version',
                action = 'version',
                version = '%(prog)s ' + __version__)
        parser.add_argument(
               '-n', '--cube-size',
               type = int,
               dest = 'n',
               default = 32,
               metavar = 'N',
               help = 'code is run by default in a grid of NxNxN')
        parser.add_argument(
                '--ncpu',
                type = int, dest = 'ncpu',
                default = 2)
        parser.add_argument(
                '--simname',
                type = str, dest = 'simname',
                default = 'test')
        parser.add_argument(
                '--environment',
                type = str,
                dest = 'environment',
                default = None)
        parser.add_argument(
                '--wd',
                type = str, dest = 'work_dir',
                default = './')
        parser.add_argument(
                '--minutes',
                type = int,
                dest = 'minutes',
                default = 5,
                help = 'If environment supports it, this is the requested wall-clock-limit.')
        return None
    def parameters_to_parser_arguments(
            self,
            parser,
            parameters = None):
        if type(parameters) == type(None):
            parameters = self.parameters
        for k in sorted(parameters.keys()):
            parser.add_argument(
                    '--{0}'.format(k),
                    type = type(parameters[k]),
                    dest = k,
                    default = None)
        return None

