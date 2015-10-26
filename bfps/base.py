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
import h5py
import bfps

class base(object):
    """
        This class contains simulation parameters, and handles parameter related
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
        self.work_dir = work_dir
        self.simname = simname
        return None
    def cdef_pars(self):
        key = self.parameters.keys()
        key.sort()
        src_txt = ''
        for i in range(len(key)):
            if type(self.parameters[key[i]]) == int:
                src_txt += 'int ' + key[i] + ';\n'
            elif type(self.parameters[key[i]]) == str:
                src_txt += 'char ' + key[i] + '[{0}];\n'.format(self.string_length)
            else:
                src_txt += 'double ' + key[i] + ';\n'
        return src_txt
    def cread_pars(self):
        key = self.parameters.keys()
        key.sort()
        src_txt = ('int read_parameters(hid_t data_file_id)\n{\n'
                 + 'hid_t dset, memtype, space;\n'
                 + 'hsize_t dims[1];\n'
                 + 'char **string_data;\n'
                 + 'std::string tempstr;\n')
        for i in range(len(key)):
            src_txt += 'dset = H5Dopen(data_file_id, "parameters/{0}", H5P_DEFAULT);\n'.format(key[i])
            if type(self.parameters[key[i]]) == int:
                src_txt += 'H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &{0});\n'.format(key[i])
            elif type(self.parameters[key[i]]) == str:
                src_txt += ('space = H5Dget_space(dset);\n' +
                            'memtype = H5Tcopy(H5T_C_S1);\n' +
                            'H5Tset_size(memtype, H5T_VARIABLE);\n' +
                            'H5Sget_simple_extent_dims(space, dims, NULL);\n' +
                            'string_data = (char**)malloc(dims[0]*sizeof(char));\n' +
                            'H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, string_data);\n' +
                            'sprintf({0}, "%s", string_data[0]);\n'.format(key[i]) +
                            'free(string_data);\n' +
                            'H5Sclose(space);\n' +
                            'H5Tclose(memtype);\n')
            else:
                src_txt += 'H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &{0});\n'.format(key[i])
            src_txt += 'H5Dclose(dset);\n'
        src_txt += 'return 0;\n}\n' # finishing read_parameters
        return src_txt
    def cprint_pars(self):
        key = self.parameters.keys()
        key.sort()
        src_txt = ''
        for i in range(len(key)):
            if type(self.parameters[key[i]]) == int:
                src_txt += 'DEBUG_MSG("'+ key[i] + ' = %d\\n", ' + key[i] + ');\n'
            elif type(self.parameters[key[i]]) == str:
                src_txt += 'DEBUG_MSG("'+ key[i] + ' = %s\\n", ' + key[i] + ');\n'
            else:
                src_txt += 'DEBUG_MSG("'+ key[i] + ' = %g\\n", ' + key[i] + ');\n'
        return src_txt
    def write_par(self, iter0 = 0):
        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)
        ofile = h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'w-')
        for k in self.parameters.keys():
            ofile['parameters/' + k] = self.parameters[k]
        ofile['iteration'] = int(iter0)
        for k in bfps.install_info.keys():
            ofile['install_info/' + k] = str(bfps.install_info[k])
        ofile.close()
        return None
    def read_parameters(self):
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r') as data_file:
            for k in data_file['parameters'].keys():
                self.parameters[k] = type(self.parameters[k])(data_file['parameters/' + k].value)
        return None
    def pars_from_namespace(self, opt):
        new_pars = vars(opt)
        self.simname = opt.simname
        self.work_dir = opt.work_dir
        for k in self.parameters.keys():
            self.parameters[k] = new_pars[k]
        return None
    def get_coord(self, direction):
        assert(direction == 'x' or direction == 'y' or direction == 'z')
        return np.arange(.0, self.parameters['n' + direction])*2*np.pi / self.parameters['n' + direction]

