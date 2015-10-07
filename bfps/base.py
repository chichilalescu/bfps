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
        src_txt = ('int read_parameters()\n{\n'
                 + 'H5::DataSet dset;\n'
                 + 'H5::StrType strdtype(0, H5T_VARIABLE);\n'
                 + 'H5::DataSpace strdspace(H5S_SCALAR);\n'
                 + 'std::string tempstr;')
        for i in range(len(key)):
            src_txt += 'dset = data_file->openDataSet("parameters/{0}");\n'.format(key[i])
            if type(self.parameters[key[i]]) == int:
                src_txt += 'dset.read(&{0}, H5::PredType::NATIVE_INT);\n'.format(key[i])
            elif type(self.parameters[key[i]]) == str:
                src_txt += ('dset.read(tempstr, strdtype, strdspace);\n' +
                            'sprintf({0}, "%s", tempstr.c_str());\n').format(key[i])
            else:
                src_txt += 'dset.read(&{0}, H5::PredType::NATIVE_DOUBLE);\n'.format(key[i])
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
                src_txt += 'DEBUG_MSG("'+ key[i] + ' = %le\\n", ' + key[i] + ');\n'
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
                self.parameters[k] = data_file['parameters/' + k].value
        return None
    def get_coord(self, direction):
        assert(direction == 'x' or direction == 'y' or direction == 'z')
        return np.arange(.0, self.parameters['n' + direction])*2*np.pi / self.parameters['n' + direction]

