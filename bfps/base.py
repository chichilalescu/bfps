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

class base(object):
    def __init__(self):
        self.iorank = 0
        ### simulation parameters
        self.parameters = {'nx' : 32,
                           'ny' : 32,
                           'nz' : 32}
        self.string_length = 512
        self.work_dir = './'
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
                 + 'int err_while_reading = 0, errr;\n'
                 + 'if (myrank == {0})'.format(self.iorank)
                 + '\n{\n'
                 + 'FILE *par_file;\n'
                 + 'char fname[{0}];\n'.format(self.string_length)
                 + 'sprintf(fname, "%s_pars.txt", simname);\n'
                 + 'par_file = fopen(fname, "r");\n')
        #src_txt += 'std::cerr << fname << std::endl;\n'
        for i in range(len(key)):
            if type(self.parameters[key[i]]) == int:
                src_txt += ('if (fscanf(par_file, "' + key[i] + ' = %d\\n", &' + key[i] + ') != 1)\n'
                          + '    err_while_reading++;\n')
            elif type(self.parameters[key[i]]) == str:
                src_txt += ('if (fscanf(par_file, "' + key[i] + ' = %s\\n", ' + key[i] + ') != 1)\n'
                          + '    err_while_reading++;\n')
            else:
                src_txt += ('if (fscanf(par_file, "' + key[i] + ' = %le\\n", &' + key[i] + ') != 1)\n'
                          + '    err_while_reading++;\n')
        src_txt += '}\n' # finishing if myrank == 0
        # now broadcasting values to all ranks
        for i in range(len(key)):
            if type(self.parameters[key[i]]) == int:
                src_txt += 'MPI_Bcast((void*)(&' + key[i] + '), 1, MPI_INTEGER, {0}, MPI_COMM_WORLD);\n'.format(self.iorank)
            elif type(self.parameters[key[i]]) == str:
                src_txt += 'MPI_Bcast((void*)(' + key[i] + '), {0}, MPI_CHAR, {1}, MPI_COMM_WORLD);\n'.format(self.string_length, self.iorank)
            else:
                src_txt += 'MPI_Bcast((void*)(&' + key[i] + '), 1, MPI_DOUBLE, {0}, MPI_COMM_WORLD);\n'.format(self.iorank)
        src_txt += ('MPI_Allreduce((void*)(&err_while_reading), (void*)(&errr), 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);\n'
                  + 'if (errr > 0)\n{\n'
                  + 'fprintf(stderr, "Error reading parameters.\\nAttempting to exit.\\n");\n'
                  + 'MPI_Finalize();\n'
                  + 'exit(0);\n'
                  + '}\n'             # finishing errr check
                  + 'return 0;\n}\n') # finishing read_parameters
        return src_txt
    def cprint_pars(self):
        key = self.parameters.keys()
        key.sort()
        src_txt = ''
        for i in range(len(key)):
            if type(self.parameters[key[i]]) == int:
                src_txt += ('fprintf(stderr, "myrank = %d, '
                          + key[i] + ' = %d\\n", myrank, ' + key[i] + ');\n')
            elif type(self.parameters[key[i]]) == str:
                src_txt += ('fprintf(stderr, "myrank = %d, '
                          + key[i] + ' = %s\\n", myrank, ' + key[i] + ');\n')
            else:
                src_txt += ('fprintf(stderr, "myrank = %d, '
                          + key[i] + ' = %le\\n", myrank, ' + key[i] + ');\n')
        return src_txt
    def write_par(self, simname = 'test'):
        filename = simname + '_pars.txt'
        ofile = open(os.path.join(self.work_dir, filename), 'w')
        key = self.parameters.keys()
        key.sort()
        for i in range(len(key)):
            if type(self.parameters[key[i]]) == float:
                ofile.write(('{0} = {1:e}\n').format(key[i], self.parameters[key[i]]))
            else:
                ofile.write('{0} = {1}\n'.format(key[i], self.parameters[key[i]]))
        ofile.close()
        return None
    def get_coord(self, direction):
        assert(direction == 'x' or direction == 'y' or direction == 'z')
        return np.arange(.0, self.parameters['n' + direction])*2*np.pi / self.parameters['n' + direction]

