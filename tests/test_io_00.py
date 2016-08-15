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



from base import *

import bfps
from bfps._code import _code

class test_io(_code):
    def __init__(
            self,
            name = 'test_io',
            work_dir = './',
            simname = 'test'):
        super(test_io, self).__init__(work_dir = work_dir, simname = simname)
        self.name = name
        self.parameters['string_parameter'] = 'test string'
        self.parameters['other_string_parameter'] = 'another test string'
        self.parameters['niter_todo'] = 0
        self.parameters['real_number'] = 1.21
        self.parameters['real_array'] = np.array([1.3, 1.5, 0.4])
        self.parameters['int_array'] = np.array([1, 3, 5, 4])
        self.main_start += self.cprint_pars()
        return None

if __name__ == '__main__':
    opt = parser.parse_args(
            ['-n', '32',
             '--ncpu', '2'] +
            sys.argv[1:])
    print('about to create test_io object')
    c = test_io(work_dir = opt.work_dir + '/io')
    print('congratulations, test_io object was created')

