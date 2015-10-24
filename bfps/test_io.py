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



import bfps
import bfps.fluid_base
import bfps.tools
import numpy as np
import pickle
import os

class test_io(bfps.fluid_base.fluid_particle_base):
    def __init__(
            self,
            name = 'test_io',
            work_dir = './',
            simname = 'test'):
        super(test_io, self).__init__(name = name, work_dir = work_dir, simname = simname)
        self.parameters['string_parameter'] = 'test'
        self.fill_up_fluid_code()
        self.finalize_code()
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <string>\n'
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += ('double t;\n')
        self.fluid_start += self.cprint_pars()
        return None

