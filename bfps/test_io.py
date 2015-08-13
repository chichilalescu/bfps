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
import bfps.fluid_base
import bfps.tools
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

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

