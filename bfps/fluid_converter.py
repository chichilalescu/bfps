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

class fluid_converter(bfps.fluid_base.fluid_particle_base):
    def __init__(
            self,
            name = 'fluid_converter',
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single'):
        super(fluid_converter, self).__init__(
                name = name,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision)
        self.parameters['write_rvelocity']  = 1
        self.parameters['write_rvorticity'] = 1
        self.parameters['fluid_name'] = 'test'
        self.fill_up_fluid_code()
        self.finalize_code()
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += ('double t;\n' +
                                 'fluid_solver<{0}> *fs;\n').format(self.C_dtype)
        self.fluid_definitions += """
                //begincpp
                void do_conversion(fluid_solver<{0}> *bla)
                {{
                    bla->read('v', 'c');
                    if (write_rvelocity)
                        bla->write('u', 'r');
                    if (write_rvorticity)
                        bla->write('v', 'r');
                }}
                //endcpp
                """.format(self.C_dtype)
        self.fluid_start += """
                //begincpp
                fs = new fluid_solver<{0}>(
                        fluid_name,
                        nx, ny, nz,
                        dkx, dky, dkz);
                fs->iteration = iteration;
                do_conversion(fs);
                //endcpp
                """.format(self.C_dtype)
        self.fluid_loop += """
                //begincpp
                fs->iteration++;
                if (fs->iteration % niter_out == 0)
                    do_conversion(fs);
                //endcpp
                """
        self.fluid_end += 'delete fs;\n'
        return None

