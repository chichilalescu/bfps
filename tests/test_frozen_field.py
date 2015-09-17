#! /usr/bin/env python2
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

from test_base import *

class FrozenFieldParticles(bfps.NavierStokes):
    def __init__(
            self,
            name = 'FrozenFieldParticles',
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single'):
        super(FrozenFieldParticles, self).__init__(
                name = name,
                work_dir = work_dir,
                simname = simname,
                fluid_precision = fluid_precision)
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += 'fluid_solver<{0}> *fs;\n'.format(self.C_dtype)
        self.write_fluid_stats()
        self.fluid_start += """
                //begincpp
                char fname[512];
                fs = new fluid_solver<{0}>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz);
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->fk0 = fk0;
                fs->fk1 = fk1;
                strncpy(fs->forcing_type, forcing_type, 128);
                fs->iteration = iteration;
                fs->read('v', 'c');
                //endcpp
                """.format(self.C_dtype)
        self.fluid_loop += """
                //begincpp
                fs->iteration++;
                if (fs->iteration % niter_out == 0)
                    fs->write('v', 'c');
                //endcpp
                """
        self.fluid_end += """
                //begincpp
                if (fs->iteration % niter_out != 0)
                    fs->write('v', 'c');
                delete fs;
                //endcpp
                """
        return None

from test_convergence import convergence_test

if __name__ == '__main__':
    convergence_test(
            parser.parse_args(),
            launch,
            code_class = FrozenFieldParticles)

