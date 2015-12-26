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

class FrozenFieldParticles(bfps.NavierStokes):
    def __init__(
            self,
            name = 'FrozenFieldParticles',
            work_dir = './',
            simname = 'test',
            frozen_fields = True,
            fluid_precision = 'single',
            use_fftw_wisdom = False):
        super(FrozenFieldParticles, self).__init__(
                name = name,
                work_dir = work_dir,
                simname = simname,
                fluid_precision = fluid_precision,
                frozen_fields = frozen_fields,
                use_fftw_wisdom = use_fftw_wisdom)
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

