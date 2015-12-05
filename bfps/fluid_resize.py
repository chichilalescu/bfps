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

import numpy as np

class fluid_resize(bfps.fluid_base.fluid_particle_base):
    def __init__(
            self,
            name = 'fluid_resize',
            work_dir = './',
            simname = 'test',
            dtype = np.float32,
            use_fftw_wisdom = False):
        super(fluid_resize, self).__init__(
                name = name,
                work_dir = work_dir,
                simname = simname,
                dtype = dtype,
                use_fftw_wisdom = use_fftw_wisdom)
        self.parameters['src_simname'] = 'test'
        self.parameters['dst_iter'] = 0
        self.parameters['dst_nx'] = 32
        self.parameters['dst_ny'] = 32
        self.parameters['dst_nz'] = 32
        self.parameters['dst_simname'] = 'new_test'
        self.parameters['dst_dkx'] = 1.0
        self.parameters['dst_dky'] = 1.0
        self.parameters['dst_dkz'] = 1.0
        self.fill_up_fluid_code()
        self.finalize_code()
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_includes += '#include "fftw_tools.hpp"\n'
        self.fluid_variables += ('double t;\n' +
                                 'fluid_solver<' + self.C_dtype + '> *fs0, *fs1;\n')
        self.fluid_start += """
                //begincpp
                char fname[512];
                fs0 = new fluid_solver<{0}>(
                        src_simname,
                        nx, ny, nz,
                        dkx, dky, dkz);
                fs1 = new fluid_solver<{0}>(
                        dst_simname,
                        dst_nx, dst_ny, dst_nz,
                        dst_dkx, dst_dky, dst_dkz);
                fs0->iteration = iteration;
                fs1->iteration = 0;
                DEBUG_MSG("about to read field\\n");
                fs0->read('v', 'c');
                DEBUG_MSG("field read, about to copy data\\n");
                double a, b;
                fs0->compute_velocity(fs0->cvorticity);
                a = 0.5*fs0->autocorrel(fs0->cvelocity);
                b = 0.5*fs0->autocorrel(fs0->cvorticity);
                DEBUG_MSG("old field %d %g %g\\n", fs0->iteration, a, b);
                copy_complex_array<{0}>(fs0->cd, fs0->cvorticity,
                                        fs1->cd, fs1->cvorticity,
                                        3);
                DEBUG_MSG("data copied, about to write new field\\n");
                fs1->write('v', 'c');
                DEBUG_MSG("finished writing\\n");
                fs1->compute_velocity(fs1->cvorticity);
                a = 0.5*fs1->autocorrel(fs1->cvelocity);
                b = 0.5*fs1->autocorrel(fs1->cvorticity);
                DEBUG_MSG("new field %d %g %g\\n", fs1->iteration, a, b);
                //endcpp
                """.format(self.C_dtype)
        self.fluid_end += """
                //begincpp
                delete fs0;
                delete fs1;
                //endcpp
                """
        return None

