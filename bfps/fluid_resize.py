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

class fluid_resize(bfps.fluid_base.fluid_particle_base):
    def __init__(
            self,
            name = 'fluid_converter',
            work_dir = './',
            simname = 'test'):
        super(fluid_resize, self).__init__(
                name = name,
                work_dir = work_dir,
                simname = simname)
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
                                 'fluid_solver<float> *fs0, *fs1;\n')
        self.fluid_start += """
                //begincpp
                char fname[512];
                fs0 = new fluid_solver<float>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz);
                fs1 = new fluid_solver<float>(
                        dst_simname,
                        dst_nx, dst_ny, dst_nz,
                        dst_dkx, dst_dky, dst_dkz);
                fs0->iteration = iteration;
                fs1->iteration = 0;
                fs0->read('v', 'c');
                double a, b;
                a = 0.5*fs0->correl_vec(fs0->cvelocity, fs0->cvelocity);
                b = 0.5*fs0->correl_vec(fs0->cvorticity, fs0->cvorticity);
                DEBUG_MSG("old field %d %g %g\\n", fs0->iteration, a, b);
                copy_complex_array(fs0->cd, fs0->cvorticity,
                                   fs1->cd, fs1->cvorticity,
                                   3);
                fs1->write('v', 'c');
                a = 0.5*fs1->correl_vec(fs1->cvelocity, fs1->cvelocity);
                b = 0.5*fs1->correl_vec(fs1->cvorticity, fs1->cvorticity);
                DEBUG_MSG("new field %d %g %g\\n", fs1->iteration, a, b);
                niter_todo = 0;
                //endcpp
                """
        self.fluid_end += """
                //begincpp
                delete fs0;
                delete fs1;
                //endcpp
                """
        return None

