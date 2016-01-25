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



import numpy as np
import pickle
import os
from ._fluid_base import _fluid_particle_base

class FluidConvert(_fluid_particle_base):
    """This class is meant to be used for conversion of native DNS field
    representations to real-space representations of velocity/vorticity
    fields.
    It may be superseeded by streamlined functionality in the future...
    """
    def __init__(
            self,
            name = 'FluidConvert',
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single',
            use_fftw_wisdom = True):
        _fluid_particle_base.__init__(
                self,
                name = name,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision,
                use_fftw_wisdom = use_fftw_wisdom)
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
    def specific_parser_arguments(
            self,
            parser):
        _fluid_particle_base.specific_parser_arguments(self, parser)
        parser.add_argument(
                '--src-wd',
                type = str,
                dest = 'src_work_dir',
                default = './')
        parser.add_argument(
                '--src-simname',
                type = str,
                dest = 'src_simname',
                default = '')
        parser.add_argument(
                '--src-iteration',
                type = int,
                dest = 'src_iteration',
                default = 0)
        return None

