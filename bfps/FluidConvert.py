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
from ._base import _base
import bfps

class FluidConvert(_fluid_particle_base):
    """This class is meant to be used for conversion of native DNS field
    representations to real-space representations of velocity/vorticity
    fields.
    It may be superseeded by streamlined functionality in the future...
    """
    def __init__(
            self,
            name = 'FluidConvert-v' + bfps.__version__,
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single',
            use_fftw_wisdom = False):
        _fluid_particle_base.__init__(
                self,
                name = name + '-' + fluid_precision,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision,
                use_fftw_wisdom = use_fftw_wisdom)
        self.spec_parameters = {}
        self.spec_parameters['write_rvelocity']  = 1
        self.spec_parameters['write_rvorticity'] = 1
        self.spec_parameters['write_rTrS2'] = 1
        self.spec_parameters['write_renstrophy'] = 1
        self.spec_parameters['write_rpressure'] = 1
        self.spec_parameters['iter0'] = 0
        self.spec_parameters['iter1'] = -1
        self.fill_up_fluid_code()
        self.finalize_code(postprocess_mode = True)
        return None
    def fill_up_fluid_code(self):
        self.definitions += self.cread_pars(
                parameters = self.spec_parameters,
                function_suffix = '_specific',
                file_group = 'conversion_parameters')
        self.variables += self.cdef_pars(
                parameters = self.spec_parameters)
        self.main_start += 'read_parameters_specific();\n'
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
                    if (write_rTrS2)
                        bla->write_rTrS2();
                    if (write_renstrophy)
                        bla->write_renstrophy();
                    if (write_rpressure)
                        bla->write_rpressure();
                }}
                //endcpp
                """.format(self.C_dtype)
        self.fluid_start += """
                //begincpp
                fs = new fluid_solver<{0}>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz,
                        dealias_type,
                        DEFAULT_FFTW_FLAG);
                //endcpp
                """.format(self.C_dtype)
        self.fluid_loop += """
                //begincpp
                fs->iteration = frame_index;
                do_conversion(fs);
                //endcpp
                """
        self.fluid_end += 'delete fs;\n'
        return None
    def specific_parser_arguments(
            self,
            parser):
        _fluid_particle_base.specific_parser_arguments(self, parser)
        self.parameters_to_parser_arguments(
                parser,
                parameters = self.spec_parameters)
        return None
    def launch(
            self,
            args = [],
            **kwargs):
        opt = self.prepare_launch(args)
        if opt.iter1 == -1:
            opt.iter1 = self.get_data_file()['iteration'].value
        self.pars_from_namespace(
                opt,
                parameters = self.spec_parameters)
        self.rewrite_par(
                group = 'conversion_parameters',
                parameters = self.spec_parameters)
        self.run(ncpu = opt.ncpu,
                 hours = opt.minutes // 60,
                 minutes = opt.minutes % 60,
                 err_file = 'err_convert',
                 out_file = 'out_convert')
        return None

