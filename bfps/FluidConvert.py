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
            use_fftw_wisdom = True):
        _fluid_particle_base.__init__(
                self,
                name = name + '-' + fluid_precision,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision,
                use_fftw_wisdom = use_fftw_wisdom)
        self.parameters['write_rvelocity']  = 1
        self.parameters['write_rvorticity'] = 1
        self.parameters['fluid_name'] = 'test'
        self.parameters['niter_todo'] = 0
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
                        dkx, dky, dkz,
                        FFTW_ESTIMATE);
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
                '--iteration',
                type = int,
                dest = 'iteration',
                default = 0)
        return None
    def launch(
            self,
            args = [],
            **kwargs):
        opt = self.prepare_launch(args)
        self.pars_from_namespace(opt)
        tmp_obj = _base(
                simname = self.simname,
                work_dir = self.work_dir)
        tmp_obj.parameters = {}
        for k in self.parameters.keys():
            tmp_obj.parameters[k] = self.parameters[k]
        _base.read_parameters(tmp_obj)
        for k in ['nx', 'ny', 'nz',
                  'dkx', 'dky', 'dkz',
                  'niter_out']:
            self.parameters[k] = tmp_obj.parameters[k]
        self.simname = opt.simname + '_convert'
        self.parameters['fluid_name'] = opt.simname
        if type(opt.niter_out) != type(None):
            self.parameters['niter_out'] = opt.niter_out
        read_file = os.path.join(
                self.work_dir,
                opt.simname + '_cvorticity_i{0:0>5x}'.format(opt.iteration))
        if not os.path.exists(read_file):
            print('FluidConvert called for nonexistent data. not running.')
            return None
        self.set_host_info(bfps.host_info)
        self.write_par(iter0 = opt.iteration)
        self.run(ncpu = opt.ncpu)
        return None

