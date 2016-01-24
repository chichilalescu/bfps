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



import os

import argparse

import bfps
from .NavierStokes import NavierStokes
from .FluidResize import FluidResize
from .FluidConvert import FluidConvert

class Launcher:
    """Objects of this class are used in the executable bfps script.
    It should work with any children of
    :class:`NavierStokes <NavierStokes.NavierStokes>`;
    failure to do so should be reported as a bug.
    """
    def __init__(
            self,
            base_class = NavierStokes):
        self.base_class = base_class
        self.parser = argparse.ArgumentParser(prog = 'bfps ' + self.base_class.__name__)
        c = self.base_class()
        c.add_parser_arguments(self.parser)
        return None
    def __call__(
            self,
            args = None):
        opt = self.parser.parse_args(args)
        if opt.environment != '':
            bfps.host_info['environment'] = opt.environment
        opt.work_dir = os.path.join(
                os.path.realpath(opt.work_dir),
                'N{0:0>4}'.format(opt.n))
        if self.base_class.__name__ == 'NavierStokes':
            c = self.base_class(
                    work_dir = opt.work_dir,
                    fluid_precision = opt.precision,
                    simname = opt.simname,
                    QR_stats_on = opt.QR_stats)
            # with the default Lundgren forcing, I can estimate the dissipation
            # with nondefault forcing, figure out the amplitude for this viscosity
            # yourself
            c.parameters['nu'] = (opt.kMeta * 2 / opt.n)**(4./3)
            c.parameters['dt'] = (opt.dtfactor / opt.n)
            if ((c.parameters['niter_todo'] % c.parameters['niter_out']) != 0):
                c.parameters['niter_out'] = c.parameters['niter_todo']
            if c.QR_stats_on:
                # max_Q_estimate and max_R_estimate are just used for the 2D pdf
                # therefore I just want them to be small multiples of mean trS2
                # I'm already estimating the dissipation with kMeta...
                meantrS2 = (opt.n//2 / opt.kMeta)**4 * c.parameters['nu']**2
                c.parameters['max_Q_estimate'] = meantrS2
                c.parameters['max_R_estimate'] = .4*meantrS2**1.5

            # command line parameters will overwrite any defaults
            cmd_line_pars = vars(opt)
            for k in ['nx', 'ny', 'nz']:
                if type(cmd_line_pars[k]) == type(None):
                    cmd_line_pars[k] = opt.n
            for k in c.parameters.keys():
                if k in cmd_line_pars.keys():
                    if not type(cmd_line_pars[k]) == type(None):
                        c.parameters[k] = cmd_line_pars[k]
            c.fill_up_fluid_code()
            c.finalize_code()
            c.write_src()
            c.set_host_info(bfps.host_info)
            if not os.path.exists(os.path.join(c.work_dir, c.simname + '.h5')):
                c.write_par()
                if c.parameters['nparticles'] > 0:
                    data = c.generate_tracer_state(species = 0, rseed = opt.particle_rand_seed)
                    for s in range(1, c.particle_species):
                        c.generate_tracer_state(species = s, data = data)
                init_condition_file = os.path.join(
                        c.work_dir,
                        c.simname + '_cvorticity_i{0:0>5x}'.format(0))
                if not os.path.exists(init_condition_file):
                    if len(opt.src_simname) > 0:
                        src_file = os.path.join(
                                c.work_dir,
                                opt.src_simname + '_cvorticity_i{0:0>5x}'.format(opt.src_iteration))
                        os.symlink(src_file, init_condition_file)
                    else:
                       c.generate_vector_field(
                               write_to_file = True,
                               spectra_slope = 2.0,
                               amplitude = 0.25)
            c.run(ncpu = opt.ncpu,
                  njobs = opt.njobs)
        return c


