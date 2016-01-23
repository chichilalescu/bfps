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
    def __init__(
            self,
            base_class = NavierStokes):
        self.base_class = base_class
        self.parser = argparse.ArgumentParser(prog = 'bfps-' + self.base_class.__name__)
        # generic arguments, all classes may use these
        self.parser.add_argument(
                '-v', '--version',
                action = 'version',
                version = '%(prog)s ' + bfps.__version__)
        self.parser.add_argument(
                '--ncpu',
                type = int, dest = 'ncpu',
                default = 2)
        self.parser.add_argument(
                '--simname',
                type = str, dest = 'simname',
                default = 'test')
        self.parser.add_argument(
                '--environment',
                type = str,
                dest = 'environment',
                default = '')
        self.parser.add_argument(
                '--wd',
                type = str, dest = 'work_dir',
                default = './')
        # now add class specific arguments
        self.add_class_specific_arguments(self.base_class.__name__)
        # now add code parameters
        c = self.base_class()
        for k in sorted(c.parameters.keys()):
            self.parser.add_argument(
                    '--{0}'.format(k),
                    type = type(c.parameters[k]),
                    dest = k,
                    default = None)
        return None
    def add_class_specific_arguments(
            self,
            class_name):
        """add arguments specific to different classes.

        .. warning:: Reimplement this for custom classes.
        """
        if class_name in ['NavierStokes', 'FluidResize']:
            self.parser.add_argument(
                    '--src-wd',
                    type = str,
                    dest = 'src_work_dir',
                    default = './')
            self.parser.add_argument(
                    '--src-simname',
                    type = str,
                    dest = 'src_simname',
                    default = '')
            self.parser.add_argument(
                    '--src-iteration',
                    type = int,
                    dest = 'src_iteration',
                    default = 0)
        if class_name == 'FluidResize':
            self.parser.add_argument(
                    '-n',
                    type = int,
                    dest = 'n',
                    default = 32,
                    metavar = 'N',
                    help = 'resize to N')
        if class_name == 'NavierStokes':
            self.parser.add_argument(
                    '-n',
                    type = int,
                    dest = 'n',
                    default = 32,
                    metavar = 'N',
                    help = 'code is run by default in a grid of NxNxN')
            self.parser.add_argument(
                    '--precision',
                    type = str, dest = 'precision',
                    default = 'single')
            self.parser.add_argument(
                    '--njobs',
                    type = int, dest = 'njobs',
                    default = 1)
            self.parser.add_argument(
                    '--QR-stats',
                    action = 'store_true',
                    dest = 'QR_stats',
                    help = 'add this option if you want to compute velocity gradient and QR stats')
            self.parser.add_argument(
                    '--kMeta',
                    type = float,
                    dest = 'kMeta',
                    default = 2.0)
            self.parser.add_argument(
                    '--dtfactor',
                    type = float,
                    dest = 'dtfactor',
                    default = 0.5,
                    help = 'dt is computed as DTFACTOR / N')
            self.parser.add_argument(
                    '--particle-rand-seed',
                    type = int,
                    dest = 'particle_rand_seed',
                    default = None)
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


