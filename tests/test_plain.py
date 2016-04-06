#! /usr/bin/env python2
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



#from base import *
import bfps
import sys

import numpy as np
import matplotlib.pyplot as plt

#parser.add_argument('--multiplejob',
#        dest = 'multiplejob', action = 'store_true')
#
#parser.add_argument(
#        '--particle-class',
#        default = 'particles',
#        dest = 'particle_class',
#        type = str)
#
#parser.add_argument(
#        '--interpolator-class',
#        default = 'interpolator',
#        dest = 'interpolator_class',
#        type = str)

class NSPlain(bfps.NavierStokes):
    def specific_parser_arguments(
            self,
            parser):
        bfps.NavierStokes.specific_parser_arguments(self, parser)
        parser.add_argument(
                '--particle-class',
                default = 'rFFTW_distributed_particles',
                dest = 'particle_class',
                type = str)
        parser.add_argument(
                '--interpolator-class',
                default = 'rFFTW_interpolator',
                dest = 'interpolator_class',
                type = str)
        parser.add_argument('--neighbours',
                type = int,
                dest = 'neighbours',
                default = 3)
        parser.add_argument('--smoothness',
                type = int,
                dest = 'smoothness',
                default = 2)
        return None
    def launch(
            self,
            args = [],
            **kwargs):
        opt = self.prepare_launch(args = args)
        self.fill_up_fluid_code()
        if type(opt.nparticles) == int:
            if opt.nparticles > 0:
                self.add_3D_rFFTW_field(
                        name = 'rFFTW_acc')
                self.add_interpolator(
                        name = 'spline',
                        neighbours = opt.neighbours,
                        smoothness = opt.smoothness,
                        class_name =  opt.interpolator_class)
                self.add_particles(
                        kcut = ['fs->kM/2', 'fs->kM/3'],
                        integration_steps = 3,
                        interpolator = 'spline',
                        class_name = opt.particle_class)
                self.add_particles(
                        integration_steps = [2, 3, 4, 6],
                        interpolator = 'spline',
                        acc_name = 'rFFTW_acc',
                        class_name = opt.particle_class)
        self.finalize_code()
        self.launch_jobs(opt = opt)
        return None

def plain(args):
    wd = opt.work_dir
    opt.work_dir = wd + '/N{0:0>3x}_1'.format(opt.n)
    c0 = launch(opt, dt = 0.2/opt.n,
            particle_class = opt.particle_class,
            interpolator_class = opt.interpolator_class)
    c0.compute_statistics()
    print ('Re = {0:.0f}'.format(c0.statistics['Re']))
    print ('Rlambda = {0:.0f}'.format(c0.statistics['Rlambda']))
    print ('Lint = {0:.4e}, etaK = {1:.4e}'.format(c0.statistics['Lint'], c0.statistics['etaK']))
    print ('Tint = {0:.4e}, tauK = {1:.4e}'.format(c0.statistics['Tint'], c0.statistics['tauK']))
    print ('kMetaK = {0:.4e}'.format(c0.statistics['kMeta']))
    for s in range(c0.particle_species):
        acceleration_test(c0, species = s, m = 1)
    if not opt.multiplejob:
        return None
    assert(opt.niter_todo % 3 == 0)
    opt.work_dir = wd + '/N{0:0>3x}_2'.format(opt.n)
    opt.njobs *= 2
    opt.niter_todo = opt.niter_todo//2
    c1 = launch(opt, dt = c0.parameters['dt'],
            particle_class = opt.particle_class,
            interpolator_class = opt.interpolator_class)
    c1.compute_statistics()
    opt.work_dir = wd + '/N{0:0>3x}_3'.format(opt.n)
    opt.njobs = 3*opt.njobs//2
    opt.niter_todo = 2*opt.niter_todo//3
    c2 = launch(opt, dt = c0.parameters['dt'],
            particle_class = opt.particle_class,
            interpolator_class = opt.interpolator_class)
    c2.compute_statistics()
    compare_stats(opt, c0, c1)
    compare_stats(opt, c0, c2)
    return None

if __name__ == '__main__':
    c = NSPlain()
    c.launch(
            ['-n', '32',
             '--ncpu', '4',
             '--nparticles', '1000',
             '--niter_todo', '48',
             '--wd', 'data/single'] +
            sys.argv[1:])

