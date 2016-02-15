#! /usr/bin/env python
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

import numpy as np
import matplotlib.pyplot as plt

def scaling(opt):
    wd = opt.work_dir
    opt.work_dir = wd + '/N{0:0>3x}_1'.format(opt.n)
    c0 = launch(opt, dt = 0.2/opt.n, particle_class = 'distributed_particles', interpolator_class = 'interpolator')
    c0.compute_statistics()
    print ('Re = {0:.0f}'.format(c0.statistics['Re']))
    print ('Rlambda = {0:.0f}'.format(c0.statistics['Rlambda']))
    print ('Lint = {0:.4e}, etaK = {1:.4e}'.format(c0.statistics['Lint'], c0.statistics['etaK']))
    print ('Tint = {0:.4e}, tauK = {1:.4e}'.format(c0.statistics['Tint'], c0.statistics['tauK']))
    print ('kMetaK = {0:.4e}'.format(c0.statistics['kMeta']))
    for s in range(c0.particle_species):
        acceleration_test(c0, species = s, m = 1)
    opt.work_dir = wd + '/N{0:0>3x}_2'.format(opt.n)
    c1 = launch(opt, dt = c0.parameters['dt'], particle_class = 'particles')
    c1.compute_statistics()
    compare_stats(opt, c0, c1)
    opt.work_dir = wd + '/N{0:0>3x}_3'.format(opt.n)
    c2 = launch(opt, dt = c0.parameters['dt'], particle_class = 'particles', interpolator_class = 'interpolator')
    c2.compute_statistics()
    compare_stats(opt, c0, c2)
    compare_stats(opt, c1, c2)
    return None

if __name__ == '__main__':
    opt = parser.parse_args(
            ['-n', '32',
             '--run',
             '--initialize',
             '--ncpu', '4',
             '--nparticles', '10000',
             '--niter_todo', '8',
             '--precision', 'single',
             '--wd', 'data/single'] +
            sys.argv[1:])
    scaling(opt)

