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



import numpy as np
from test_base import *

class test_interpolation(bfps.NavierStokes):
    def __init__(
            self,
            name = 'test_interpolation',
            work_dir = './',
            simname = 'test'):
        super(test_interpolation, self).__init__(
                work_dir = work_dir,
                simname = simname,
                name = name)
        return None

if __name__ == '__main__':
    opt = parser.parse_args()
    c = test_interpolation(work_dir = opt.work_dir + '/io')
    c.pars_from_namespace(opt)
    c.add_particles(
            integration_steps = 1,
            neighbours = 1,
            smoothness = 1)
    c.fill_up_fluid_code()
    c.finalize_code()
    c.write_src()
    c.write_par()
    pos = np.zeros((c.parameters['nparticles'], 3), dtype = np.float64)
    pos[:, 0] = np.linspace(0, 2*np.pi, pos.shape[0])
    c.generate_tracer_state(
            species = 0,
            write_to_file = False,
            data = pos)
    c.set_host_info({'type' : 'pc'})
    if opt.run:
        c.run(ncpu = opt.ncpu)

