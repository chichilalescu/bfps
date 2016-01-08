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
from base import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = parser.parse_args(
            ['-n', '32',
             '--run',
             '--ncpu', '2',
             '--initialize',
             '--neighbours', '4',
             '--smoothness', '1',
             '--nparticles', '4225',
             '--niter_todo', '1',
             '--niter_stat', '1',
             '--niter_part', '1',
             '--niter_out', '1',
             '--wd', 'interp'] +
            sys.argv[1:])
    c = bfps.NavierStokes(
            name = 'test_interpolation',
            use_fftw_wisdom = False,
            frozen_fields = True)
    c.pars_from_namespace(opt)
    c.fill_up_fluid_code()
    c.add_particle_fields(
            name = 'n{0}m{1}'.format(opt.neighbours, opt.smoothness),
            neighbours = opt.neighbours,
            smoothness = opt.smoothness)
    c.add_particle_fields(
            name = 'n{0}'.format(opt.neighbours),
            neighbours = opt.neighbours,
            interp_type = 'Lagrange')
    c.add_particles(
            integration_steps = 1,
            fields_name = 'n{0}m{1}'.format(opt.neighbours, opt.smoothness))
    c.add_particles(
            integration_steps = 1,
            fields_name = 'n{0}'.format(opt.neighbours))
    c.finalize_code()
    c.write_src()
    c.write_par()
    nn = int(c.parameters['nparticles']**.5)
    pos = np.zeros((nn, nn, 3), dtype = np.float64)
    pos[:, :, 0] = np.linspace(0, 2*np.pi, nn)[None, :]
    pos[:, :, 2] = np.linspace(0, 2*np.pi, nn)[:, None]
    pos[:, :, 1] = np.random.random()*2*np.pi
    c.generate_vector_field(write_to_file = True,
                            spectra_slope = 1.)
    c.generate_tracer_state(
            species = 0,
            write_to_file = False,
            data = pos.reshape(-1, 3))
    c.generate_tracer_state(
            species = 1,
            write_to_file = False,
            data = pos.reshape(-1, 3))
    c.set_host_info({'type' : 'pc'})
    if opt.run:
        c.run(ncpu = opt.ncpu)
    df = c.get_data_file()
    fig = plt.figure(figsize=(10,5))
    a = fig.add_subplot(121)
    a.contour(
            df['particles/tracers0/state'][0, :, 0].reshape(nn, nn),
            df['particles/tracers0/state'][0, :, 2].reshape(nn, nn),
            df['particles/tracers0/velocity'][1, :, 0].reshape(nn, nn))
    a = fig.add_subplot(122)
    a.contour(
            df['particles/tracers1/state'][0, :, 0].reshape(nn, nn),
            df['particles/tracers1/state'][0, :, 2].reshape(nn, nn),
            df['particles/tracers1/velocity'][1, :, 0].reshape(nn, nn))
    fig.savefig('interp.pdf')

