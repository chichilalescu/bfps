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
import subprocess
import pickle

import bfps
from bfps import fluid_resize

parser = bfps.get_parser()
parser.add_argument('--initialize', dest = 'initialize', action = 'store_true')
parser.add_argument('--iteration',
        type = int, dest = 'iteration', default = 0)
parser.add_argument('--neighbours',
        type = int, dest = 'neighbours', default = 3)
parser.add_argument('--smoothness',
        type = int, dest = 'smoothness', default = 2)

def double(opt):
    old_simname = 'N{0:0>3x}'.format(opt.n)
    new_simname = 'N{0:0>3x}'.format(opt.n*2)
    c = fluid_resize(
            work_dir = opt.work_dir,
            simname = old_simname + '_double',
            dtype = opt.precision)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['dst_nx'] = 2*opt.n
    c.parameters['dst_ny'] = 2*opt.n
    c.parameters['dst_nz'] = 2*opt.n
    c.parameters['dst_simname'] = new_simname
    c.parameters['src_simname'] = old_simname
    c.parameters['niter_todo'] = 0
    c.write_src()
    c.set_host_info({'type' : 'pc'})
    c.write_par()
    c.run(ncpu = opt.ncpu,
          err_file = 'err_',
          out_file = 'out_')
    return None

def launch(
        opt,
        nu = None,
        dt = None,
        tracer_state_file = None,
        vorticity_field = None,
        code_class = bfps.NavierStokes):
    c = code_class(
            work_dir = opt.work_dir,
            fluid_precision = opt.precision)
    c.pars_from_namespace(opt)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    if type(nu) == type(None):
        c.parameters['nu'] = 5.5*opt.n**(-4./3)
    else:
        c.parameters['nu'] = nu
    if type(dt) == type(None):
        c.parameters['dt'] = .4 / opt.n
    else:
        c.parameters['dt'] = dt
    c.parameters['niter_out'] = c.parameters['niter_todo']
    c.parameters['niter_part'] = 1
    c.parameters['famplitude'] = 0.2
    if c.parameters['nparticles'] > 0:
        c.add_particles(kcut = 'fs->kM/2',
                        integration_steps = 1, neighbours = opt.neighbours, smoothness = opt.smoothness)
        c.add_particles(integration_steps = 1, neighbours = opt.neighbours, smoothness = opt.smoothness)
        c.add_particles(integration_steps = 2, neighbours = opt.neighbours, smoothness = opt.smoothness)
        c.add_particles(integration_steps = 3, neighbours = opt.neighbours, smoothness = opt.smoothness)
        c.add_particles(integration_steps = 4, neighbours = opt.neighbours, smoothness = opt.smoothness)
        c.add_particles(integration_steps = 5, neighbours = opt.neighbours, smoothness = opt.smoothness)
        c.add_particles(integration_steps = 6, neighbours = opt.neighbours, smoothness = opt.smoothness)
    c.fill_up_fluid_code()
    c.finalize_code()
    c.write_src()
    c.write_par()
    c.set_host_info({'type' : 'pc'})
    if opt.run:
        if opt.iteration == 0 and opt.initialize:
            if type(vorticity_field) == type(None):
                c.generate_vector_field(write_to_file = True, spectra_slope = 1.5)
            else:
                vorticity_field.tofile(
                        os.path.join(c.work_dir,
                                     c.simname + "_c{0}_i{1:0>5x}".format('vorticity',
                                                                          opt.iteration)))
        if opt.iteration == 0:
            for species in range(c.particle_species):
                if type(tracer_state_file) == type(None):
                    data = None
                else:
                    data = tracer_state_file['particles/tracers{0}/state'.format(species)][0]
                c.generate_tracer_state(
                        species = species,
                        write_to_file = False,
                        testing = True,
                        rseed = 3284,
                        data = data)
        c.run(ncpu = opt.ncpu,
              njobs = opt.njobs)
    return c

def acceleration_test(c):
    import numpy as np
    import matplotlib.pyplot as plt
    d = c.get_data_file()
    pos = d['particles/tracers4/state'].value
    vel = d['particles/tracers4/velocity'].value
    acc = d['particles/tracers4/acceleration'].value

    num_acc1 = (- vel[ :-2] + vel[2:])/(2*d['parameters/dt'].value*d['parameters/niter_part'].value)
    num_acc2 = (pos[ :-2] - 2*pos[1:-1] + pos[2:])/((d['parameters/dt'].value*d['parameters/niter_part'].value)**2)
    num_acc3 = (-vel[4:] + 8*vel[3:-1] - 8*vel[1:-3] + vel[:-4])/(12*d['parameters/dt'].value*d['parameters/niter_part'].value)
    num_acc4 = (-pos[4:] + 16*pos[3:-1] - 30*pos[2:-2] + 16*pos[1:-3] - pos[:-4])/(12*(d['parameters/dt'].value*d['parameters/niter_part'].value)**2)
    num_vel = (- pos[ :-2] + pos[2:])/(2*d['parameters/dt'].value*d['parameters/niter_part'].value)

    pid = np.unravel_index(np.argmax(np.abs(num_acc3 - acc[2:-2])), dims = num_acc3.shape)[1]
    fig = plt.figure()
    a = fig.add_subplot(111)
    col = ['red', 'green', 'blue']
    for cc in range(3):
        a.plot(num_acc1[2:, pid, cc], color = col[cc], dashes = (3, 4))
        a.plot(num_acc2[2:, pid, cc], color = col[cc], dashes = (2, 2))
        a.plot(num_acc3[1:, pid, cc], color = col[cc])
        a.plot(num_acc4[1:, pid, cc], color = col[cc], dashes = (3, 5))
        a.plot(acc[3:, pid, cc], color = col[cc], dashes = (1, 1))
    fig.savefig(os.path.join(c.work_dir, 'acc_test_{0}.pdf'.format(c.simname)))
    return None

if __name__ == '__main__':
    print('this file doesn\'t do anything')

