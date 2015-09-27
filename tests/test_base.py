#! /usr/bin/env python2
########################################################################
#
#  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: Cristian.Lalescu@ds.mpg.de
#
########################################################################


import os
import subprocess
import argparse
import pickle

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import h5py

import bfps
from bfps import fluid_resize

parser = argparse.ArgumentParser()
parser.add_argument('--run', dest = 'run', action = 'store_true')
parser.add_argument('--initialize', dest = 'initialize', action = 'store_true')
parser.add_argument('-n',
        type = int, dest = 'n', default = 64)
parser.add_argument('--iteration',
        type = int, dest = 'iteration', default = 0)
parser.add_argument('--ncpu',
        type = int, dest = 'ncpu', default = 2)
parser.add_argument('--nsteps',
        type = int, dest = 'nsteps', default = 16)
parser.add_argument('--njobs',
        type = int, dest = 'njobs', default = 1)
parser.add_argument('--nparticles',
        type = int, dest = 'nparticles', default = 8)
parser.add_argument('--neighbours',
        type = int, dest = 'neighbours', default = 3)
parser.add_argument('--smoothness',
        type = int, dest = 'smoothness', default = 2)
parser.add_argument('--wd',
        type = str, dest = 'work_dir', default = 'data')
parser.add_argument('--precision',
        type = str, dest = 'precision', default = 'single')

def double(opt):
    old_simname = 'N{0:0>3x}'.format(opt.n)
    new_simname = 'N{0:0>3x}'.format(opt.n*2)
    c = fluid_resize(
            work_dir = opt.work_dir + '/resize',
            simname = old_simname,
            dtype = opt.precision)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['dst_nx'] = 2*opt.n
    c.parameters['dst_ny'] = 2*opt.n
    c.parameters['dst_nz'] = 2*opt.n
    c.parameters['dst_simname'] = new_simname
    c.write_src()
    c.set_host_info({'type' : 'pc'})
    if not os.path.isdir(os.path.join(opt.work_dir, 'resize')):
        os.mkdir(os.path.join(opt.work_dir, 'resize'))
    c.write_par()
    cp_command = ('cp {0}/test_cvorticity_i{1:0>5x} {2}/{3}_cvorticity_i{1:0>5x}'.format(
            opt.work_dir + '/' + old_simname, opt.iteration, opt.work_dir + '/resize', old_simname))
    subprocess.call([cp_command], shell = True)
    c.run(ncpu = opt.ncpu,
          err_file = 'err_' + old_simname + '_' + new_simname,
          out_file = 'out_' + old_simname + '_' + new_simname)
    if not os.path.isdir(os.path.join(opt.work_dir, new_simname)):
        os.mkdir(os.path.join(opt.work_dir, new_simname))
    cp_command = ('cp {2}/{3}_cvorticity_i{1:0>5x} {0}/test_cvorticity_i{1:0>5x}'.format(
            opt.work_dir + '/' + new_simname, 0, opt.work_dir + '/resize', new_simname))
    subprocess.call([cp_command], shell = True)
    return None

def launch(
        opt,
        nu = None,
        tracer_state_file = None,
        vorticity_field = None,
        code_class = bfps.NavierStokes):
    c = code_class(
            work_dir = opt.work_dir,
            fluid_precision = opt.precision)
    assert((opt.nsteps % 4) == 0)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    if type(nu) == type(None):
        c.parameters['nu'] = 5.5*opt.n**(-4./3)
    else:
        c.parameters['nu'] = nu
    c.parameters['dt'] = 5e-3 * (64. / opt.n)
    c.parameters['niter_todo'] = opt.nsteps
    c.parameters['niter_out'] = opt.nsteps
    c.parameters['niter_part'] = 1
    c.parameters['famplitude'] = 0.2
    c.parameters['nparticles'] = opt.nparticles
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
                c.generate_vector_field(write_to_file = True)
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

if __name__ == '__main__':
    print('this file doesn\'t do anything')

