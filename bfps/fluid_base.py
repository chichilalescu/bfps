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

import bfps
import bfps.code
import bfps.tools

import os
import numpy as np
import h5py

class fluid_particle_base(bfps.code):
    def __init__(
            self,
            name = 'solver',
            work_dir = './',
            simname = 'test',
            dtype = np.float32):
        super(fluid_particle_base, self).__init__(
                work_dir = work_dir,
                simname = simname)
        self.name = name
        self.particle_species = 0
        if dtype in [np.float32, np.float64]:
            self.dtype = dtype
        elif dtype in ['single', 'double']:
            if dtype == 'single':
                self.dtype = np.float32
            elif dtype == 'double':
                self.dtype = np.float64
        self.rtype = self.dtype
        if self.rtype == np.float32:
            self.ctype = np.complex64
            self.C_dtype = 'float'
        elif self.rtype == np.float64:
            self.ctype = np.complex128
            self.C_dtype = 'double'
        self.parameters['dkx'] = 1.0
        self.parameters['dky'] = 1.0
        self.parameters['dkz'] = 1.0
        self.parameters['niter_todo'] = 8
        self.parameters['niter_part'] = 1
        self.parameters['niter_out'] = 1024
        self.parameters['nparticles'] = 0
        self.parameters['dt'] = 0.01
        self.parameters['nu'] = 0.1
        self.parameters['famplitude'] = 1.0
        self.parameters['fmode'] = 1
        self.parameters['fk0'] = 0.0
        self.parameters['fk1'] = 3.0
        self.parameters['forcing_type'] = 'linear'
        self.fluid_includes = '#include "fluid_solver.hpp"\n'
        self.fluid_variables = ''
        self.fluid_definitions = ''
        self.fluid_start = ''
        self.fluid_loop = ''
        self.fluid_end  = ''
        self.particle_includes = '#include "tracers.hpp"\n'
        self.particle_variables = ''
        self.particle_definitions = ''
        self.particle_start = ''
        self.particle_loop = ''
        self.particle_end  = ''
        self.stat_src = ''
        self.file_datasets_create = ''
        self.file_datasets_grow   = ''
        return None
    def finalize_code(self):
        self.variables  += self.cdef_pars()
        self.definitions+= self.cread_pars()
        self.includes   += self.fluid_includes
        self.variables  += self.fluid_variables
        self.definitions+= self.fluid_definitions
        self.file_datasets_create = 'data_file.createGroup("/statistics");\n' + self.file_datasets_create
        if self.particle_species > 0:
            self.includes    += self.particle_includes
            self.variables   += self.particle_variables
            self.definitions += self.particle_definitions
        self.definitions += ('void create_file_datasets()\n{\n' +
                             self.file_datasets_create +
                             '}\n')
        self.definitions += ('void grow_file_datasets()\n{\n' +
                             self.file_datasets_grow +
                             '}\n')
        self.definitions += """
                //begincpp
                void init_stats()
                {
                    try
                    {
                        //H5::Exception::dontPrint();
                        data_file.openGroup("statistics");
                        grow_file_datasets();
                    }
                    catch (H5::FileIException)
                    {
                        DEBUG_MSG("Please ignore above messages about no group in the file. I will create it now.\\n");
                        create_file_datasets();
                    }
                }
                //endcpp
                """
        self.definitions += 'void do_stats()\n{\n' + self.stat_src + '}\n'
        self.main        = self.fluid_start
        if self.particle_species > 0:
            self.main   += self.particle_start
        self.main       += ('if (myrank == 0) init_stats();\n' +
                            'do_stats();\n')
        self.main       += 'for (int max_iter = iteration+niter_todo; iteration < max_iter; iteration++)\n{\n'
        self.main       += self.fluid_loop
        if self.particle_species > 0:
            self.main   += self.particle_loop
        self.main       += 'do_stats();\n}\n'
        if self.particle_species > 0:
            self.main   += self.particle_end
        self.main       += self.fluid_end
        return None
    def read_rfield(
            self,
            field = 'velocity',
            iteration = 0,
            filename = None):
        if type(filename) == type(None):
            filename = os.path.join(
                    self.work_dir,
                    self.simname + '_r' + field + '_i{0:0>5x}'.format(iteration))
        return np.memmap(
                filename,
                dtype = self.dtype,
                shape = (self.parameters['nz'],
                         self.parameters['ny'],
                         self.parameters['nx'], 3))
    def transpose_frame(
            self,
            field = 'velocity',
            iteration = 0,
            filename = None,
            ofile = None):
        Rdata = self.read_rfield(
                field = field,
                iteration = iteration,
                filename = filename)
        new_data = np.zeros(
                (3,
                 self.parameters['nz'],
                 self.parameters['ny'],
                 self.parameters['nx']),
                dtype = self.dtype)
        for i in range(3):
            new_data[i] = Rdata[..., i]
        if type(ofile) == type(None):
            ofile = os.path.join(
                    self.work_dir,
                    self.simname + '_r' + field + '_i{0:0>5x}_3xNZxNYxNX'.format(iteration))
        new_data.tofile(ofile)
        return new_data
    def plot_vel_cut(
            self,
            axis,
            field = 'velocity',
            iteration = 0,
            yval = 13,
            filename = None):
        axis.set_axis_off()
        Rdata0 = self.read_rfield(field = field, iteration = iteration, filename = filename)
        energy = np.sum(Rdata0[:, yval, :, :]**2, axis = 2)*.5
        axis.imshow(energy, interpolation='none')
        axis.set_title('{0}'.format(np.average(Rdata0[..., 0]**2 +
                                               Rdata0[..., 1]**2 +
                                               Rdata0[..., 2]**2)*.5))
        return Rdata0
    def generate_vdf(
            self,
            field = 'velocity',
            iteration = 0,
            filename = None):
        Rdata0 = self.read_rfield(field = field, iteration = iteration, filename = filename)
        subprocess.call(['vdfcreate',
                         '-dimension',
                         '{0}x{1}x{2}'.format(self.parameters['nz'],
                                              self.parameters['ny'],
                                              self.parameters['nx']),
                         '-numts', '1',
                         '-vars3d', '{0}x:{0}y:{0}z'.format(field),
                         filename + '.vdf'])
        for loop_data in [(0, 'x'), (1, 'y'), (2, 'z')]:
            Rdata0[..., loop_data[0]].tofile('tmprawfile')
            subprocess.call(['raw2vdf',
                             '-ts', '0',
                             '-varname', '{0}{1}'.format(field, loop_data[1]),
                             filename + '.vdf',
                             'tmprawfile'])
        return Rdata0
    def generate_vector_field(
            self,
            rseed = 7547,
            spectra_slope = 1.,
            precision = 'single',
            iteration = 0,
            field_name = 'vorticity',
            write_to_file = False):
        np.random.seed(rseed)
        Kdata00 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(self.ctype)
        Kdata01 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(self.ctype)
        Kdata02 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(self.ctype)
        Kdata0 = np.zeros(
                Kdata00.shape + (3,),
                Kdata00.dtype)
        Kdata0[..., 0] = Kdata00
        Kdata0[..., 1] = Kdata01
        Kdata0[..., 2] = Kdata02
        Kdata1 = bfps.tools.padd_with_zeros(
                Kdata0,
                self.parameters['ny'],
                self.parameters['nz'],
                self.parameters['nx'])
        if write_to_file:
            Kdata1.tofile(
                    os.path.join(self.work_dir,
                                 self.simname + "_c{0}_i{1:0>5x}".format(field_name, iteration)))
        return Kdata1
    def generate_tracer_state(
            self,
            rseed = None,
            iteration = 0,
            species = 0,
            write_to_file = False,
            ncomponents = 3,
            testing = False,
            data = None):
        if (type(data) == type(None)):
            if not type(rseed) == type(None):
                np.random.seed(rseed)
            #point with problems: 5.37632864e+00,   6.10414710e+00,   6.25256493e+00]
            data = np.zeros(self.parameters['nparticles']*ncomponents).reshape(-1, ncomponents)
            data[:, :3] = np.random.random((self.parameters['nparticles'], 3))*2*np.pi
        else:
            assert(data.shape == (self.parameters['nparticles'], ncomponents))
        if testing:
            data[0] = np.array([5.37632864e+00,   6.10414710e+00,   6.25256493e+00])
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r+') as data_file:
            dset = data_file.create_dataset(
                    '/particles/tracers{0}/state'.format(species),
                    (self.parameters['niter_todo']//self.parameters['niter_part'] + 1,
                     self.parameters['nparticles'],
                     ncomponents),
                    chunks = (1, self.parameters['nparticles'], ncomponents),
                    maxshape = (None, self.parameters['nparticles'], ncomponents))
            dset[0] = data
        if write_to_file:
            data.tofile(
                    os.path.join(
                        self.work_dir,
                        "tracers{0}_state_i{1:0>5x}".format(species, iteration)))
        return data
    def generate_initial_condition(self):
        self.generate_vector_field(write_to_file = True)
        for species in range(self.particle_species):
            self.generate_tracer_state(
                    species = species,
                    write_to_file = False)
        return None
    def write_par(self, iter0 = 0):
        super(fluid_particle_base, self).write_par(iter0 = iter0)
        ofile = h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r+')
        ofile['field_dtype'] = np.dtype(self.dtype).str
        ofile.close()
        return None

