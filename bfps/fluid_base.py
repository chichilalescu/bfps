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
            dtype = np.float32,
            use_fftw_wisdom = True):
        super(fluid_particle_base, self).__init__(
                work_dir = work_dir,
                simname = simname)
        self.use_fftw_wisdom = use_fftw_wisdom
        self.name = name + '_' + simname
        self.particle_species = 0
        if dtype in [np.float32, np.float64]:
            self.dtype = dtype
        elif dtype in ['single', 'double']:
            if dtype == 'single':
                self.dtype = np.dtype(np.float32)
            elif dtype == 'double':
                self.dtype = np.dtype(np.float64)
        self.rtype = self.dtype
        if self.rtype == np.float32:
            self.ctype = np.dtype(np.complex64)
            self.C_dtype = 'float'
        elif self.rtype == np.float64:
            self.ctype = np.dtype(np.complex128)
            self.C_dtype = 'double'
        self.parameters['dealias_type'] = 1
        self.parameters['dkx'] = 1.0
        self.parameters['dky'] = 1.0
        self.parameters['dkz'] = 1.0
        self.parameters['niter_todo'] = 8
        self.parameters['niter_part'] = 1
        self.parameters['niter_stat'] = 1
        self.parameters['niter_out'] = 1024
        self.parameters['nparticles'] = 0
        self.parameters['dt'] = 0.01
        self.parameters['nu'] = 0.1
        self.parameters['famplitude'] = 1.0
        self.parameters['fmode'] = 1
        self.parameters['fk0'] = 0.0
        self.parameters['fk1'] = 3.0
        self.parameters['forcing_type'] = 'linear'
        self.parameters['histogram_bins'] = 256
        self.parameters['max_velocity_estimate'] = 1.0
        self.parameters['max_vorticity_estimate'] = 1.0
        self.parameters['QR2D_histogram_bins'] = 64
        self.parameters['max_trS2_estimate'] = 1.0
        self.parameters['max_Q_estimate'] = 1.0
        self.parameters['max_R_estimate'] = 1.0
        self.fluid_includes = '#include "fluid_solver.hpp"\n'
        self.fluid_variables = ''
        self.fluid_definitions = ''
        self.fluid_start = ''
        self.fluid_loop = ''
        self.fluid_end  = ''
        self.fluid_output = ''
        self.particle_includes = ''
        self.particle_variables = ''
        self.particle_definitions = ''
        self.particle_start = ''
        self.particle_loop = ''
        self.particle_end  = ''
        self.stat_src = ''
        self.file_datasets_grow   = ''
        return None
    def finalize_code(self):
        self.includes   += self.fluid_includes
        self.includes   += '#include <ctime>\n'
        self.variables  += self.fluid_variables
        self.definitions+= self.fluid_definitions
        if self.particle_species > 0:
            self.includes    += self.particle_includes
            self.variables   += self.particle_variables
            self.definitions += self.particle_definitions
        self.definitions += ('int grow_single_dataset(hid_t dset, int tincrement)\n{\n' +
                             'int ndims;\n' +
                             'hsize_t dims[5];\n' +
                             'hsize_t space;\n' +
                             'space = H5Dget_space(dset);\n' +
                             'ndims = H5Sget_simple_extent_dims(space, dims, NULL);\n' +
                             'dims[0] += tincrement;\n' +
                             'H5Dset_extent(dset, dims);\n' +
                             'H5Sclose(space);\n' +
                             'return EXIT_SUCCESS;\n}\n')
        self.definitions += ('herr_t grow_statistics_dataset(hid_t o_id, const char *name, const H5O_info_t *info, void *op_data)\n{\n' +
                             'if (info->type == H5O_TYPE_DATASET)\n{\n' +
                             'hsize_t dset = H5Dopen(o_id, name, H5P_DEFAULT);\n' +
                             'grow_single_dataset(dset, niter_todo/niter_stat);\n'
                             'H5Dclose(dset);\n}\n' +
                             'return 0;\n}\n')
        self.definitions += ('herr_t grow_particle_datasets(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data)\n{\n' +
                             'std::string full_name;\n' +
                             'hsize_t dset;\n')
        for key in ['state', 'velocity', 'acceleration']:
            self.definitions += ('full_name = (std::string(name) + std::string("/{0}"));\n'.format(key) +
                                 'dset = H5Dopen(g_id, full_name.c_str(), H5P_DEFAULT);\n' +
                                 'grow_single_dataset(dset, niter_todo/niter_part);\n' +
                                 'H5Dclose(dset);\n')
        self.definitions += ('full_name = (std::string(name) + std::string("/rhs"));\n' +
                             'if (H5Lexists(g_id, full_name.c_str(), H5P_DEFAULT))\n{\n' +
                             'dset = H5Dopen(g_id, full_name.c_str(), H5P_DEFAULT);\n' +
                             'grow_single_dataset(dset, 1);\n' +
                             'H5Dclose(dset);\n}\n' +
                             'return 0;\n}\n')
        self.definitions += ('int grow_file_datasets()\n{\n' +
                             'int file_problems = 0;\n' +
                             self.file_datasets_grow +
                             'return file_problems;\n'
                             '}\n')
        self.definitions += 'void do_stats()\n{\n' + self.stat_src + '}\n'
        # take care of wisdom
        if self.use_fftw_wisdom:
            if self.dtype == np.float32:
                fftw_prefix = 'fftwf_'
            elif self.dtype == np.float64:
                fftw_prefix = 'fftw_'
            self.main_start += """
                        //begincpp
                        if (myrank == 0)
                        {{
                            char fname[256];
                            sprintf(fname, "%s_fftw_wisdom.txt", simname);
                            {0}import_wisdom_from_filename(fname);
                        }}
                        {0}mpi_broadcast_wisdom(MPI_COMM_WORLD);
                        //endcpp
                        """.format(fftw_prefix)
            self.main_end = """
                        //begincpp
                        {0}mpi_gather_wisdom(MPI_COMM_WORLD);
                        MPI_Barrier(MPI_COMM_WORLD);
                        if (myrank == 0)
                        {{
                            char fname[256];
                            sprintf(fname, "%s_fftw_wisdom.txt", simname);
                            {0}export_wisdom_to_filename(fname);
                        }}
                        //endcpp
                        """.format(fftw_prefix) + self.main_end
        self.main        = self.fluid_start
        if self.particle_species > 0:
            self.main   += self.particle_start
        self.main       += """
                           //begincpp
                           int data_file_problem;
                           clock_t time0, time1;
                           double time_difference, local_time_difference;
                           time0 = clock();
                           if (myrank == 0) data_file_problem = grow_file_datasets();
                           MPI_Bcast(&data_file_problem, 1, MPI_INT, 0, MPI_COMM_WORLD);
                           if (data_file_problem > 0)
                           {
                               std::cerr << data_file_problem << " problems growing file datasets.\\ntrying to exit now." << std::endl;
                               MPI_Finalize();
                               return EXIT_SUCCESS;
                           }
                           do_stats();
                           //endcpp
                           """
        output_time_difference = ('time1 = clock();\n' +
                                  'local_time_difference = ((unsigned int)(time1 - time0))/((double)CLOCKS_PER_SEC);\n' +
                                  'time_difference = 0.0;\n' +
                                  'MPI_Allreduce(&local_time_difference, &time_difference, ' +
                                      '1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);\n' +
                                  'if (myrank == 0) std::cout << "iteration " ' +
                                      '<< iteration << " took " ' +
                                      '<< time_difference/nprocs << " seconds" << std::endl;\n' +
                                  'time0 = time1;\n')
        self.main       += 'for (int max_iter = iteration+niter_todo; iteration < max_iter; iteration++)\n{\n'
        self.main       += output_time_difference
        self.main       += self.fluid_loop
        if self.particle_species > 0:
            self.main   += self.particle_loop
        self.main       += 'if (iteration % niter_stat == 0) do_stats();\n}\n'
        self.main       += output_time_difference
        if self.particle_species > 0:
            self.main   += self.particle_end
        self.main       += 'do_stats();\n'
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
        else:
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
    def generate_vector_field(
            self,
            rseed = 7547,
            spectra_slope = 1.,
            amplitude = 1.,
            iteration = 0,
            field_name = 'vorticity',
            write_to_file = False):
        np.random.seed(rseed)
        Kdata00 = bfps.tools.generate_data_3D(
                self.parameters['nz']//2,
                self.parameters['ny']//2,
                self.parameters['nx']//2,
                p = spectra_slope,
                amplitude = amplitude).astype(self.ctype)
        Kdata01 = bfps.tools.generate_data_3D(
                self.parameters['nz']//2,
                self.parameters['ny']//2,
                self.parameters['nx']//2,
                p = spectra_slope,
                amplitude = amplitude).astype(self.ctype)
        Kdata02 = bfps.tools.generate_data_3D(
                self.parameters['nz']//2,
                self.parameters['ny']//2,
                self.parameters['nx']//2,
                p = spectra_slope,
                amplitude = amplitude).astype(self.ctype)
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
            #data[0] = np.array([3.26434, 4.24418, 3.12157])
            data[0] = np.array([ 0.72086101,  2.59043666,  6.27501953])
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r+') as data_file:
            time_chunk = 2**20 // (8*ncomponents*
                                   self.parameters['nparticles'])
            time_chunk = max(time_chunk, 1)
            dset = data_file.create_dataset(
                    '/particles/tracers{0}/state'.format(species),
                    (1,
                     self.parameters['nparticles'],
                     ncomponents),
                    chunks = (time_chunk, self.parameters['nparticles'], ncomponents),
                    maxshape = (None, self.parameters['nparticles'], ncomponents),
                    dtype = np.float64)
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
    def get_kspace(self):
        kspace = {}
        if self.parameters['dealias_type'] == 1:
            kMx = self.parameters['dkx']*(2*self.parameters['nx']//5)
            kMy = self.parameters['dky']*(2*self.parameters['ny']//5)
            kMz = self.parameters['dkz']*(2*self.parameters['nz']//5)
        else:
            kMx = self.parameters['dkx']*(self.parameters['nx']//3 - 1)
            kMy = self.parameters['dky']*(self.parameters['ny']//3 - 1)
            kMz = self.parameters['dkz']*(self.parameters['nz']//3 - 1)
        kspace['kM'] = max(kMx, kMy, kMz)
        kspace['dk'] = min(self.parameters['dkx'],
                           self.parameters['dky'],
                           self.parameters['dkz'])
        nshells = int(kspace['kM'] / kspace['dk']) + 2
        kspace['nshell'] = np.zeros(nshells, dtype = np.int64)
        kspace['kshell'] = np.zeros(nshells, dtype = np.float64)
        kspace['kx'] = np.arange( 0,
                                  self.parameters['nx']//2 + 1).astype(np.float64)*self.parameters['dkx']
        kspace['ky'] = np.arange(-self.parameters['ny']//2 + 1,
                                  self.parameters['ny']//2 + 1).astype(np.float64)*self.parameters['dky']
        kspace['ky'] = np.roll(kspace['ky'], self.parameters['ny']//2+1)
        kspace['kz'] = np.arange(-self.parameters['nz']//2 + 1,
                                  self.parameters['nz']//2 + 1).astype(np.float64)*self.parameters['dkz']
        kspace['kz'] = np.roll(kspace['kz'], self.parameters['nz']//2+1)
        return kspace
    def write_par(self, iter0 = 0):
        assert (self.parameters['niter_todo'] % self.parameters['niter_stat'] == 0)
        assert (self.parameters['niter_todo'] % self.parameters['niter_out']  == 0)
        assert (self.parameters['niter_todo'] % self.parameters['niter_part'] == 0)
        assert (self.parameters['niter_out']  % self.parameters['niter_stat'] == 0)
        assert (self.parameters['niter_out']  % self.parameters['niter_part'] == 0)
        super(fluid_particle_base, self).write_par(iter0 = iter0)
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r+') as ofile:
            ofile['field_dtype'] = np.dtype(self.dtype).str
            kspace = self.get_kspace()
            for k in kspace.keys():
                ofile['kspace/' + k] = kspace[k]
            nshells = kspace['nshell'].shape[0]
            for k in ['velocity', 'vorticity']:
                time_chunk = 2**20//(8*3*self.parameters['nx'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/xlines/' + k,
                                     (1, self.parameters['nx'], 3),
                                     chunks = (time_chunk, self.parameters['nx'], 3),
                                     maxshape = (None, self.parameters['nx'], 3),
                                     dtype = self.dtype,
                                     compression = 'gzip')
                time_chunk = 2**20//(8*3*3*nshells)
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/spectra/' + k + '_' + k,
                                     (1, nshells, 3, 3),
                                     chunks = (time_chunk, nshells, 3, 3),
                                     maxshape = (None, nshells, 3, 3),
                                     dtype = np.float64,
                                     compression = 'gzip')
                time_chunk = 2**20//(8*4*10)
                time_chunk = max(time_chunk, 1)
                a = ofile.create_dataset('statistics/moments/' + k,
                                     (1, 10, 4),
                                     chunks = (time_chunk, 10, 4),
                                     maxshape = (None, 10, 4),
                                     dtype = np.float64,
                                     compression = 'gzip')
                time_chunk = 2**20//(8*4*self.parameters['histogram_bins'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/' + k,
                                     (1,
                                      self.parameters['histogram_bins'],
                                      4),
                                     chunks = (time_chunk,
                                               self.parameters['histogram_bins'],
                                               4),
                                     maxshape = (None,
                                                 self.parameters['histogram_bins'],
                                                 4),
                                     dtype = np.int64,
                                     compression = 'gzip')
            for s in range(self.particle_species):
                if self.parameters['tracers{0}_integration_method'.format(s)] == 'AdamsBashforth':
                    time_chunk = 2**20 // (8*3*
                                           self.parameters['nparticles']*
                                           self.parameters['tracers{0}_integration_steps'.format(s)])
                    time_chunk = max(time_chunk, 1)
                    ofile.create_dataset('particles/tracers{0}/rhs'.format(s),
                                         (1,
                                          self.parameters['tracers{0}_integration_steps'.format(s)],
                                          self.parameters['nparticles'],
                                          3),
                                         maxshape = (None,
                                                     self.parameters['tracers{0}_integration_steps'.format(s)],
                                                     self.parameters['nparticles'],
                                                     3),
                                         chunks =  (time_chunk,
                                                    self.parameters['tracers{0}_integration_steps'.format(s)],
                                                    self.parameters['nparticles'],
                                                    3),
                                         dtype = np.float64)
                time_chunk = 2**20 // (8*3*self.parameters['nparticles'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset(
                    '/particles/tracers{0}/velocity'.format(s),
                    (1,
                     self.parameters['nparticles'],
                     3),
                    chunks = (time_chunk, self.parameters['nparticles'], 3),
                    maxshape = (None, self.parameters['nparticles'], 3),
                    dtype = np.float64)
                ofile.create_dataset(
                    '/particles/tracers{0}/acceleration'.format(s),
                    (1,
                     self.parameters['nparticles'],
                     3),
                    chunks = (time_chunk, self.parameters['nparticles'], 3),
                    maxshape = (None, self.parameters['nparticles'], 3),
                    dtype = np.float64)
            ofile.close()
        return None

