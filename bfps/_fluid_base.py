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



from ._code import _code
from bfps import tools

import os
import numpy as np
import h5py

class _fluid_particle_base(_code):
    """This class is meant to put together all common code between the
    different C++ solvers/postprocessing tools, so that development of
    specific functionalities is not overwhelming.
    """
    def __init__(
            self,
            name = 'solver',
            work_dir = './',
            simname = 'test',
            dtype = np.float32,
            use_fftw_wisdom = True):
        _code.__init__(
                self,
                work_dir = work_dir,
                simname = simname)
        self.use_fftw_wisdom = use_fftw_wisdom
        self.name = name
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
        self.fluid_includes = '#include "fluid_solver.hpp"\n'
        self.fluid_includes = '#include "field.hpp"\n'
        self.fluid_variables = ''
        self.fluid_definitions = ''
        self.fluid_start = ''
        self.fluid_loop = ''
        self.fluid_end  = ''
        self.fluid_output = ''
        self.stat_src = ''
        self.particle_includes = ''
        self.particle_variables = ''
        self.particle_definitions = ''
        self.particle_start = ''
        self.particle_loop = ''
        self.particle_end  = ''
        self.particle_stat_src = ''
        self.file_datasets_grow   = ''
        self.store_kspace = """
                //begincpp
                if (myrank == 0 && iteration == 0)
                {
                    hsize_t dims[4];
                    hid_t space, dset;
                    // store kspace information
                    hid_t parameter_file = stat_file;
                    //char fname[256];
                    //sprintf(fname, "%s.h5", simname);
                    //parameter_file = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
                    dset = H5Dopen(parameter_file, "/kspace/kshell", H5P_DEFAULT);
                    space = H5Dget_space(dset);
                    H5Sget_simple_extent_dims(space, dims, NULL);
                    H5Sclose(space);
                    if (fs->nshells != dims[0])
                    {
                        DEBUG_MSG(
                            "ERROR: computed nshells %d not equal to data file nshells %d\\n",
                            fs->nshells, dims[0]);
                    }
                    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, fs->kshell);
                    H5Dclose(dset);
                    dset = H5Dopen(parameter_file, "/kspace/nshell", H5P_DEFAULT);
                    H5Dwrite(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, fs->nshell);
                    H5Dclose(dset);
                    dset = H5Dopen(parameter_file, "/kspace/kM", H5P_DEFAULT);
                    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kMspec);
                    H5Dclose(dset);
                    dset = H5Dopen(parameter_file, "/kspace/dk", H5P_DEFAULT);
                    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->dk);
                    H5Dclose(dset);
                    //H5Fclose(parameter_file);
                }
                //endcpp
                """
        return None
    def get_data_file_name(self):
        return os.path.join(self.work_dir, self.simname + '.h5')
    def get_data_file(self):
        return h5py.File(self.get_data_file_name(), 'r')
    def get_particle_file_name(self):
        return os.path.join(self.work_dir, self.simname + '_particles.h5')
    def get_particle_file(self):
        return h5py.File(self.get_particle_file_name(), 'r')
    def finalize_code(
            self,
            postprocess_mode = False):
        self.includes   += self.fluid_includes
        self.includes   += '#include <ctime>\n'
        self.variables  += (self.fluid_variables +
                            'hid_t particle_file;\n')
        self.definitions += ('int grow_single_dataset(hid_t dset, int tincrement)\n{\n' +
                             'int ndims;\n' +
                             'hsize_t space;\n' +
                             'space = H5Dget_space(dset);\n' +
                             'ndims = H5Sget_simple_extent_ndims(space);\n' +
                             'hsize_t *dims = new hsize_t[ndims];\n' +
                             'H5Sget_simple_extent_dims(space, dims, NULL);\n' +
                             'dims[0] += tincrement;\n' +
                             'H5Dset_extent(dset, dims);\n' +
                             'H5Sclose(space);\n' +
                             'delete[] dims;\n' +
                             'return EXIT_SUCCESS;\n}\n')
        self.definitions+= self.fluid_definitions
        if self.particle_species > 0:
            self.includes    += self.particle_includes
            self.variables   += self.particle_variables
            self.definitions += self.particle_definitions
        self.definitions += ('herr_t grow_statistics_dataset(hid_t o_id, const char *name, const H5O_info_t *info, void *op_data)\n{\n' +
                             'if (info->type == H5O_TYPE_DATASET)\n{\n' +
                             'hsize_t dset = H5Dopen(o_id, name, H5P_DEFAULT);\n' +
                             'grow_single_dataset(dset, niter_todo/niter_stat);\n'
                             'H5Dclose(dset);\n}\n' +
                             'return 0;\n}\n')
        self.definitions += ('herr_t grow_particle_datasets(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data)\n{\n' +
                             'hsize_t dset;\n')
        for key in ['state', 'velocity', 'acceleration', 'velocity_gradient']:
            self.definitions += ('if (H5Lexists(g_id, "{0}", H5P_DEFAULT))\n'.format(key) +
                                 '{\n' +
                                 'dset = H5Dopen(g_id, "{0}", H5P_DEFAULT);\n'.format(key) +
                                 'grow_single_dataset(dset, niter_todo/niter_part);\n' +
                                 'H5Dclose(dset);\n}\n')
        self.definitions += ('if (H5Lexists(g_id, "rhs", H5P_DEFAULT))\n{\n' +
                             'dset = H5Dopen(g_id, "rhs", H5P_DEFAULT);\n' +
                             'grow_single_dataset(dset, 1);\n' +
                             'H5Dclose(dset);\n}\n' +
                             'return 0;\n}\n')
        self.definitions += ('int grow_file_datasets()\n{\n' +
                             'int file_problems = 0;\n' +
                             self.file_datasets_grow +
                             'return file_problems;\n'
                             '}\n')
        self.definitions += 'void do_stats()\n{\n' + self.stat_src + '}\n'
        self.definitions += 'void do_particle_stats()\n{\n' + self.particle_stat_src + '}\n'
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
        if self.particle_species > 0:
            self.main_start += """
                if (myrank == 0)
                {
                    // set caching parameters
                    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
                    herr_t cache_err = H5Pset_cache(fapl, 0, 521, 134217728, 1.0);
                    DEBUG_MSG("when setting cache for particles I got %d\\n", cache_err);
                    sprintf(fname, "%s_particles.h5", simname);
                    particle_file = H5Fopen(fname, H5F_ACC_RDWR, fapl);
                }
                """
            self.main_end = ('if (myrank == 0)\n' +
                             '{\n' +
                             'H5Fclose(particle_file);\n' +
                             '}\n') + self.main_end
        self.main        = """
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
                           //endcpp
                           """
        self.main       += self.fluid_start
        if self.particle_species > 0:
            self.main   += self.particle_start
        output_time_difference = ('time1 = clock();\n' +
                                  'local_time_difference = ((unsigned int)(time1 - time0))/((double)CLOCKS_PER_SEC);\n' +
                                  'time_difference = 0.0;\n' +
                                  'MPI_Allreduce(&local_time_difference, &time_difference, ' +
                                      '1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);\n' +
                                  'if (myrank == 0) std::cout << "iteration " ' +
                                      '<< {0} << " took " ' +
                                      '<< time_difference/nprocs << " seconds" << std::endl;\n' +
                                  'if (myrank == 0) std::cerr << "iteration " ' +
                                      '<< {0} << " took " ' +
                                      '<< time_difference/nprocs << " seconds" << std::endl;\n' +
                                  'time0 = time1;\n')
        if not postprocess_mode:
            self.main       += 'for (int max_iter = iteration+niter_todo; iteration < max_iter; iteration++)\n'
            self.main       += '{\n'
            self.main       += 'if (iteration % niter_stat == 0) do_stats();\n'
            if self.particle_species > 0:
                self.main       += 'if (iteration % niter_part == 0) do_particle_stats();\n'
                self.main   += self.particle_loop
            self.main       += self.fluid_loop
            self.main       += output_time_difference.format('iteration')
            self.main       += '}\n'
            self.main       += 'do_stats();\n'
            self.main       += 'do_particle_stats();\n'
            self.main       += output_time_difference.format('iteration')
        else:
            self.main       += 'for (int frame_index = iter0; frame_index <= iter1; frame_index += niter_out)\n'
            self.main       += '{\n'
            if self.particle_species > 0:
                self.main   += self.particle_loop
            self.main       += self.fluid_loop
            self.main       += output_time_difference.format('frame_index')
            self.main       += '}\n'
        if self.particle_species > 0:
            self.main   += self.particle_end
        self.main       += self.fluid_end
        return None
    def read_rfield(
            self,
            field = 'velocity',
            iteration = 0,
            filename = None):
        """
            :note: assumes field is a vector field
        """
        if type(filename) == type(None):
            filename = os.path.join(
                    self.work_dir,
                    self.simname + '_r' + field + '_i{0:0>5x}'.format(iteration))
        return np.memmap(
                filename,
                dtype = self.dtype,
                mode = 'r',
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
            write_to_file = False,
            # to switch to constant field, use generate_data_3D_uniform
            # for scalar_generator
            scalar_generator = tools.generate_data_3D):
        """generate vector field.

        The generated field is not divergence free, but it has the proper
        shape.

        :param rseed: seed for random number generator
        :param spectra_slope: spectrum of field will look like k^(-p)
        :param amplitude: all amplitudes are multiplied with this value
        :param iteration: the field is written at this iteration
        :param field_name: the name of the field being generated
        :param write_to_file: should we write the field to file?
        :param scalar_generator: which function to use for generating the
            individual components.
            Possible values: bfps.tools.generate_data_3D,
            bfps.tools.generate_data_3D_uniform
        :type rseed: int
        :type spectra_slope: float
        :type amplitude: float
        :type iteration: int
        :type field_name: str
        :type write_to_file: bool
        :type scalar_generator: function

        :returns: ``Kdata``, a complex valued 4D ``numpy.array`` that uses the
            transposed FFTW layout.
            Kdata[ky, kz, kx, i] is the amplitude of mode (kx, ky, kz) for
            the i-th component of the field.
            (i.e. x is the fastest index and z the slowest index in the
            real-space representation).
        """
        np.random.seed(rseed)
        Kdata00 = scalar_generator(
                self.parameters['nz']//2,
                self.parameters['ny']//2,
                self.parameters['nx']//2,
                p = spectra_slope,
                amplitude = amplitude).astype(self.ctype)
        Kdata01 = scalar_generator(
                self.parameters['nz']//2,
                self.parameters['ny']//2,
                self.parameters['nx']//2,
                p = spectra_slope,
                amplitude = amplitude).astype(self.ctype)
        Kdata02 = scalar_generator(
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
        Kdata1 = tools.padd_with_zeros(
                Kdata0,
                self.parameters['nz'],
                self.parameters['ny'],
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
        if testing:
            #data[0] = np.array([3.26434, 4.24418, 3.12157])
            data[0] = np.array([ 0.72086101,  2.59043666,  6.27501953])
        with h5py.File(self.get_particle_file_name(), 'r+') as data_file:
            data_file['tracers{0}/state'.format(species)][0] = data
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
            kMx = self.parameters['dkx']*(self.parameters['nx']//2 - 1)
            kMy = self.parameters['dky']*(self.parameters['ny']//2 - 1)
            kMz = self.parameters['dkz']*(self.parameters['nz']//2 - 1)
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
        _code.write_par(self, iter0 = iter0)
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r+') as ofile:
            ofile['bfps_info/exec_name'] = self.name
            ofile['field_dtype'] = np.dtype(self.dtype).str
            kspace = self.get_kspace()
            for k in kspace.keys():
                ofile['kspace/' + k] = kspace[k]
            nshells = kspace['nshell'].shape[0]
            ofile.close()
        return None
    def specific_parser_arguments(
            self,
            parser):
        _code.specific_parser_arguments(self, parser)
        return None

