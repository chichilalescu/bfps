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
import matplotlib.pyplot as plt

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
    def add_particles(
            self,
            neighbours = 1,
            smoothness = 1,
            integration_steps = 2,
            kcut = 'fs->kM'):
        self.parameters['neighbours{0}'.format(self.particle_species)] = neighbours
        self.parameters['smoothness{0}'.format(self.particle_species)] = smoothness
        self.parameters['kcut{0}'.format(self.particle_species)] = kcut
        self.parameters['integration_steps{0}'.format(self.particle_species)] = integration_steps
        self.particle_variables += 'tracers<{0}> *ps{1};\n'.format(self.C_dtype, self.particle_species)
        grow_template = """
                        //begincpp
                        temp_string = (std::string("/particles/") +
                                       std::string(ps{0}->name) +
                                       std::string("/{1}"));
                        Cdset = H5Dopen(stat_file, temp_string.c_str(), H5P_DEFAULT);
                        Cspace = H5Dget_space(Cdset);
                        ndims = H5Sget_simple_extent_dims(Cspace, dims, NULL);
                        dims[0] += niter_todo/niter_part;
                        H5Dset_extent(Cdset, dims);
                        H5Sclose(Cspace);
                        H5Dclose(Cdset);
                        //endcpp
                        """
        self.file_datasets_grow += grow_template.format(self.particle_species, 'state')
        self.file_datasets_grow += grow_template.format(self.particle_species, 'rhs')
        self.file_datasets_grow += grow_template.format(self.particle_species, 'velocity')
        self.file_datasets_grow += grow_template.format(self.particle_species, 'acceleration')
        #self.particle_definitions
        if kcut == 'fs->kM':
            if self.particle_species == 0:
                update_field = ('fs->compute_velocity(fs->cvorticity);\n' +
                                'fs->ift_velocity();\n')
            else:
                update_field = ''
        else:
            update_field = ('fs->compute_velocity(fs->cvorticity);\n' +
                            'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut) +
                            'fs->ift_velocity();\n')
        update_field += 'ps{0}->update_field();\n'.format(self.particle_species)
        if self.dtype == np.float32:
            FFTW = 'fftwf'
        elif self.dtype == np.float64:
            FFTW = 'fftw'
        compute_acc = ('{0} *acc_field = {1}_alloc_real(ps{2}->buffered_field_descriptor->local_size);\n' +
                       '{0} *acc_field_tmp = {1}_alloc_real(fs->rd->local_size);\n' +
                       'fs->compute_Lagrangian_acceleration(acc_field_tmp);\n' +
                       'ps{2}->rFFTW_to_buffered(acc_field_tmp, acc_field);\n' +
                       'ps{2}->sample_vec_field(acc_field, acceleration);\n' +
                       '{1}_free(acc_field_tmp);\n' +
                       '{1}_free(acc_field);\n').format(self.C_dtype, FFTW, self.particle_species)
        output_vel_acc =  """
                          //begincpp
                          {{
                              double *acceleration = new double[ps{0}->array_size];
                              double *velocity     = new double[ps{0}->array_size];
                              {1}
                              ps{0}->sample_vec_field(ps{0}->data, velocity);
                              {2}
                              if (ps{0}->fs->rd->myrank == 0)
                              {{
                                  //VELOCITY begin
                                  std::string temp_string = (std::string("/particles/") +
                                                             std::string(ps{0}->name) +
                                                             std::string("/velocity"));
                                  hid_t Cdset = H5Dopen(stat_file, temp_string.c_str(), H5P_DEFAULT);
                                  hid_t mspace, wspace;
                                  int ndims;
                                  hsize_t count[3], offset[3];
                                  wspace = H5Dget_space(Cdset);
                                  ndims = H5Sget_simple_extent_dims(wspace, count, NULL);
                                  count[0] = 1;
                                  offset[0] = ps{0}->iteration / ps{0}->traj_skip;
                                  offset[1] = 0;
                                  offset[2] = 0;
                                  mspace = H5Screate_simple(ndims, count, NULL);
                                  H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
                                  H5Dwrite(Cdset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, velocity);
                                  H5Dclose(Cdset);
                                  //VELOCITY end
                                  //ACCELERATION begin
                                  temp_string = (std::string("/particles/") +
                                                 std::string(ps{0}->name) +
                                                 std::string("/acceleration"));
                                  Cdset = H5Dopen(stat_file, temp_string.c_str(), H5P_DEFAULT);
                                  H5Dwrite(Cdset, H5T_NATIVE_DOUBLE, mspace, wspace, H5P_DEFAULT, acceleration);
                                  H5Sclose(mspace);
                                  H5Sclose(wspace);
                                  H5Dclose(Cdset);
                                  //ACCELERATION end
                              }}
                              delete[] acceleration;
                              delete[] velocity;
                          }}
                          //endcpp
                          """.format(self.particle_species, update_field, compute_acc)
        self.particle_start += ('sprintf(fname, "tracers{1}");\n' +
                                'ps{1} = new tracers<{0}>(\n' +
                                    'fname, fs,\n' +
                                    'nparticles,\n' +
                                    'neighbours{1}, smoothness{1}, niter_part, integration_steps{1},\n' +
                                    'fs->ru);\n' +
                                'ps{1}->dt = dt;\n' +
                                'ps{1}->iteration = iteration;\n' +
                                update_field +
                                'ps{1}->read(stat_file);\n').format(self.C_dtype, self.particle_species)
        self.particle_loop += ((update_field +
                                'if (ps{0}->iteration % niter_part == 0)\n' +
                                'ps{0}->write(stat_file, false);\n').format(self.particle_species) +
                               output_vel_acc)
        self.particle_end += ('ps{0}->write(stat_file);\n' +
                              'delete ps{0};\n').format(self.particle_species)
        self.particle_species += 1
        return None

if __name__ == '__main__':
    opt = parser.parse_args()
    c = test_interpolation(work_dir = opt.work_dir + '/io')
    c.pars_from_namespace(opt)
    c.add_particles(
            integration_steps = 1,
            neighbours = 4,
            smoothness = 1)
    c.fill_up_fluid_code()
    c.finalize_code()
    c.write_src()
    c.write_par()
    assert((c.parameters['nparticles'] - 2) % c.parameters['nx'] == 0)
    pos = np.zeros((c.parameters['nparticles'], 3), dtype = np.float64)
    pos[:c.parameters['nparticles']/2, 0] = np.linspace(0, 2*np.pi, pos.shape[0]/2)
    pos[c.parameters['nparticles']/2:, 2] = np.linspace(0, 2*np.pi, pos.shape[0]/2)
    c.generate_vector_field(write_to_file = True, spectra_slope = 5./3)
    c.generate_tracer_state(
            species = 0,
            write_to_file = False,
            data = pos)
    c.set_host_info({'type' : 'pc'})
    if opt.run:
        c.run(ncpu = opt.ncpu)
    df = c.get_data_file()
    x = c.get_coord('x')
    fig = plt.figure()
    a = fig.add_subplot(111)
    a.plot(x, df['statistics/xlines/velocity'][1])
    a.plot(df['particles/tracers0/state'][0, :pos.shape[0]/2, 0],
           df['particles/tracers0/velocity'][0, :pos.shape[0]/2],
           dashes = (1, 1))
    fig.savefig('xinterp.pdf')
    fig = plt.figure()
    a = fig.add_subplot(111)
    a.plot(df['particles/tracers0/state'][0, pos.shape[0]/2:, 2],
           df['particles/tracers0/velocity'][0, pos.shape[0]/2:],
           dashes = (1, 1))
    fig.savefig('zinterp.pdf')

