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
import numpy as np
import h5py

import bfps
import bfps.fluid_base
import bfps.tools

class NavierStokes(bfps.fluid_base.fluid_particle_base):
    def __init__(
            self,
            name = 'NavierStokes',
            work_dir = './',
            simname = 'test',
            fluid_precision = 'single',
            fftw_plan_rigor = 'FFTW_MEASURE',
            frozen_fields = False,
            use_fftw_wisdom = True,
            QR_stats_on = False):
        self.QR_stats_on = QR_stats_on
        self.frozen_fields = frozen_fields
        self.fftw_plan_rigor = fftw_plan_rigor
        super(NavierStokes, self).__init__(
                name = name,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision,
                use_fftw_wisdom = use_fftw_wisdom)
        self.file_datasets_grow = """
                //begincpp
                std::string temp_string;
                hsize_t dims[4];
                hid_t group;
                hid_t Cspace, Cdset;
                int ndims;
                // store kspace information
                Cdset = H5Dopen(stat_file, "/kspace/kshell", H5P_DEFAULT);
                Cspace = H5Dget_space(Cdset);
                H5Sget_simple_extent_dims(Cspace, dims, NULL);
                H5Sclose(Cspace);
                if (fs->nshells != dims[0])
                {
                    DEBUG_MSG(
                        "ERROR: computed nshells %d not equal to data file nshells %d\\n",
                        fs->nshells, dims[0]);
                    file_problems++;
                }
                H5Dwrite(Cdset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, fs->kshell);
                H5Dclose(Cdset);
                Cdset = H5Dopen(stat_file, "/kspace/nshell", H5P_DEFAULT);
                H5Dwrite(Cdset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, fs->nshell);
                H5Dclose(Cdset);
                Cdset = H5Dopen(stat_file, "/kspace/kM", H5P_DEFAULT);
                H5Dwrite(Cdset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kM);
                H5Dclose(Cdset);
                Cdset = H5Dopen(stat_file, "/kspace/dk", H5P_DEFAULT);
                H5Dwrite(Cdset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->dk);
                H5Dclose(Cdset);
                group = H5Gopen(stat_file, "/statistics", H5P_DEFAULT);
                H5Ovisit(group, H5_INDEX_NAME, H5_ITER_NATIVE, grow_statistics_dataset, NULL);
                H5Gclose(group);
                //endcpp
                """
        self.style = {}
        self.statistics = {}
        self.fluid_output = 'fs->write(\'v\', \'c\');\n'
        return None
    def create_stat_output(
            self,
            dset_name,
            data_buffer,
            data_type = 'H5T_NATIVE_DOUBLE',
            size_setup = None,
            close_spaces = True):
        new_stat_output_txt = 'Cdset = H5Dopen(stat_file, "{0}", H5P_DEFAULT);\n'.format(dset_name)
        if not type(size_setup) == type(None):
            new_stat_output_txt += (
                    size_setup +
                    'wspace = H5Dget_space(Cdset);\n' +
                    'ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);\n' +
                    'mspace = H5Screate_simple(ndims, count, NULL);\n' +
                    'H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);\n')
        new_stat_output_txt += ('H5Dwrite(Cdset, {0}, mspace, wspace, H5P_DEFAULT, {1});\n' +
                                'H5Dclose(Cdset);\n').format(data_type, data_buffer)
        if close_spaces:
            new_stat_output_txt += ('H5Sclose(mspace);\n' +
                                    'H5Sclose(wspace);\n')
        return new_stat_output_txt
    def write_fluid_stats(self):
        self.fluid_includes += '#include <cmath>\n'
        self.fluid_includes += '#include "fftw_tools.hpp"\n'
        self.stat_src += """
                //begincpp
                //endcpp
                """
        self.stat_src += """
                //begincpp
                    double *velocity_moments  = new double[10*4];
                    double *vorticity_moments = new double[10*4];
                    ptrdiff_t *hist_velocity  = new ptrdiff_t[histogram_bins*4];
                    ptrdiff_t *hist_vorticity = new ptrdiff_t[histogram_bins*4];
                    double max_estimates[4];
                    fs->compute_velocity(fs->cvorticity);
                    double *spec_velocity  = new double[fs->nshells*9];
                    double *spec_vorticity = new double[fs->nshells*9];
                    fs->cospectrum(fs->cvelocity, fs->cvelocity, spec_velocity);
                    fs->cospectrum(fs->cvorticity, fs->cvorticity, spec_vorticity);
                    //endcpp
                    """
        if self.QR_stats_on:
            self.stat_src += """
                //begincpp
                    double *trS2_Q_R_moments  = new double[10*3];
                    ptrdiff_t *hist_trS2_Q_R  = new ptrdiff_t[histogram_bins*3];
                    ptrdiff_t *hist_QR2D      = new ptrdiff_t[QR2D_histogram_bins*QR2D_histogram_bins];
                    max_estimates[0] = max_trS2_estimate;
                    max_estimates[1] = max_Q_estimate;
                    max_estimates[2] = max_R_estimate;
                    fs->compute_gradient_statistics(
                        fs->cvelocity,
                        trS2_Q_R_moments,
                        hist_trS2_Q_R,
                        hist_QR2D,
                        max_estimates,
                        histogram_bins,
                        QR2D_histogram_bins);
                    //endcpp
                    """
        self.stat_src += """
                //begincpp
                    fs->ift_velocity();
                    max_estimates[0] = max_velocity_estimate/sqrt(3);
                    max_estimates[1] = max_estimates[0];
                    max_estimates[2] = max_estimates[0];
                    max_estimates[3] = max_velocity_estimate;
                    fs->compute_rspace_stats(fs->rvelocity,
                                             velocity_moments,
                                             hist_velocity,
                                             max_estimates,
                                             histogram_bins);
                    fs->ift_vorticity();
                    max_estimates[0] = max_vorticity_estimate/sqrt(3);
                    max_estimates[1] = max_estimates[0];
                    max_estimates[2] = max_estimates[0];
                    max_estimates[3] = max_vorticity_estimate;
                    fs->compute_rspace_stats(fs->rvorticity,
                                             vorticity_moments,
                                             hist_vorticity,
                                             max_estimates,
                                             histogram_bins);
                    if (fs->cd->myrank == 0)
                    {{
                        hid_t Cdset, wspace, mspace;
                        int ndims;
                        hsize_t count[4], offset[4], dims[4];
                        offset[0] = fs->iteration/niter_stat;
                        offset[1] = 0;
                        offset[2] = 0;
                        offset[3] = 0;
                //endcpp
                """.format(self.C_dtype)
        if self.dtype == np.float32:
            field_H5T = 'H5T_NATIVE_FLOAT'
        elif self.dtype == np.float64:
            field_H5T = 'H5T_NATIVE_DOUBLE'
        self.stat_src += self.create_stat_output(
                '/statistics/xlines/velocity',
                'fs->rvelocity',
                data_type = field_H5T,
                size_setup = """
                    count[0] = 1;
                    count[1] = nx;
                    count[2] = 3;
                    """,
                close_spaces = False)
        self.stat_src += self.create_stat_output(
                '/statistics/xlines/vorticity',
                'fs->rvorticity',
                data_type = field_H5T)
        self.stat_src += self.create_stat_output(
                '/statistics/moments/velocity',
                'velocity_moments',
                size_setup = """
                    count[0] = 1;
                    count[1] = 10;
                    count[2] = 4;
                    """,
                close_spaces = False)
        self.stat_src += self.create_stat_output(
                '/statistics/moments/vorticity',
                'vorticity_moments')
        self.stat_src += self.create_stat_output(
                '/statistics/spectra/velocity_velocity',
                'spec_velocity',
                size_setup = """
                    count[0] = 1;
                    count[1] = fs->nshells;
                    count[2] = 3;
                    count[3] = 3;
                    """,
                close_spaces = False)
        self.stat_src += self.create_stat_output(
                '/statistics/spectra/vorticity_vorticity',
                'spec_vorticity')
        self.stat_src += self.create_stat_output(
                '/statistics/histograms/velocity',
                'hist_velocity',
                data_type = 'H5T_NATIVE_INT64',
                size_setup = """
                    count[0] = 1;
                    count[1] = histogram_bins;
                    count[2] = 4;
                    """,
                close_spaces = False)
        self.stat_src += self.create_stat_output(
                '/statistics/histograms/vorticity',
                'hist_vorticity',
                data_type = 'H5T_NATIVE_INT64')
        if self.QR_stats_on:
            self.stat_src += self.create_stat_output(
                    '/statistics/moments/trS2_Q_R',
                    'trS2_Q_R_moments',
                    size_setup ="""
                        count[0] = 1;
                        count[1] = 10;
                        count[2] = 3;
                        """)
            self.stat_src += self.create_stat_output(
                    '/statistics/histograms/trS2_Q_R',
                    'hist_trS2_Q_R',
                    data_type = 'H5T_NATIVE_INT64',
                    size_setup = """
                        count[0] = 1;
                        count[1] = histogram_bins;
                        count[2] = 3;
                        """)
            self.stat_src += self.create_stat_output(
                    '/statistics/histograms/QR2D',
                    'hist_QR2D',
                    data_type = 'H5T_NATIVE_INT64',
                    size_setup = """
                        count[0] = 1;
                        count[1] = QR2D_histogram_bins;
                        count[2] = QR2D_histogram_bins;
                        """)
        self.stat_src += """
                //begincpp
                    }
                    delete[] spec_velocity;
                    delete[] spec_vorticity;
                    delete[] velocity_moments;
                    delete[] vorticity_moments;
                    delete[] hist_velocity;
                    delete[] hist_vorticity;
                //endcpp
                """
        if self.QR_stats_on:
            self.stat_src += """
                //begincpp
                    delete[] trS2_Q_R_moments;
                    delete[] hist_trS2_Q_R;
                    delete[] hist_QR2D;
                //endcpp
                """
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += ('fluid_solver<{0}> *fs;\n'.format(self.C_dtype) +
                                 'int *kindices;\n' +
                                 'hid_t H5T_field_complex;\n')
        self.fluid_definitions += """
                    typedef struct {{
                        {0} re;
                        {0} im;
                    }} tmp_complex_type;
                    """.format(self.C_dtype)
        self.write_fluid_stats()
        if self.dtype == np.float32:
            field_H5T = 'H5T_NATIVE_FLOAT'
        elif self.dtype == np.float64:
            field_H5T = 'H5T_NATIVE_DOUBLE'
        self.fluid_start += """
                //begincpp
                char fname[512];
                fs = new fluid_solver<{0}>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz,
                        dealias_type,
                        {1});
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->fk0 = fk0;
                fs->fk1 = fk1;
                strncpy(fs->forcing_type, forcing_type, 128);
                fs->iteration = iteration;
                fs->read('v', 'c');
                if (fs->cd->myrank == 0)
                {{
                    H5T_field_complex = H5Tcreate(H5T_COMPOUND, sizeof(tmp_complex_type));
                    H5Tinsert(H5T_field_complex, "r", HOFFSET(tmp_complex_type, re), {2});
                    H5Tinsert(H5T_field_complex, "i", HOFFSET(tmp_complex_type, im), {2});
                }}
                //endcpp
                """.format(self.C_dtype, self.fftw_plan_rigor, field_H5T)
        if not self.frozen_fields:
            self.fluid_loop = 'fs->step(dt);\n'
        else:
            self.fluid_loop = ''
        self.fluid_loop += ('if (fs->iteration % niter_out == 0)\n{\n' +
                            self.fluid_output + '\n}\n')
        self.fluid_end = ('if (fs->iteration % niter_out != 0)\n{\n' +
                          self.fluid_output + '\n}\n' +
                          'if (fs->cd->myrank == 0)\n' +
                          '{\n' +
                          'delete[] kindices;\n' +
                          'H5Tclose(H5T_field_complex);\n' +
                          '}\n' +
                          'delete fs;\n')
        return None
    def add_particle_fields(
            self,
            interp_type = 'spline',
            kcut = None,
            neighbours = 1,
            smoothness = 1,
            name = 'particle_field',
            field_class = 'rFFTW_interpolator'):
        self.fluid_includes += '#include "{0}.hpp"\n'.format(field_class)
        self.fluid_variables += field_class + '<{0}, {1}> *vel_{2}, *acc_{2};\n'.format(self.C_dtype, neighbours, name)
        self.parameters[name + '_type'] = interp_type
        self.parameters[name + '_neighbours'] = neighbours
        if interp_type == 'spline':
            self.parameters[name + '_smoothness'] = smoothness
            beta_name = 'beta_n{0}_m{1}'.format(neighbours, smoothness)
        elif interp_type == 'Lagrange':
            beta_name = 'beta_Lagrange_n{0}'.format(neighbours)
        self.fluid_start += ('vel_{0} = new {1}<{2}, {3}>(fs, {4});\n' +
                             'acc_{0} = new {1}<{2}, {3}>(fs, {4});\n').format(name,
                                                                               field_class,
                                                                               self.C_dtype,
                                                                               neighbours,
                                                                               beta_name)
        self.fluid_end += ('delete vel_{0};\n' +
                           'delete acc_{0};\n').format(name)
        update_fields = 'fs->compute_velocity(fs->cvorticity);\n'
        if not type(kcut) == type(None):
            update_fields += 'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut)
        update_fields += ('fs->ift_velocity();\n' +
                          'vel_{0}->read_rFFTW(fs->rvelocity);\n' +
                          'fs->compute_Lagrangian_acceleration(acc_{0}->temp);\n' +
                          'acc_{0}->read_rFFTW(acc_{0}->temp);\n').format(name)
        self.fluid_start += update_fields
        self.fluid_loop += update_fields
        return None
    def add_particles(
            self,
            integration_method = 'AdamsBashforth',
            integration_steps = 2,
            kcut = 'fs->kM',
            frozen_particles = False,
            fields_name = None,
            particle_class = 'rFFTW_particles'):
        if integration_method == 'cRK4':
            integration_steps = 4
        elif integration_method == 'Heun':
            integration_steps = 2
        neighbours = self.parameters[fields_name + '_neighbours']
        self.parameters['tracers{0}_field'.format(self.particle_species)] = fields_name
        self.parameters['tracers{0}_integration_method'.format(self.particle_species)] = integration_method
        self.parameters['tracers{0}_kcut'.format(self.particle_species)] = kcut
        self.parameters['tracers{0}_integration_steps'.format(self.particle_species)] = integration_steps
        self.file_datasets_grow += """
                        //begincpp
                        temp_string = (std::string("/particles/") +
                                       std::string(ps{0}->name));
                        group = H5Gopen(stat_file, temp_string.c_str(), H5P_DEFAULT);
                        grow_particle_datasets(group, temp_string.c_str(), NULL, NULL);
                        H5Gclose(group);
                        //endcpp
                        """.format(self.particle_species)
        if self.dtype == np.float32:
            FFTW = 'fftwf'
        elif self.dtype == np.float64:
            FFTW = 'fftw'
        output_vel_acc =  """
                          //begincpp
                          {{
                              double *acceleration = new double[ps{0}->array_size];
                              double *velocity     = new double[ps{0}->array_size];
                              ps{0}->sample_vec_field(vel_{1}, velocity);
                              ps{0}->sample_vec_field(acc_{1}, acceleration);
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
                          """.format(self.particle_species, fields_name)
        self.particle_start += 'sprintf(fname, "tracers{0}");\n'.format(self.particle_species)
        self.particle_end += ('ps{0}->write(stat_file);\n' +
                              'delete ps{0};\n').format(self.particle_species)
        self.particle_includes += '#include "{0}.hpp"\n'.format(particle_class)
        if particle_class == 'particles':
            if integration_method == 'AdamsBashforth':
                multistep = 'true'
            else:
                multistep = 'false'
            self.particle_variables += 'particles<VELOCITY_TRACER, {0}, {1}, {2}> *ps{3};\n'.format(
                    self.C_dtype,
                    multistep,
                    neighbours,
                    self.particle_species)
            self.particle_start += ('ps{0} = new particles<VELOCITY_TRACER, {1}, {2},{3}>(\n' +
                                    'fname, fs, vel_{4},\n' +
                                    'nparticles,\n' +
                                    'niter_part, tracers{0}_integration_steps);\n').format(
                                            self.particle_species, self.C_dtype, multistep, neighbours, fields_name)
        else:
            self.particle_variables += 'rFFTW_particles<VELOCITY_TRACER, {0}, {1}> *ps{2};\n'.format(
                    self.C_dtype,
                    neighbours,
                    self.particle_species)
            self.particle_start += ('ps{0} = new rFFTW_particles<VELOCITY_TRACER, {1}, {2}>(\n' +
                                    'fname, fs, vel_{3},\n' +
                                    'nparticles,\n' +
                                    'niter_part, tracers{0}_integration_steps);\n').format(
                                            self.particle_species, self.C_dtype, neighbours, fields_name)
        self.particle_start += ('ps{0}->dt = dt;\n' +
                                'ps{0}->iteration = iteration;\n' +
                                'ps{0}->read(stat_file);\n').format(self.particle_species)
        self.particle_start += output_vel_acc
        if not frozen_particles:
            if particle_class == 'particles':
                if integration_method == 'AdamsBashforth':
                    self.particle_loop += 'ps{0}->AdamsBashforth((ps{0}->iteration < ps{0}->integration_steps) ? ps{0}->iteration+1 : ps{0}->integration_steps);\n'.format(self.particle_species)
                elif integration_method == 'Euler':
                    self.particle_loop += 'ps{0}->Euler();\n'.format(self.particle_species)
                elif integration_method == 'Heun':
                    assert(integration_steps == 2)
                    self.particle_loop += 'ps{0}->Heun();\n'.format(self.particle_species)
                elif integration_method == 'cRK4':
                    assert(integration_steps == 4)
                    self.particle_loop += 'ps{0}->cRK4();\n'.format(self.particle_species)
                self.particle_loop += 'ps{0}->iteration++;\n'.format(self.particle_species)
                self.particle_loop += 'ps{0}->synchronize();\n'.format(self.particle_species)
            elif particle_class == 'rFFTW_particles':
                self.particle_loop += 'ps{0}->step();\n'.format(self.particle_species)
        self.particle_loop += (('if (ps{0}->iteration % niter_part == 0)\n' +
                                '{{\n' +
                                'ps{0}->write(stat_file, false);\n').format(self.particle_species) +
                               output_vel_acc + '}\n')
        self.particle_species += 1
        return None
    def get_data_file(self):
        return h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r')
    def get_postprocess_file_name(self):
        return os.path.join(self.work_dir, self.simname + '_postprocess.h5')
    def compute_statistics(self, iter0 = 0, iter1 = None):
        if len(list(self.statistics.keys())) > 0:
            return None
        self.read_parameters()
        with self.get_data_file() as data_file:
            if 'moments' not in data_file['statistics'].keys():
                return None
            iter0 = min(data_file['statistics/moments/velocity'].shape[0]*self.parameters['niter_stat']-1,
                        iter0)
            if type(iter1) == type(None):
                iter1 = data_file['iteration'].value
            else:
                iter1 = min(data_file['iteration'].value, iter1)
            ii0 = iter0 // self.parameters['niter_stat']
            ii1 = iter1 // self.parameters['niter_stat']
            self.statistics['kshell'] = data_file['kspace/kshell'].value
            self.statistics['kM'] = data_file['kspace/kM'].value
            self.statistics['dk'] = data_file['kspace/dk'].value
            if self.particle_species > 0:
                self.trajectories = [
                        data_file['particles/' + key + '/state'][
                            iter0//self.parameters['niter_part']:iter1//self.parameters['niter_part']+1]
                                     for key in data_file['particles'].keys()]
            computation_needed = True
            pp_file = h5py.File(self.get_postprocess_file_name(), 'a')
            if 'ii0' in pp_file.keys():
                computation_needed =  not (ii0 == pp_file['ii0'].value and
                                           ii1 == pp_file['ii1'].value)
                if computation_needed:
                    for k in pp_file.keys():
                        del pp_file[k]
            if computation_needed:
                pp_file['iter0'] = iter0
                pp_file['iter1'] = iter1
                pp_file['ii0'] = ii0
                pp_file['ii1'] = ii1
                pp_file['t'] = (self.parameters['dt']*
                                self.parameters['niter_stat']*
                                (np.arange(ii0, ii1+1).astype(np.float)))
                pp_file['energy(t, k)'] = (
                    data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 0, 0] +
                    data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 1, 1] +
                    data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 2, 2])/2
                pp_file['enstrophy(t, k)'] = (
                    data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 0, 0] +
                    data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 1, 1] +
                    data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 2, 2])/2
                pp_file['vel_max(t)'] = data_file['statistics/moments/velocity']  [ii0:ii1+1, 9, 3]
                pp_file['renergy(t)'] = data_file['statistics/moments/velocity'][ii0:ii1+1, 2, 3]/2
                if 'trS2_Q_R' in data_file['statistics/moments'].keys():
                    pp_file['mean_trS2(t)'] = data_file['statistics/moments/trS2_Q_R'][:, 1, 0]
            for k in ['t',
                      'energy(t, k)',
                      'enstrophy(t, k)',
                      'vel_max(t)',
                      'renergy(t)',
                      'mean_trS2(t)']:
                if k in pp_file.keys():
                    self.statistics[k] = pp_file[k].value
            self.compute_time_averages()
        return None
    def compute_time_averages(self):
        for key in ['energy', 'enstrophy']:
            self.statistics[key + '(t)'] = (self.statistics['dk'] *
                                            np.sum(self.statistics[key + '(t, k)'], axis = 1))
        self.statistics['Uint(t)'] = np.sqrt(2*self.statistics['energy(t)'] / 3)
        self.statistics['Lint(t)'] = ((self.statistics['dk']*np.pi / (2*self.statistics['Uint(t)']**2)) *
                                      np.nansum(self.statistics['energy(t, k)'] /
                                                self.statistics['kshell'][None, :], axis = 1))
        for key in ['energy',
                    'enstrophy',
                    'vel_max',
                    'mean_trS2',
                    'Uint',
                    'Lint']:
            if key + '(t)' in self.statistics.keys():
                self.statistics[key] = np.average(self.statistics[key + '(t)'], axis = 0)
        for suffix in ['', '(t)']:
            self.statistics['diss'    + suffix] = (self.parameters['nu'] *
                                                   self.statistics['enstrophy' + suffix]*2)
            self.statistics['etaK'    + suffix] = (self.parameters['nu']**3 /
                                                   self.statistics['diss' + suffix])**.25
            self.statistics['tauK'    + suffix] =  (self.parameters['nu'] /
                                                    self.statistics['diss' + suffix])**.5
            self.statistics['Re' + suffix] = (self.statistics['Uint' + suffix] *
                                              self.statistics['Lint' + suffix] /
                                              self.parameters['nu'])
            self.statistics['lambda' + suffix] = (15 * self.parameters['nu'] *
                                                  self.statistics['Uint' + suffix]**2 /
                                                  self.statistics['diss' + suffix])**.5
            self.statistics['Rlambda' + suffix] = (self.statistics['Uint' + suffix] *
                                                   self.statistics['lambda' + suffix] /
                                                   self.parameters['nu'])
        self.statistics['Tint'] = self.statistics['Lint'] / self.statistics['Uint']
        self.statistics['Taylor_microscale'] = self.statistics['lambda']
        return None
    def set_plt_style(
            self,
            style = {'dashes' : (None, None)}):
        self.style.update(style)
        return None
    def read_cfield(
            self,
            field_name = 'vorticity',
            iteration = 0):
        return np.memmap(
                os.path.join(self.work_dir,
                             self.simname + '_{0}_i{1:0>5x}'.format('c' + field_name, iteration)),
                dtype = self.ctype,
                mode = 'r',
                shape = (self.parameters['ny'],
                         self.parameters['nz'],
                         self.parameters['nx']//2+1,
                         3))
    def write_par(self, iter0 = 0):
        super(NavierStokes, self).write_par(iter0 = iter0)
        with h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'r+') as ofile:
            kspace = self.get_kspace()
            nshells = kspace['nshell'].shape[0]
            if self.QR_stats_on:
                time_chunk = 2**20//(8*3*self.parameters['histogram_bins'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/trS2_Q_R',
                                     (1,
                                      self.parameters['histogram_bins'],
                                      3),
                                     chunks = (time_chunk,
                                               self.parameters['histogram_bins'],
                                               3),
                                     maxshape = (None,
                                                 self.parameters['histogram_bins'],
                                                 3),
                                     dtype = np.int64,
                                     compression = 'gzip')
                time_chunk = 2**20//(8*3*10)
                time_chunk = max(time_chunk, 1)
                a = ofile.create_dataset('statistics/moments/trS2_Q_R',
                                     (1, 10, 3),
                                     chunks = (time_chunk, 10, 3),
                                     maxshape = (None, 10, 3),
                                     dtype = np.float64,
                                     compression = 'gzip')
                time_chunk = 2**20//(8*self.parameters['QR2D_histogram_bins']**2)
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/QR2D',
                                     (1,
                                      self.parameters['QR2D_histogram_bins'],
                                      self.parameters['QR2D_histogram_bins']),
                                     chunks = (time_chunk,
                                               self.parameters['QR2D_histogram_bins'],
                                               self.parameters['QR2D_histogram_bins']),
                                     maxshape = (None,
                                                 self.parameters['QR2D_histogram_bins'],
                                                 self.parameters['QR2D_histogram_bins']),
                                     dtype = np.int64,
                                     compression = 'gzip')
        return None

def launch(
        opt,
        nu = None):
    c = NavierStokes(work_dir = opt.work_dir)
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
    c.parameters['niter_part'] = 2
    c.parameters['famplitude'] = 0.2
    c.parameters['nparticles'] = 32
    if opt.particles:
        c.add_particles()
        c.add_particles(kcut = 'fs->kM/2')
    c.finalize_code()
    c.write_src()
    c.write_par()
    if opt.run:
        if opt.iteration == 0 and opt.initialize:
            c.generate_initial_condition()
        c.run(ncpu = opt.ncpu, njobs = opt.njobs)
    return c

