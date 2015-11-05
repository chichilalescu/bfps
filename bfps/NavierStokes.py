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
            frozen_fields = False):
        super(NavierStokes, self).__init__(
                name = name,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision)
        self.frozen_fields = frozen_fields
        self.fftw_plan_rigor = fftw_plan_rigor
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
                    std::cerr << "ERROR: computed nshells not equal to data file nshells\\n" << std::endl;
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
    def write_fluid_stats(self):
        self.fluid_includes += '#include <cmath>\n'
        self.fluid_includes += '#include "fftw_tools.hpp"\n'
        if self.dtype == np.float32:
            field_H5T = 'H5T_NATIVE_FLOAT'
        elif self.dtype == np.float64:
            field_H5T = 'H5T_NATIVE_DOUBLE'
        self.stat_src += """
                //begincpp
                    double *velocity_moments = fftw_alloc_real(10*4);
                    double *vorticity_moments = fftw_alloc_real(10*4);
                    ptrdiff_t *hist_velocity = new ptrdiff_t[histogram_bins*4];
                    ptrdiff_t *hist_vorticity = new ptrdiff_t[histogram_bins*4];
                    fs->compute_velocity(fs->cvorticity);
                    double *spec_velocity = fftw_alloc_real(fs->nshells*9);
                    double *spec_vorticity = fftw_alloc_real(fs->nshells*9);
                    fs->cospectrum(fs->cvelocity, fs->cvelocity, spec_velocity);
                    fs->cospectrum(fs->cvorticity, fs->cvorticity, spec_vorticity);
                    fs->ift_velocity();
                    fs->compute_rspace_stats(fs->rvelocity,
                                             velocity_moments,
                                             hist_velocity,
                                             max_velocity_estimate,
                                             histogram_bins);
                    fs->ift_vorticity();
                    fs->compute_rspace_stats(fs->rvorticity,
                                             vorticity_moments,
                                             hist_vorticity,
                                             max_vorticity_estimate,
                                             histogram_bins);
                    if (myrank == 0)
                    {
                        hid_t Cdset, wspace, mspace;
                        int ndims;
                        hsize_t count[4], offset[4], dims[4];
                        offset[0] = fs->iteration/niter_stat;
                        offset[1] = 0;
                        offset[2] = 0;
                        offset[3] = 0;
                //endcpp
                """
        size_setups = ["""
                        count[0] = 1;
                        count[1] = nx;
                        count[2] = 3;
                       """,
                       """
                        count[0] = 1;
                        count[1] = 10;
                        count[2] = 4;
                       """,
                       """
                        count[0] = 1;
                        count[1] = histogram_bins;
                        count[2] = 4;
                       """,
                       """
                        count[0] = 1;
                        count[1] = fs->nshells;
                        count[2] = 3;
                        count[3] = 3;
                       """]
        stat_template = """
                //begincpp
                        Cdset = H5Dopen(stat_file, "{0}", H5P_DEFAULT);
                        wspace = H5Dget_space(Cdset);
                        ndims = H5Sget_simple_extent_dims(wspace, dims, NULL);
                        mspace = H5Screate_simple(ndims, count, NULL);
                        H5Sselect_hyperslab(wspace, H5S_SELECT_SET, offset, NULL, count, NULL);
                        H5Dwrite(Cdset, {1}, mspace, wspace, H5P_DEFAULT, {2});
                        H5Dclose(Cdset);
                        Cdset = H5Dopen(stat_file, "{3}", H5P_DEFAULT);
                        H5Dwrite(Cdset, {1}, mspace, wspace, H5P_DEFAULT, {4});
                        H5Sclose(mspace);
                        H5Sclose(wspace);
                        H5Dclose(Cdset);
                //endcpp
                """
        stat_outputs = [stat_template.format('/statistics/xlines/velocity',
                                              field_H5T,
                                              'fs->rvelocity',
                                              '/statistics/xlines/vorticity',
                                              'fs->rvorticity'),
                        stat_template.format('/statistics/moments/velocity',
                                             'H5T_NATIVE_DOUBLE',
                                             'velocity_moments',
                                             '/statistics/moments/vorticity',
                                             'vorticity_moments'),
                        stat_template.format('/statistics/histograms/velocity',
                                             'H5T_NATIVE_DOUBLE',
                                             'hist_velocity',
                                             '/statistics/histograms/vorticity',
                                             'hist_vorticity'),
                        stat_template.format('/statistics/spectra/velocity_velocity',
                                             'H5T_NATIVE_DOUBLE',
                                             'spec_velocity',
                                             '/statistics/spectra/vorticity_vorticity',
                                             'spec_vorticity')]
        for i in range(len(size_setups)):
            self.stat_src += size_setups[i] + stat_outputs[i]
        self.stat_src += """
                //begincpp
                    }
                    fftw_free(spec_velocity);
                    fftw_free(spec_vorticity);
                    fftw_free(velocity_moments);
                    fftw_free(vorticity_moments);
                    delete[] hist_velocity;
                    delete[] hist_vorticity;
                //endcpp
                """
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += 'fluid_solver<{0}> *fs;\n'.format(self.C_dtype)
        self.write_fluid_stats()
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
                //endcpp
                """.format(self.C_dtype, self.fftw_plan_rigor)
        if not self.frozen_fields:
            self.fluid_loop = 'fs->step(dt);\n'
        else:
            self.fluid_loop = ''
        self.fluid_loop += ('if (fs->iteration % niter_out == 0)\n{\n' +
                            self.fluid_output + '\n}\n')
        self.fluid_end = ('if (fs->iteration % niter_out != 0)\n{\n' +
                          self.fluid_output + '\n}\n' +
                          'delete fs;\n')
        return None
    def add_particles(
            self,
            neighbours = 1,
            smoothness = 1,
            interp_type = 'spline',
            integration_method = 'AdamsBashforth',
            integration_steps = 2,
            kcut = 'fs->kM',
            force_vel_reset = True,
            frozen_particles = False,
            particle_type = 'old_tracers'):
        self.parameters['integration_method{0}'.format(self.particle_species)] = integration_method
        self.parameters['interp_type{0}'.format(self.particle_species)] = interp_type
        self.parameters['neighbours{0}'.format(self.particle_species)] = neighbours
        self.parameters['smoothness{0}'.format(self.particle_species)] = smoothness
        self.parameters['kcut{0}'.format(self.particle_species)] = kcut
        self.parameters['integration_steps{0}'.format(self.particle_species)] = integration_steps
        self.file_datasets_grow += """
                        //begincpp
                        temp_string = (std::string("/particles/") +
                                       std::string(ps{0}->name));
                        group = H5Gopen(stat_file, temp_string.c_str(), H5P_DEFAULT);
                        grow_particle_datasets(group, temp_string.c_str(), NULL, NULL);
                        H5Gclose(group);
                        //endcpp
                        """.format(self.particle_species)
        update_field = 'fs->compute_velocity(fs->cvorticity);\n'
        if kcut != 'fs->kM':
            update_field += 'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut)
        update_field += ('fs->ift_velocity();\n' +
                         'clip_zero_padding(fs->rd, fs->rvelocity, 3);\n' +
                         'ps{0}->rFFTW_to_buffered(fs->rvelocity, ps{0}->data);\n').format(self.particle_species)
        if self.dtype == np.float32:
            FFTW = 'fftwf'
        elif self.dtype == np.float64:
            FFTW = 'fftw'
        compute_acc = ('{0} *acc_field_tmp = {1}_alloc_real(fs->rd->local_size);\n' +
                       'fs->compute_Lagrangian_acceleration(acc_field_tmp);\n' +
                       'ps{2}->rFFTW_to_buffered(acc_field_tmp, ps{2}->data);\n' +
                       'ps{2}->sample_vec_field(ps{2}->data, acceleration);\n' +
                       '{1}_free(acc_field_tmp);\n').format(self.C_dtype, FFTW, self.particle_species)
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
        if interp_type == 'spline':
            beta_name = 'beta_n{0}_m{1}'.format(neighbours, smoothness)
        elif interp_type == 'Lagrange':
            beta_name = 'beta_Lagrange_n{0}'.format(neighbours)
        self.particle_start += 'sprintf(fname, "tracers{0}");\n'.format(self.particle_species)
        self.particle_end += ('ps{0}->write(stat_file);\n' +
                              'delete ps{0};\n').format(self.particle_species)
        if particle_type == 'old_tracers':
            self.particle_includes += '#include "tracers.hpp"\n'
            self.particle_variables += 'tracers<{0}> *ps{1};\n'.format(self.C_dtype, self.particle_species)
            self.particle_start += ('ps{1} = new tracers<{0}>(\n' +
                                        'fname, fs,\n' +
                                        'nparticles,\n' +
                                        '{2},\n' +
                                        'neighbours{1}, niter_part, integration_steps{1},\n' +
                                        'fs->ru);\n').format(self.C_dtype, self.particle_species, beta_name)
        else:
            self.particle_includes += '#include "particles.hpp"\n'
            self.particle_variables += 'particles<VELOCITY_TRACER, {0}, true, 3, {1}> *ps{2};\n'.format(
                    self.C_dtype,
                    neighbours,
                    self.particle_species)
            self.particle_variables += '{0} *particle_field{1};\n'.format(self.C_dtype, self.particle_species)
            self.particle_start += ('ps{0} = new particles<VELOCITY_TRACER, {1}, true, 3, {2}>(\n' +
                                        'fname, fs,\n' +
                                        'nparticles,\n' +
                                        '{2},\n' +
                                        'niter_part, integration_steps{1});\n').format(self.particle_species, self.C_dtype, neighbours, beta_name)
            self.particle_start += ('particle_field{0} = {1}_alloc_real(fs->rd->local_size);\n' +
                                    'ps{0}->data = particle_field{0};\n').format(self.particle_species, FFTW)
            self.particle_end += '{0}_free(particle_field{1});\n'.format(FFTW, self.particle_species)
        self.particle_start += ('ps{0}->dt = dt;\n' +
                                'ps{0}->iteration = iteration;\n' +
                                update_field +
                                'ps{0}->read(stat_file);\n').format(self.particle_species)
        self.particle_start += output_vel_acc
        self.particle_loop += update_field
        if not frozen_particles:
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
        self.particle_loop += (('if (ps{0}->iteration % niter_part == 0)\n' +
                                'ps{0}->write(stat_file, false);\n').format(self.particle_species) +
                               output_vel_acc)
        self.particle_species += 1
        return None
    def get_data_file(self):
        return h5py.File(os.path.join(self.work_dir, self.simname + '.h5'), 'a')
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
            if self.particle_species > 0:
                self.trajectories = [
                        data_file['particles/' + key + '/state'][
                            iter0//self.parameters['niter_part']:iter1//self.parameters['niter_part']+1]
                                     for key in data_file['particles'].keys()]
            computation_needed = True
            if 'postprocess' in data_file.keys():
                computation_needed =  not (ii0 == data_file['postprocess/ii0'].value and
                                           ii1 == data_file['postprocess/ii1'].value)
                if computation_needed:
                    del data_file['postprocess']
            if computation_needed:
                data_file['postprocess/iter0'] = iter0
                data_file['postprocess/iter1'] = iter1
                data_file['postprocess/ii0'] = ii0
                data_file['postprocess/ii1'] = ii1
                data_file['postprocess/t'] = (self.parameters['dt']*
                                              self.parameters['niter_stat']*
                                              (np.arange(ii0, ii1+1).astype(np.float)))
                data_file['postprocess/energy(t, k)'] = (
                        data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 0, 0] +
                        data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 1, 1] +
                        data_file['statistics/spectra/velocity_velocity'][ii0:ii1+1, :, 2, 2])/2
                data_file['postprocess/enstrophy(t, k)'] = (
                        data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 0, 0] +
                        data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 1, 1] +
                        data_file['statistics/spectra/vorticity_vorticity'][ii0:ii1+1, :, 2, 2])/2
                data_file['postprocess/vel_max(t)'] = data_file['statistics/moments/velocity']  [ii0:ii1+1, 9, 3]
                data_file['postprocess/renergy(t)'] = data_file['statistics/moments/velocity'][ii0:ii1+1, 2, 3]/2
            for k in ['t',
                      'energy(t, k)',
                      'enstrophy(t, k)',
                      'vel_max(t)',
                      'renergy(t)']:
                self.statistics[k] = data_file['postprocess/' + k].value
            self.compute_time_averages()
        return None
    def compute_time_averages(self):
        for key in ['energy', 'enstrophy']:
            self.statistics[key + '(t)'] = np.sum(self.statistics[key + '(t, k)'], axis = 1)
        for key in ['energy', 'enstrophy', 'vel_max']:
            self.statistics[key] = np.average(self.statistics[key + '(t)'], axis = 0)
        for suffix in ['', '(t)']:
            self.statistics['diss'    + suffix] = (self.parameters['nu'] *
                                                   self.statistics['enstrophy' + suffix]*2)
            self.statistics['etaK'    + suffix] = (self.parameters['nu']**3 /
                                                   self.statistics['diss' + suffix])**.25
            self.statistics['Rlambda' + suffix] = (2*np.sqrt(5./3) *
                                                   (self.statistics['energy' + suffix] /
                                                   (self.parameters['nu']*self.statistics['diss' + suffix])**.5))
            self.statistics['tauK'    + suffix] =  (self.parameters['nu'] /
                                                    self.statistics['diss' + suffix])**.5
        self.statistics['Tint'] = 2*self.statistics['energy'] / self.statistics['diss']
        self.statistics['Lint'] = (2*self.statistics['energy'])**1.5 / self.statistics['diss']
        self.statistics['Taylor_microscale'] = (10 * self.parameters['nu'] * self.statistics['energy'] / self.statistics['diss'])**.5
        return None
    def set_plt_style(
            self,
            style = {'dashes' : (None, None)}):
        self.style.update(style)
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

