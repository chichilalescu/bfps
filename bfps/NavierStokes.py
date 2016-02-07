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



import sys
import os
import numpy as np
import h5py
import argparse

import bfps
from ._code import _code
from ._fluid_base import _fluid_particle_base

class NavierStokes(_fluid_particle_base):
    """Objects of this class can be used to generate production DNS codes.
    Any functionality that users require should be available through this class,
    in the sense that they can implement whatever they need by simply inheriting
    this class.
    """
    def __init__(
            self,
            name = 'NavierStokes-v' + bfps.__version__,
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
        _fluid_particle_base.__init__(
                self,
                name = name + '-' + fluid_precision,
                work_dir = work_dir,
                simname = simname,
                dtype = fluid_precision,
                use_fftw_wisdom = use_fftw_wisdom)
        self.parameters['nu'] = 0.1
        self.parameters['fmode'] = 1
        self.parameters['famplitude'] = 0.5
        self.parameters['fk0'] = 2.0
        self.parameters['fk1'] = 4.0
        self.parameters['forcing_type'] = 'linear'
        self.parameters['histogram_bins'] = 256
        self.parameters['max_velocity_estimate'] = 1.0
        self.parameters['max_vorticity_estimate'] = 1.0
        self.parameters['QR2D_histogram_bins'] = 64
        self.parameters['max_trS2_estimate'] = 1.0
        self.parameters['max_Q_estimate'] = 1.0
        self.parameters['max_R_estimate'] = 1.0
        self.file_datasets_grow = """
                //begincpp
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
                H5Dwrite(Cdset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fs->kMspec);
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
                double *gradu_moments     = new double[10*9];
                ptrdiff_t *hist_trS2_Q_R  = new ptrdiff_t[histogram_bins*3];
                ptrdiff_t *hist_gradu     = new ptrdiff_t[histogram_bins*9];
                ptrdiff_t *hist_QR2D      = new ptrdiff_t[QR2D_histogram_bins*QR2D_histogram_bins];
                double trS2QR_max_estimates[3];
                double gradu_max_estimates[9];
                trS2QR_max_estimates[0] = max_trS2_estimate;
                trS2QR_max_estimates[1] = max_Q_estimate;
                trS2QR_max_estimates[2] = max_R_estimate;
                std::fill_n(gradu_max_estimates, 9, sqrt(3*max_trS2_estimate));
                fs->compute_gradient_statistics(
                    fs->cvelocity,
                    gradu_moments,
                    trS2_Q_R_moments,
                    hist_gradu,
                    hist_trS2_Q_R,
                    hist_QR2D,
                    trS2QR_max_estimates,
                    gradu_max_estimates,
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
                fs->compute_rspace_stats4(fs->rvelocity,
                                         velocity_moments,
                                         hist_velocity,
                                         max_estimates,
                                         histogram_bins);
                fs->ift_vorticity();
                max_estimates[0] = max_vorticity_estimate/sqrt(3);
                max_estimates[1] = max_estimates[0];
                max_estimates[2] = max_estimates[0];
                max_estimates[3] = max_vorticity_estimate;
                fs->compute_rspace_stats4(fs->rvorticity,
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
                    '/statistics/moments/velocity_gradient',
                    'gradu_moments',
                    size_setup ="""
                        count[0] = 1;
                        count[1] = 10;
                        count[2] = 3;
                        count[3] = 3;
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
                    '/statistics/histograms/velocity_gradient',
                    'hist_gradu',
                    data_type = 'H5T_NATIVE_INT64',
                    size_setup = """
                        count[0] = 1;
                        count[1] = histogram_bins;
                        count[2] = 3;
                        count[3] = 3;
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
                delete[] gradu_moments;
                delete[] hist_trS2_Q_R;
                delete[] hist_gradu;
                delete[] hist_QR2D;
                //endcpp
                """
        return None
    def fill_up_fluid_code(self):
        self.fluid_includes += '#include <cstring>\n'
        self.fluid_variables += ('fluid_solver<{0}> *fs;\n'.format(self.C_dtype) +
                                 'hid_t particle_file;\n' +
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
        if self.parameters['nparticles'] > 0:
            self.fluid_start += """
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
                          'H5Tclose(H5T_field_complex);\n' +
                          '}\n' +
                          'delete fs;\n')
        if self.parameters['nparticles'] > 0:
            self.fluid_end += 'H5Fclose(particle_file);\n'
        return None
    def add_3D_rFFTW_field(
            self,
            name = 'rFFTW_acc'):
        if self.dtype == np.float32:
            FFTW = 'fftwf'
        elif self.dtype == np.float64:
            FFTW = 'fftw'
        self.fluid_variables += '{0} *{1};\n'.format(self.C_dtype, name)
        self.fluid_start += '{0} = {1}_alloc_real(2*fs->cd->local_size);\n'.format(name, FFTW)
        self.fluid_end   += '{0}_free({1});\n'.format(FFTW, name)
        return None
    def add_interpolator(
            self,
            interp_type = 'spline',
            neighbours = 1,
            smoothness = 1,
            name = 'field_interpolator',
            field_name = 'fs->rvelocity'):
        self.fluid_includes += '#include "rFFTW_interpolator.hpp"\n'
        self.fluid_variables += 'rFFTW_interpolator <{0}, {1}> *{2};\n'.format(
                self.C_dtype, neighbours, name)
        self.parameters[name + '_type'] = interp_type
        self.parameters[name + '_neighbours'] = neighbours
        if interp_type == 'spline':
            self.parameters[name + '_smoothness'] = smoothness
            beta_name = 'beta_n{0}_m{1}'.format(neighbours, smoothness)
        elif interp_type == 'Lagrange':
            beta_name = 'beta_Lagrange_n{0}'.format(neighbours)
        self.fluid_start += '{0} = new rFFTW_interpolator<{1}, {2}>(fs, {3}, {4});\n'.format(
                name,
                self.C_dtype,
                neighbours,
                beta_name,
                field_name)
        self.fluid_end += 'delete {0};\n'.format(name)
        return None
    def add_particles(
            self,
            integration_steps = 2,
            kcut = None,
            interpolator = 'field_interpolator',
            frozen_particles = False,
            acc_name = None):
        """Adds code for tracking a series of particle species, each
        consisting of `nparticles` particles.

        :type integration_steps: int, list of int
        :type kcut: None (default), str, list of str
        :type interpolator: str, list of str
        :type frozen_particles: bool
        :type acc_name: str

        .. warning:: if not None, kcut must be a list of decreasing
                     wavenumbers, since filtering is done sequentially
                     on the same complex FFTW field.
        """
        if self.dtype == np.float32:
            FFTW = 'fftwf'
        elif self.dtype == np.float64:
            FFTW = 'fftw'
        s0 = self.particle_species
        if type(integration_steps) == int:
            integration_steps = [integration_steps]
        if type(kcut) == str:
            kcut = [kcut]
        if type(interpolator) == str:
            interpolator = [interpolator]
        nspecies = max(len(integration_steps), len(interpolator))
        if type(kcut) == list:
            nspecies = max(nspecies, len(kcut))
        if len(integration_steps) == 1:
            integration_steps = [integration_steps[0] for s in range(nspecies)]
        if len(interpolator) == 1:
            interpolator = [interpolator[0] for s in range(nspecies)]
        if type(kcut) == list:
            if len(kcut) == 1:
                kcut = [kcut[0] for s in range(nspecies)]
        assert(len(integration_steps) == nspecies)
        assert(len(interpolator) == nspecies)
        if type(kcut) == list:
            assert(len(kcut) == nspecies)
        for s in range(nspecies):
            neighbours = self.parameters[interpolator[s] + '_neighbours']
            if type(kcut) == list:
                self.parameters['tracers{0}_kcut'.format(s0 + s)] = kcut[s]
            self.parameters['tracers{0}_interpolator'.format(s0 + s)] = interpolator[s]
            self.parameters['tracers{0}_acc_on'.format(s0 + s)] = int(not type(acc_name) == type(None))
            self.parameters['tracers{0}_integration_steps'.format(s0 + s)] = integration_steps[s]
            self.file_datasets_grow += """
                        //begincpp
                        group = H5Gopen(particle_file, ps{0}->name, H5P_DEFAULT);
                        grow_particle_datasets(group, "", NULL, NULL);
                        H5Gclose(group);
                        //endcpp
                        """.format(s0 + s)

        #### code that outputs statistics
        output_vel_acc = '{\n'
        # array for putting sampled velocity in
        # must compute velocity, just in case it was messed up by some
        # other particle species before the stats
        output_vel_acc += ('double *velocity = new double[3*nparticles];\n' +
                           'fs->compute_velocity(fs->cvorticity);\n')
        if not type(kcut) == list:
            output_vel_acc += 'fs->ift_velocity();\n'
        if not type(acc_name) == type(None):
            # array for putting sampled acceleration in
            # must compute acceleration
            output_vel_acc += 'double *acceleration = new double[3*nparticles];\n'
            output_vel_acc += 'fs->compute_Lagrangian_acceleration({0});\n'.format(acc_name)
        for s in range(nspecies):
            if type(kcut) == list:
                output_vel_acc += 'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut[s])
                output_vel_acc += 'fs->ift_velocity();\n'
            output_vel_acc += """
                {0}->field = fs->rvelocity;
                {0}->sample(ps{1}->nparticles, ps{1}->ncomponents, ps{1}->state, velocity);
                """.format(interpolator[s], s0 + s)
            if not type(acc_name) == type(None):
                output_vel_acc += """
                    {0}->field = {1};
                    {0}->sample(ps{2}->nparticles, ps{2}->ncomponents, ps{2}->state, acceleration);
                    """.format(interpolator[s], acc_name, s0 + s)
            output_vel_acc += (
                    'if (myrank == 0)\n' +
                    '{\n' +
                    'ps{0}->write(particle_file, "velocity", velocity);\n'.format(s0 + s))
            if not type(acc_name) == type(None):
                output_vel_acc += (
                        'ps{0}->write(particle_file, "acceleration", acceleration);\n'.format(s0 + s))
            output_vel_acc += '}\n'
        output_vel_acc += 'delete[] velocity;\n'
        if not type(acc_name) == type(None):
            output_vel_acc += 'delete[] acceleration;\n'
        output_vel_acc += '}\n'

        #### initialize, stepping and finalize code
        if not type(kcut) == list:
            update_fields = ('fs->compute_velocity(fs->cvorticity);\n' +
                             'fs->ift_velocity();\n')
            self.particle_start += update_fields
            self.particle_loop  += update_fields
        else:
            self.particle_loop += 'fs->compute_velocity(fs->cvorticity);\n'
        self.particle_includes += '#include "rFFTW_particles.hpp"\n'
        self.particle_stat_src += (
                'if (ps0->iteration % niter_part == 0)\n' +
                '{\n')
        for s in range(nspecies):
            neighbours = self.parameters[interpolator[s] + '_neighbours']
            self.particle_start += 'sprintf(fname, "tracers{0}");\n'.format(s0 + s)
            self.particle_end += ('ps{0}->write(particle_file);\n' +
                                  'delete ps{0};\n').format(s0 + s)
            self.particle_variables += 'rFFTW_particles<VELOCITY_TRACER, {0}, {1}> *ps{2};\n'.format(
                    self.C_dtype,
                    neighbours,
                    s0 + s)
            self.particle_start += ('ps{0} = new rFFTW_particles<VELOCITY_TRACER, {1}, {2}>(\n' +
                                    'fname, {3},\n' +
                                    'nparticles,\n' +
                                    'niter_part, tracers{0}_integration_steps);\n').format(
                                            s0 + s,
                                            self.C_dtype,
                                            neighbours,
                                            interpolator[s])
            self.particle_start += ('ps{0}->dt = dt;\n' +
                                    'ps{0}->iteration = iteration;\n' +
                                    'ps{0}->read(particle_file);\n').format(s0 + s)
            if not frozen_particles:
                if type(kcut) == list:
                    update_field = ('fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut[s]) +
                                    'fs->ift_velocity();\n')
                    self.particle_loop += update_field
                self.particle_loop += '{0}->field = fs->rvelocity;\n'.format(interpolator[s])
                self.particle_loop += 'ps{0}->step();\n'.format(s0 + s)
            self.particle_stat_src += 'ps{0}->write(particle_file, false);\n'.format(s0 + s)
        self.particle_stat_src += output_vel_acc
        self.particle_stat_src += '}\n'
        self.particle_species += nspecies
        return None
    def get_data_file_name(self):
        return os.path.join(self.work_dir, self.simname + '.h5')
    def get_data_file(self):
        return h5py.File(self.get_data_file_name(), 'r')
    def get_particle_file_name(self):
        return os.path.join(self.work_dir, self.simname + '_particles.h5')
    def get_particle_file(self):
        return h5py.File(self.get_particle_file_name(), 'r')
    def get_postprocess_file_name(self):
        return os.path.join(self.work_dir, self.simname + '_postprocess.h5')
    def get_postprocess_file(self):
        return h5py.File(self.get_postprocess_file_name(), 'r')
    def compute_statistics(self, iter0 = 0, iter1 = None):
        """Run basic postprocessing on raw data.
        The energy spectrum :math:`E(t, k)` and the enstrophy spectrum
        :math:`\\frac{1}{2}\omega^2(t, k)` are computed from the

        .. math::

            \sum_{k \\leq \\|\\mathbf{k}\\| \\leq k+dk}\\hat{u_i} \\hat{u_j}^*, \\hskip .5cm
            \sum_{k \\leq \\|\\mathbf{k}\\| \\leq k+dk}\\hat{\omega_i} \\hat{\\omega_j}^*

        tensors, and the enstrophy spectrum is also used to
        compute the dissipation :math:`\\varepsilon(t)`.
        These basic quantities are stored in a newly created HDF5 file,
        ``simname_postprocess.h5``.
        """
        if len(list(self.statistics.keys())) > 0:
            return None
        self.read_parameters()
        with self.get_data_file() as data_file:
            if 'moments' not in data_file['statistics'].keys():
                return None
            iter0 = min((data_file['statistics/moments/velocity'].shape[0] *
                         self.parameters['niter_stat']-1),
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
        """Compute easy stats.

        Further computation of statistics based on the contents of
        ``simname_postprocess.h5``.
        Standard quantities are as follows
        (consistent with [Ishihara]_):

        .. math::

            U_{\\textrm{int}}(t) = \\sqrt{\\frac{2E(t)}{3}}, \\hskip .5cm
            L_{\\textrm{int}}(t) = \\frac{\pi}{2U_{int}^2} \\int \\frac{dk}{k} E(t, k), \\hskip .5cm
            T_{\\textrm{int}}(t) =
            \\frac{L_{\\textrm{int}}(t)}{U_{\\textrm{int}}(t)}

            \\eta_K = \\left(\\frac{\\nu^3}{\\varepsilon}\\right)^{1/4}, \\hskip .5cm
            \\tau_K = \\left(\\frac{\\nu}{\\varepsilon}\\right)^{1/2}, \\hskip .5cm
            \\lambda = \\sqrt{\\frac{15 \\nu U_{\\textrm{int}}^2}{\\varepsilon}}

            Re = \\frac{U_{\\textrm{int}} L_{\\textrm{int}}}{\\nu}, \\hskip
            .5cm
            R_{\\lambda} = \\frac{U_{\\textrm{int}} \\lambda}{\\nu}

        .. [Ishihara] T. Ishihara et al,
                      *Small-scale statistics in high-resolution direct numerical
                      simulation of turbulence: Reynolds number dependence of
                      one-point velocity gradient statistics*.
                      J. Fluid Mech.,
                      **592**, 335-366, 2007
        """
        for key in ['energy', 'enstrophy']:
            self.statistics[key + '(t)'] = (self.statistics['dk'] *
                                            np.sum(self.statistics[key + '(t, k)'], axis = 1))
        self.statistics['Uint(t)'] = np.sqrt(2*self.statistics['energy(t)'] / 3)
        self.statistics['Lint(t)'] = ((self.statistics['dk']*np.pi /
                                       (2*self.statistics['Uint(t)']**2)) *
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
            self.statistics['kMeta' + suffix] = (self.statistics['kM'] *
                                                 self.statistics['etaK' + suffix])
            if self.parameters['dealias_type'] == 1:
                self.statistics['kMeta' + suffix] *= 0.8
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
        """read the Fourier representation of a vector field.

        Read the binary file containing iteration ``iteration`` of the
        field ``field_name``, and return it as a properly shaped
        ``numpy.memmap`` object.
        """
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
        _fluid_particle_base.write_par(self, iter0 = iter0)
        with h5py.File(self.get_data_file_name(), 'r+') as ofile:
            kspace = self.get_kspace()
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
                time_chunk = 2**20//(8*9*self.parameters['histogram_bins'])
                time_chunk = max(time_chunk, 1)
                ofile.create_dataset('statistics/histograms/velocity_gradient',
                                     (1,
                                      self.parameters['histogram_bins'],
                                      3,
                                      3),
                                     chunks = (time_chunk,
                                               self.parameters['histogram_bins'],
                                               3,
                                               3),
                                     maxshape = (None,
                                                 self.parameters['histogram_bins'],
                                                 3,
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
                time_chunk = 2**20//(8*9*10)
                time_chunk = max(time_chunk, 1)
                a = ofile.create_dataset('statistics/moments/velocity_gradient',
                                     (1, 10, 3, 3),
                                     chunks = (time_chunk, 10, 3, 3),
                                     maxshape = (None, 10, 3, 3),
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
        if self.particle_species == 0:
            return None
        def create_particle_dataset(
                data_file,
                dset_name,
                dset_shape,
                dset_maxshape,
                dset_chunks,
                # maybe something more general can be used here
                dset_dtype = h5py.h5t.IEEE_F64LE):
            # create the dataspace.
            space_id = h5py.h5s.create_simple(
                    dset_shape,
                    dset_maxshape)
            # create the dataset creation property list.
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            # set the allocation time to "early".
            dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
            dcpl.set_chunk(dset_chunks)
            # and now create dataset
            if sys.version_info[0] == 3:
                dset_name = dset_name.encode()
            return h5py.h5d.create(
                    data_file.id,
                    dset_name,
                    dset_dtype,
                    space_id,
                    dcpl,
                    h5py.h5p.DEFAULT)

        with h5py.File(self.get_particle_file_name(), 'a') as ofile:
            for s in range(self.particle_species):
                ofile.create_group('tracers{0}'.format(s))
                time_chunk = 2**20 // (8*3*
                                       self.parameters['nparticles']*
                                       self.parameters['tracers{0}_integration_steps'.format(s)])
                time_chunk = max(time_chunk, 1)
                dims = (1,
                        self.parameters['tracers{0}_integration_steps'.format(s)],
                        self.parameters['nparticles'],
                        3)
                maxshape = (h5py.h5s.UNLIMITED,
                            self.parameters['tracers{0}_integration_steps'.format(s)],
                            self.parameters['nparticles'],
                            3)
                chunks = (time_chunk,
                          self.parameters['tracers{0}_integration_steps'.format(s)],
                          self.parameters['nparticles'],
                          3)
                create_particle_dataset(
                        ofile,
                        '/tracers{0}/rhs'.format(s),
                        dims, maxshape, chunks)
                time_chunk = 2**20 // (8*3*self.parameters['nparticles'])
                time_chunk = max(time_chunk, 1)
                create_particle_dataset(
                        ofile,
                        '/tracers{0}/state'.format(s),
                        (1, self.parameters['nparticles'], 3),
                        (h5py.h5s.UNLIMITED, self.parameters['nparticles'], 3),
                        (time_chunk, self.parameters['nparticles'], 3))
                create_particle_dataset(
                        ofile,
                        '/tracers{0}/velocity'.format(s),
                        (1, self.parameters['nparticles'], 3),
                        (h5py.h5s.UNLIMITED, self.parameters['nparticles'], 3),
                        (time_chunk, self.parameters['nparticles'], 3))
                if self.parameters['tracers{0}_acc_on'.format(s)]:
                    create_particle_dataset(
                            ofile,
                            '/tracers{0}/acceleration'.format(s),
                            (1, self.parameters['nparticles'], 3),
                            (h5py.h5s.UNLIMITED, self.parameters['nparticles'], 3),
                            (time_chunk, self.parameters['nparticles'], 3))
        return None
    def add_particle_fields(
            self,
            interp_type = 'spline',
            kcut = None,
            neighbours = 1,
            smoothness = 1,
            name = 'particle_field',
            field_class = 'rFFTW_interpolator',
            acc_field_name = 'rFFTW_acc'):
        self.fluid_includes += '#include "{0}.hpp"\n'.format(field_class)
        self.fluid_variables += field_class + '<{0}, {1}> *vel_{2}, *acc_{2};\n'.format(
                self.C_dtype, neighbours, name)
        self.parameters[name + '_type'] = interp_type
        self.parameters[name + '_neighbours'] = neighbours
        if interp_type == 'spline':
            self.parameters[name + '_smoothness'] = smoothness
            beta_name = 'beta_n{0}_m{1}'.format(neighbours, smoothness)
        elif interp_type == 'Lagrange':
            beta_name = 'beta_Lagrange_n{0}'.format(neighbours)
        if field_class == 'rFFTW_interpolator':
            self.fluid_start += ('vel_{0} = new {1}<{2}, {3}>(fs, {4}, fs->rvelocity);\n' +
                                 'acc_{0} = new {1}<{2}, {3}>(fs, {4}, {5});\n').format(name,
                                                                                   field_class,
                                                                                   self.C_dtype,
                                                                                   neighbours,
                                                                                   beta_name,
                                                                                   acc_field_name)
        elif field_class == 'interpolator':
            self.fluid_start += ('vel_{0} = new {1}<{2}, {3}>(fs, {4});\n' +
                                 'acc_{0} = new {1}<{2}, {3}>(fs, {4});\n').format(name,
                                                                                   field_class,
                                                                                   self.C_dtype,
                                                                                   neighbours,
                                                                                   beta_name,
                                                                                   acc_field_name)
        self.fluid_end += ('delete vel_{0};\n' +
                           'delete acc_{0};\n').format(name)
        update_fields = 'fs->compute_velocity(fs->cvorticity);\n'
        if not type(kcut) == type(None):
            update_fields += 'fs->low_pass_Fourier(fs->cvelocity, 3, {0});\n'.format(kcut)
        update_fields += ('fs->ift_velocity();\n' +
                          'fs->compute_Lagrangian_acceleration(acc_{0}->field);\n').format(name)
        self.fluid_start += update_fields
        self.fluid_loop += update_fields
        return None
    def specific_parser_arguments(
            self,
            parser):
        _fluid_particle_base.specific_parser_arguments(self, parser)
        parser.add_argument(
                '--src-wd',
                type = str,
                dest = 'src_work_dir',
                default = '')
        parser.add_argument(
                '--src-simname',
                type = str,
                dest = 'src_simname',
                default = '')
        parser.add_argument(
                '--src-iteration',
                type = int,
                dest = 'src_iteration',
                default = 0)
        parser.add_argument(
               '--njobs',
               type = int, dest = 'njobs',
               default = 1)
        parser.add_argument(
               '--QR-stats',
               action = 'store_true',
               dest = 'QR_stats',
               help = 'add this option if you want to compute velocity gradient and QR stats')
        parser.add_argument(
               '--kMeta',
               type = float,
               dest = 'kMeta',
               default = 2.0)
        parser.add_argument(
               '--dtfactor',
               type = float,
               dest = 'dtfactor',
               default = 0.5,
               help = 'dt is computed as DTFACTOR / N')
        parser.add_argument(
               '--particle-rand-seed',
               type = int,
               dest = 'particle_rand_seed',
               default = None)
        return None
    def prepare_launch(
            self,
            args = []):
        """Set up reasonable parameters.

        With the default Lundgren forcing applied in the band [2, 4],
        we can estimate the dissipation, therefore we can estimate
        :math:`k_M \\eta_K` and constrain the viscosity.
        Also, if velocity gradient statistics are computed, the
        dissipation is used for estimating the bins of the QR histogram.

        In brief, the command line parameter :math:`k_M \\eta_K` is
        used in the following formula for :math:`\\nu` (:math:`N` is the
        number of real space grid points per coordinate):

        .. math::

            \\nu = \\left(\\frac{2 k_M \\eta_K}{N} \\right)^{4/3}

        With this choice, the average dissipation :math:`\\varepsilon`
        will be close to 0.4, and the integral scale velocity will be
        close to 0.77, yielding the approximate value for the Taylor
        microscale and corresponding Reynolds number:

        .. math::

            \\lambda \\approx 4.75\\left(\\frac{2 k_M \\eta_K}{N} \\right)^{4/6}, \\hskip .5in
            R_\\lambda \\approx 3.7 \\left(\\frac{N}{2 k_M \\eta_K} \\right)^{4/6}

        """
        opt = _code.prepare_launch(self, args = args)
        self.QR_stats_on = opt.QR_stats
        self.parameters['nu'] = (opt.kMeta * 2 / opt.n)**(4./3)
        self.parameters['dt'] = (opt.dtfactor / opt.n)
        # custom famplitude for 288 and 576
        if opt.n == 288:
            self.parameters['famplitude'] = 0.45
        elif opt.n == 576:
            self.parameters['famplitude'] = 0.47
        if ((self.parameters['niter_todo'] % self.parameters['niter_out']) != 0):
            self.parameters['niter_out'] = self.parameters['niter_todo']
        if self.QR_stats_on:
            # max_Q_estimate and max_R_estimate are just used for the 2D pdf
            # therefore I just want them to be small multiples of mean trS2
            # I'm already estimating the dissipation with kMeta...
            meantrS2 = (opt.n//2 / opt.kMeta)**4 * self.parameters['nu']**2
            self.parameters['max_Q_estimate'] = meantrS2
            self.parameters['max_R_estimate'] = .4*meantrS2**1.5
            # add QR suffix to code name, since we now expect additional
            # datasets in the .h5 file
            self.name += '-QR'
        if len(opt.src_work_dir) == 0:
            opt.src_work_dir = opt.work_dir
        self.pars_from_namespace(opt)
        return opt
    def launch(
            self,
            args = [],
            **kwargs):
        opt = self.prepare_launch(args = args)
        self.fill_up_fluid_code()
        self.finalize_code()
        self.launch_jobs(opt = opt)
        return None
    def launch_jobs(
            self,
            opt = None):
        if not os.path.exists(os.path.join(self.work_dir, self.simname + '.h5')):
            self.write_par()
            if self.parameters['nparticles'] > 0:
                data = self.generate_tracer_state(
                        species = 0,
                        rseed = opt.particle_rand_seed)
                for s in range(1, self.particle_species):
                    self.generate_tracer_state(species = s, data = data)
            init_condition_file = os.path.join(
                    self.work_dir,
                    self.simname + '_cvorticity_i{0:0>5x}'.format(0))
            if not os.path.exists(init_condition_file):
                if len(opt.src_simname) > 0:
                    src_file = os.path.join(
                            os.path.realpath(opt.src_work_dir),
                            opt.src_simname + '_cvorticity_i{0:0>5x}'.format(opt.src_iteration))
                    os.symlink(src_file, init_condition_file)
                else:
                   self.generate_vector_field(
                           write_to_file = True,
                           spectra_slope = 2.0,
                           amplitude = 0.05)
        self.run(
                ncpu = opt.ncpu,
                njobs = opt.njobs)
        return None

