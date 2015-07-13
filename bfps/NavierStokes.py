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
import numpy as np
import pickle

class NavierStokes(bfps.code):
    def __init__(
            self,
            name = 'NavierStokes',
            particles_on = False):
        super(NavierStokes, self).__init__()
        self.particles_on = particles_on
        self.name = name
        self.parameters['dkx'] = 1.0
        self.parameters['dky'] = 1.0
        self.parameters['dkz'] = 1.0
        self.parameters['niter_todo'] = 8
        self.parameters['dt'] = 0.01
        self.parameters['nu'] = 0.1
        self.parameters['famplitude'] = 1.0
        self.parameters['fmode'] = 1
        self.includes += '#include <cstring>\n'
        self.includes += '#include "fftw_tools.hpp"\n'
        self.variables += ('double t;\n' +
                           'FILE *stat_file;\n'
                           'double stats[2];\n')
        self.definitions += """
                //begincpp
                void do_stats(fluid_solver<float> *fsolver)
                {
                    fsolver->compute_velocity(fsolver->cvorticity);
                    stats[0] = .5*fsolver->correl_vec(fsolver->cvelocity,  fsolver->cvelocity);
                    stats[1] = .5*fsolver->correl_vec(fsolver->cvorticity, fsolver->cvorticity);
                    if (myrank == 0)
                    {
                        fwrite((void*)&fsolver->iteration, sizeof(int), 1, stat_file);
                        fwrite((void*)&t, sizeof(double), 1, stat_file);
                        fwrite((void*)stats, sizeof(double), 2, stat_file);
                    }
                }
                //endcpp
                """
        self.stats_dtype = np.dtype([('iteration', np.int32),
                                     ('t', np.float64),
                                     ('energy', np.float64),
                                     ('enstrophy', np.float64)])
        pickle.dump(
                self.stats_dtype,
                open(self.name + '_dtype.pickle', 'w'))
        if self.particles_on:
            self.parameters['nparticles'] = 1
            self.includes += '#include "tracers.hpp"\n'
            self.variables += 'FILE *traj_file;\n'
            self.definitions += """
                    //begincpp
                    void out_traj(tracers<float> *tracers)
                    {
                        if (myrank == 0)
                        {
                            fwrite((void*)tracers->state, sizeof(double), tracers->array_size, traj_file);
                        }
                    }
                    //endcpp
                    """
        self.variables += self.cdef_pars()
        self.definitions += self.cread_pars()
        self.main = """
                //begincpp
                fluid_solver<float> *fs;
                tracers<float> *ps;
                char fname[512];
                fs = new fluid_solver<float>(
                        simname,
                        nx, ny, nz,
                        dkx, dky, dkz);
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->iteration = iter0;
                fs->read('v', 'c');
                fs->low_pass_Fourier(fs->cvorticity, 3, fs->kM);
                fs->force_divfree(fs->cvorticity);
                fs->symmetrize(fs->cvorticity, 3);
                if (myrank == 0)
                {
                    sprintf(fname, "%s_stats.bin", simname);
                    stat_file = fopen(fname, "wb");
                }
                t = 0.0;
                //endcpp
                """
        if self.particles_on:
            self.main += """
                    //begincpp
                    sprintf(fname, "%s_tracers", simname);
                    ps = new tracers<float>(
                            fname, fs,
                            nparticles, 1, 1,
                            fs->ru);
                    ps->dt = dt;
                    ps->iteration = iter0;
                    ps->read();
                    fs->compute_velocity(fs->cvorticity);
                    fftwf_execute(*((fftwf_plan*)fs->c2r_velocity));
                    ps->update_field();
                    if (myrank == 0)
                    {
                        sprintf(fname, "%s_traj.bin", ps->name);
                        traj_file = fopen(fname, "wb");
                    }
                    out_traj(ps);
                    //endcpp
                    """
        self.main += """
                //begincpp
                do_stats(fs);
                fs->write_spectrum("velocity", fs->cvelocity);
                for (; fs->iteration < iter0 + niter_todo;)
                {
                    fs->step(dt);
                    t += dt;
                    do_stats(fs);
                //endcpp
                """
        if self.particles_on:
            self.main += """
                //begincpp
                        fs->compute_velocity(fs->cvorticity);
                        fftwf_execute(*((fftwf_plan*)fs->c2r_velocity));
                        ps->update_field();
                        ps->Euler();
                        ps->iteration++;
                        ps->synchronize();
                //endcpp
                    """
        self.main += """
                //begincpp
                }
                if (myrank == 0)
                {
                    fclose(stat_file);
                }
                fs->write('v', 'c');
                fs->write_spectrum("velocity", fs->cvelocity);
                //endcpp
                """
        if self.particles_on:
            self.main += """
                    ps->write();
                    delete ps;
                    """
        self.main += 'delete fs;\n'
        return None
    def plot_vel_cut(
            self,
            axis,
            simname = 'test',
            field = 'velocity',
            iteration = 0,
            yval = 13,
            filename = None):
        axis.set_axis_off()
        if type(filename) == type(None):
            filename = simname + '_' + field + '_i{0:0>5x}'.format(iteration)
        Rdata0 = np.fromfile(
                filename,
                dtype = np.float32).reshape((-1,
                                             self.parameters['ny'],
                                             self.parameters['nx'], 3))
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
            precision = 'single',
            simname = None,
            iteration = 0):
        if precision == 'single':
            dtype = np.complex64
        elif precision == 'double':
            dtype = np.complex128
        np.random.seed(rseed)
        Kdata00 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(dtype)
        Kdata01 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(dtype)
        Kdata02 = bfps.tools.generate_data_3D(
                self.parameters['nz']/2,
                self.parameters['ny']/2,
                self.parameters['nx']/2,
                p = spectra_slope).astype(dtype)
        Kdata0 = np.zeros(
                Kdata00.shape + (3,),
                Kdata00.dtype)
        Kdata0[..., 0] = Kdata00
        Kdata0[..., 1] = Kdata01
        Kdata0[..., 2] = Kdata02
        Kdata1 = bfps.tools.padd_with_zeros(
                Kdata0,
                self.parameters['nz'],
                self.parameters['ny'],
                self.parameters['nx'])
        if not (type(simname) == type(None)):
            Kdata1.tofile(simname + "_cvorticity_i{0:0>5x}".format(iteration))
        return Kdata1
    def generate_tracer_state(
            self,
            rseed = 34982,
            simname = None,
            iteration = 0):
        data = np.random.random(self.parameters['nparticles']*3)*2*np.pi
        if not (type(simname) == type(None)):
            data.tofile(simname + "_tracers_state_i{0:0>5x}".format(iteration))
        return data

import subprocess

def test(opt):
    if opt.run or opt.clean:
        subprocess.call(['rm test_*'], shell = True)
        subprocess.call(['rm *.pickle'], shell = True)
    if opt.clean:
        subprocess.call(['make', 'clean'])
        return None
    c = NavierStokes(particles_on = True)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['nu'] = 1e-1
    c.parameters['dt'] = 2e-3
    c.parameters['niter_todo'] = opt.nsteps
    c.parameters['famplitude'] = 0.0
    c.parameters['nparticles'] = 32
    c.write_src()
    c.write_par(simname = 'test')
    c.generate_vector_field(simname = 'test')
    c.generate_tracer_state(simname = 'test')
    if opt.run:
        c.run(ncpu = opt.ncpu,
              simname = 'test')
    return None

