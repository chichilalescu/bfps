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

class convergence_test(bfps.code):
    def __init__(self, name = 'convergence_test'):
        super(convergence_test, self).__init__()
        self.name = name
        self.parameters['niter_todo'] = 8
        self.parameters['dt'] = 0.01
        self.parameters['nu'] = 0.1
        self.parameters['famplitude'] = 1.0
        self.parameters['fmode'] = 1
        self.parameters['nparticles'] = 1
        self.includes += '#include <cstring>\n'
        self.includes += '#include "fftw_tools.hpp"\n'
        self.includes += '#include "tracers.hpp"\n'
        self.variables += ('double t;\n' +
                           'FILE *stat_file;\n'
                           'FILE *traj_file;\n'
                           'double stats[2];\n')
        self.variables += self.cdef_pars()
        self.definitions += self.cread_pars()
        self.definitions += """
                //begincpp
                void do_stats(fluid_solver<float> *fsolver,
                              tracers<float> *tracers)
                {
                    fsolver->compute_velocity(fsolver->cvorticity);
                    stats[0] = .5*fsolver->correl_vec(fsolver->cvelocity,  fsolver->cvelocity);
                    stats[1] = .5*fsolver->correl_vec(fsolver->cvorticity, fsolver->cvorticity);
                    if (myrank == 0)
                    {
                        fwrite((void*)&fsolver->iteration, sizeof(int), 1, stat_file);
                        fwrite((void*)&t, sizeof(double), 1, stat_file);
                        fwrite((void*)stats, sizeof(double), 2, stat_file);
                        fwrite((void*)tracers->state, sizeof(double), tracers->array_size, traj_file);
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
        self.main = """
                //begincpp
                fluid_solver<float> *fs;
                tracers<float> *ps;
                char fname[512];
                fs = new fluid_solver<float>(simname, nx, ny, nz);
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->iteration = iter0;
                fs->read('v', 'c');
                fs->low_pass_Fourier(fs->cvorticity, 3, fs->kM);
                fs->force_divfree(fs->cvorticity);
                fs->symmetrize(fs->cvorticity, 3);
                sprintf(fname, "%s_tracers", simname);
                ps = new tracers<float>(
                        fname, fs,
                        nparticles, 2,
                        fs->ru);
                ps->dt = dt;
                ps->iteration = iter0;
                ps->read();
                fs->compute_velocity(fs->cvorticity);
                fftwf_execute(*((fftwf_plan*)fs->c2r_velocity));
                ps->update_field();
                sprintf(fname, "%s_particle_field_i00000", simname);
                ps->buffered_field_descriptor->write(fname, ps->data);
                t = 0.0;
                if (myrank == 0)
                {
                    sprintf(fname, "%s_stats.bin", simname);
                    stat_file = fopen(fname, "wb");
                    sprintf(fname, "%s_traj.bin", ps->name);
                    traj_file = fopen(fname, "wb");
                }
                do_stats(fs, ps);
                fs->write('u', 'r');
                fs->write('v', 'r');
                for (; fs->iteration < iter0 + niter_todo;)
                {
                    fs->step(dt);
                    t += dt;
                    fs->compute_velocity(fs->cvorticity);
                    fftwf_execute(*((fftwf_plan*)fs->c2r_velocity));
                    ps->update_field();
                    ps->Euler();
                    ps->iteration++;
                    ps->synchronize();
                    do_stats(fs, ps);
                }
                if (myrank == 0)
                {
                    fclose(stat_file);
                    fclose(traj_file);
                }
                fs->write('v', 'c');
                fs->write('v', 'r');
                fs->write('u', 'r');
                fs->write('u', 'c');
                ps->write();
                delete ps;
                delete fs;
                //endcpp
                """
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
    def execute(
            self,
            rseed = 7547,
            ncpu = 2,
            particle_rseed = 3281):
        assert(self.parameters['nx'] == self.parameters['ny'] == self.parameters['nz'])
        np.random.seed(particle_rseed)
        tracer_state = np.random.random(self.parameters['nparticles']*3)*2*np.pi
        tracer_state.tofile('test1_tracers_state_i00000')
        tracer_state.tofile('test2_tracers_state_i00000')
        np.random.seed(rseed)
        Kdata00 = bfps.tools.generate_data_3D(self.parameters['nx']/2, p = 1.).astype(np.complex64)
        Kdata01 = bfps.tools.generate_data_3D(self.parameters['nx']/2, p = 1.).astype(np.complex64)
        Kdata02 = bfps.tools.generate_data_3D(self.parameters['nx']/2, p = 1.).astype(np.complex64)
        Kdata0 = np.zeros(
                Kdata00.shape + (3,),
                Kdata00.dtype)
        Kdata0[..., 0] = Kdata00
        Kdata0[..., 1] = Kdata01
        Kdata0[..., 2] = Kdata02
        Kdata1 = bfps.tools.padd_with_zeros(Kdata0, self.parameters['nx'])
        Kdata1.tofile("test1_cvorticity_i00000")
        self.write_src()
        self.write_par(simname = 'test1')
        self.run(ncpu = ncpu, simname = 'test1')
        Rdata = np.fromfile(
                'test1_rvorticity_i00000',
                dtype = np.float32).reshape(self.parameters['nz'],
                                            self.parameters['ny'],
                                            self.parameters['nx'], 3)
        tdata = Rdata.transpose(3, 0, 1, 2).copy()
        tdata.tofile('input_split_per_component')
        self.parameters['dt'] /= 2
        self.parameters['niter_todo'] *= 2
        self.parameters['nx'] *= 2
        self.parameters['ny'] *= 2
        self.parameters['nz'] *= 2
        self.write_par(simname = 'test2')
        Kdata2 = bfps.tools.padd_with_zeros(Kdata0, self.parameters['nx'])
        Kdata2.tofile("test2_cvorticity_i00000")
        self.run(ncpu = ncpu, simname = 'test2')
        self.parameters['dt'] *= 2
        self.parameters['niter_todo'] /= 2
        self.parameters['nx'] /= 2
        self.parameters['ny'] /= 2
        self.parameters['nz'] /= 2
        return None

