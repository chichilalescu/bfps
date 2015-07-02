#! /usr/bin/env python2

from code import code
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import argparse
import pickle

def generate_data_3D(
        n,
        dtype = np.complex128,
        p = 1.5):
    """
    generate something that has the proper shape
    """
    assert(n % 2 == 0)
    a = np.zeros((n, n, n/2+1), dtype = dtype)
    a[:] = np.random.randn(*a.shape) + 1j*np.random.randn(*a.shape)
    k, j, i = np.mgrid[-n/2+1:n/2+1, -n/2+1:n/2+1, 0:n/2+1]
    k = (k**2 + j**2 + i**2)**.5
    k = np.roll(k, n//2+1, axis = 0)
    k = np.roll(k, n//2+1, axis = 1)
    a /= k**p
    a[0, :, :] = 0
    a[:, 0, :] = 0
    a[:, :, 0] = 0
    ii = np.where(k == 0)
    a[ii] = 0
    ii = np.where(k > n/3)
    a[ii] = 0
    return a

def padd_with_zeros(
        a,
        n,
        odtype = None):
    if (type(odtype) == type(None)):
        odtype = a.dtype
    assert(a.shape[0] <= n)
    b = np.zeros((n, n, n/2 + 1) + a.shape[3:], dtype = odtype)
    m = a.shape[0]
    b[     :m/2,      :m/2, :m/2+1] = a[     :m/2,      :m/2, :m/2+1]
    b[     :m/2, n-m/2:   , :m/2+1] = a[     :m/2, m-m/2:   , :m/2+1]
    b[n-m/2:   ,      :m/2, :m/2+1] = a[m-m/2:   ,      :m/2, :m/2+1]
    b[n-m/2:   , n-m/2:   , :m/2+1] = a[m-m/2:   , m-m/2:   , :m/2+1]
    return b

def basic_test(
        nsteps = 8):
    nsteps_str = '{0}'.format(nsteps)
    src_txt = """
            //begincpp
            fluid_solver<float> *fs;
            fs = new fluid_solver<float>(32, 32, 32);
            DEBUG_MSG("fluid_solver object created\\n");

            DEBUG_MSG("nu = %g\\n", fs->nu);
            fs->cd->read(
                    "Kdata0",
                    (void*)fs->cvorticity);
            fs->low_pass_Fourier(fs->cvorticity, 3, fs->kM);
            fs->force_divfree(fs->cvorticity);
            fs->symmetrize(fs->cvorticity, 3);
            DEBUG_MSG("field read\\n");
            DEBUG_MSG(
                "######### %d %g\\n",
                fs->iteration,
                fs->correl_vec(fs->cvorticity, fs->cvorticity));
            for (int t = 0; t < """ + nsteps_str + """; t++)
            {
                fs->step(0.01);
                DEBUG_MSG(
                    "######### %d %g\\n",
                    fs->iteration,
                    fs->correl_vec(fs->cvorticity, fs->cvorticity));
            }

            delete fs;
            DEBUG_MSG("fluid_solver object deleted\\n");
            //endcpp
            """
    return src_txt

class stat_test(code):
    def __init__(self, name = 'stat_test'):
        super(stat_test, self).__init__()
        self.name = name
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
        self.variables += self.cdef_pars()
        self.definitions += self.cread_pars()
        self.definitions += """
                //begincpp
                void do_stats(fluid_solver<float> *fsolver)
                {
                    fsolver->compute_velocity(fsolver->cvorticity);
                    stats[0] = .5*fsolver->correl_vec(fsolver->cvelocity,  fsolver->cvelocity);
                    stats[1] = .5*fsolver->correl_vec(fsolver->cvorticity, fsolver->cvorticity);
                    if (myrank == fsolver->cd->io_myrank)
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
        self.main = """
                //begincpp
                fluid_solver<float> *fs;
                char fname[512];
                fs = new fluid_solver<float>(simname, nx, ny, nz);
                fs->nu = nu;
                fs->fmode = fmode;
                fs->famplitude = famplitude;
                fs->iteration = iter0;
                if (myrank == fs->cd->io_myrank)
                    {
                        sprintf(fname, "%s_stats.bin", simname);
                        stat_file = fopen(fname, "wb");
                    }
                fs->read('v', 'c');
                fs->low_pass_Fourier(fs->cvorticity, 3, fs->kM);
                fs->force_divfree(fs->cvorticity);
                fs->symmetrize(fs->cvorticity, 3);
                t = 0.0;
                do_stats(fs);
                fs->write('u', 'r');
                fs->write('v', 'r');
                for (; fs->iteration < iter0 + niter_todo;)
                {
                    fs->step(dt);
                    t += dt;
                    do_stats(fs);
                }
                fclose(stat_file);
                fs->write('v', 'c');
                fs->write('v', 'r');
                fs->write('u', 'r');
                fs->write('u', 'c');
                delete fs;
                //endcpp
                """
        return None
    def get_coord(self, direction):
        assert(direction == 'x' or direction == 'y' or direction == 'z')
        return np.arange(.0, self.parameters['n' + direction])*2*np.pi / self.parameters['n' + direction]
    def plot_vel_cut(
            self,
            axis,
            simname = 'test',
            field = 'velocity',
            iteration = 0):
        axis.set_axis_off()
        Rdata0 = np.fromfile(
                simname + '_r' + field + '_i{0:0>5x}'.format(iteration),
                dtype = np.float32).reshape((self.parameters['nz'],
                                             self.parameters['ny'],
                                             self.parameters['nx'], 3))
        energy = np.sum(Rdata0[13, :, :, :]**2, axis = 2)*.5
        axis.imshow(energy, interpolation='none')
        axis.set_title('{0}'.format(np.average(Rdata0[..., 0]**2 +
                                               Rdata0[..., 1]**2 +
                                               Rdata0[..., 2]**2)*.5))
        return Rdata0

def convergence_test(opt):
    c = stat_test(name = opt.test_name)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['nu'] = 1e-1
    c.parameters['dt'] = 4e-3
    c.parameters['niter_todo'] = opt.nsteps
    c.parameters['famplitude'] = 0.0
    if opt.run:
        np.random.seed(7547)
        Kdata00 = generate_data_3D(opt.n/2, p = 1.).astype(np.complex64)
        Kdata01 = generate_data_3D(opt.n/2, p = 1.).astype(np.complex64)
        Kdata02 = generate_data_3D(opt.n/2, p = 1.).astype(np.complex64)
        Kdata0 = np.zeros(
                Kdata00.shape + (3,),
                Kdata00.dtype)
        Kdata0[..., 0] = Kdata00
        Kdata0[..., 1] = Kdata01
        Kdata0[..., 2] = Kdata02
        Kdata1 = padd_with_zeros(Kdata0, opt.n)
        Kdata1.tofile("test1_cvorticity_i00000")
        c.write_src()
        c.write_par(simname = 'test1')
        c.run(ncpu = opt.ncpu, simname = 'test1')
        Rdata = np.fromfile(
                'test1_rvorticity_i00000',
                dtype = np.float32).reshape(opt.n, opt.n, opt.n, 3)
        tdata = Rdata.transpose(3, 0, 1, 2).copy()
        tdata.tofile('input_for_vortex')
        c.parameters['dt'] /= 2
        c.parameters['niter_todo'] *= 2
        c.parameters['nx'] *= 2
        c.parameters['ny'] *= 2
        c.parameters['nz'] *= 2
        c.write_par(simname = 'test2')
        Kdata2 = padd_with_zeros(Kdata0, opt.n*2)
        Kdata2.tofile("test2_cvorticity_i00000")
        c.run(ncpu = opt.ncpu, simname = 'test2')
    dtype = pickle.load(open(opt.test_name + '_dtype.pickle'))
    stats1 = np.fromfile('test1_stats.bin', dtype = dtype)
    stats2 = np.fromfile('test2_stats.bin', dtype = dtype)
    stats_vortex = np.loadtxt('../vortex/sim_000000.log')
    fig = plt.figure(figsize = (12,6))
    a = fig.add_subplot(121)
    a.plot(stats1['t'], stats1['energy'])
    a.plot(stats2['t'], stats2['energy'])
    a.plot(stats_vortex[:, 2], stats_vortex[:, 3])
    a = fig.add_subplot(122)
    a.plot(stats1['t'], stats1['enstrophy'])
    a.plot(stats2['t'], stats2['enstrophy'])
    a.plot(stats_vortex[:, 2], stats_vortex[:, 9]/2)
    fig.savefig('test.pdf', format = 'pdf')

    #fig = plt.figure(figsize=(12, 12))
    #a = fig.add_subplot(221)
    #c.plot_vel_cut(
    #        a,
    #        simname = 'test1',
    #        field = 'velocity',
    #        iteration = 0)
    #a = fig.add_subplot(222)
    #a.set_axis_off()
    #c.plot_vel_cut(
    #        a,
    #        simname = 'test2',
    #        field = 'velocity',
    #        iteration = 0)
    #a = fig.add_subplot(223)
    #c.plot_vel_cut(
    #        a,
    #        simname = 'test1',
    #        field = 'velocity',
    #        iteration = stats1.shape[0] - 1)
    #a = fig.add_subplot(224)
    #c.plot_vel_cut(
    #        a,
    #        simname = 'test2',
    #        field = 'velocity',
    #        iteration = stats2.shape[0] - 1)
    #fig.savefig('vel_cut.pdf', format = 'pdf')
    return None

def Kolmogorov_flow_test(opt):
    c = stat_test(name = opt.test_name)
    c.parameters['nx'] = opt.n
    c.parameters['ny'] = opt.n
    c.parameters['nz'] = opt.n
    c.parameters['nu'] = 1.0
    c.parameters['dt'] = 0.02
    c.parameters['niter_todo'] = opt.nsteps
    if opt.run:
        np.random.seed(7547)
        Kdata00 = generate_data_3D(opt.n, p = 1.).astype(np.complex64)
        Kdata01 = generate_data_3D(opt.n, p = 1.).astype(np.complex64)
        Kdata02 = generate_data_3D(opt.n, p = 1.).astype(np.complex64)
        Kdata0 = np.zeros(
                Kdata00.shape + (3,),
                Kdata00.dtype)
        Kdata0[..., 0] = Kdata00
        Kdata0[..., 1] = Kdata01
        Kdata0[..., 2] = Kdata02
        c.write_src()
        c.write_par(simname = 'test')
        Kdata0.tofile("test_cvorticity_i00000")
        c.run(ncpu = opt.ncpu, simname = 'test')
        Rdata = np.fromfile(
                'test_rvorticity_i00000',
                dtype = np.float32).reshape(opt.n, opt.n, opt.n, 3)
        tdata = Rdata.transpose(3, 0, 1, 2).copy()
    dtype = pickle.load(open(opt.test_name + '_dtype.pickle'))
    stats = np.fromfile('test_stats.bin', dtype = dtype)
    fig = plt.figure(figsize = (12,6))
    a = fig.add_subplot(121)
    a.plot(stats['t'], stats['energy'])
    a = fig.add_subplot(122)
    a.plot(stats['t'], stats['enstrophy'])
    fig.savefig('test.pdf', format = 'pdf')

    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(221)
    c.plot_vel_cut(a,
            simname = 'test',
            field = 'velocity',
            iteration = 0)
    a = fig.add_subplot(222)
    a.set_axis_off()
    c.plot_vel_cut(a,
            simname = 'test',
            field = 'vorticity',
            iteration = 0)
    a = fig.add_subplot(223)
    ufin = c.plot_vel_cut(a,
            simname = 'test',
            field = 'velocity',
            iteration = stats.shape[0]-1)
    a = fig.add_subplot(224)
    vfin = c.plot_vel_cut(a,
            simname = 'test',
            field = 'vorticity',
            iteration = stats.shape[0]-1)
    fig.savefig('field_cut.pdf', format = 'pdf')

    fig = plt.figure()
    a = fig.add_subplot(111)
    ycoord = c.get_coord('y')
    a.plot(ycoord, ufin[0, :, 0, 0])
    amp = c.parameters['famplitude'] / (c.parameters['fmode']**2 * c.parameters['nu'])
    a.plot(ycoord, amp*np.sin(ycoord), dashes = (2, 2))
    a.plot(ycoord, vfin[0, :, 0, 2])
    a.plot(ycoord, -np.cos(ycoord), dashes = (2, 2))
    a.set_xticks(np.linspace(.0, 2*np.pi, 9))
    a.set_xlim(.0, 2*np.pi)
    a.grid()
    fig.savefig('ux_vs_y.pdf', format = 'pdf')

    fig = plt.figure(figsize=(12, 12))
    tmp = np.fromfile('test_cvelocity_i{0:0>5x}'.format(stats.shape[0]-1),
                      dtype = np.complex64).reshape(opt.n, opt.n, opt.n/2+1, 3)
    a = fig.add_subplot(321)
    a.plot(np.sum(np.abs(tmp), axis = (1, 2)))
    a.set_yscale('log')
    a = fig.add_subplot(323)
    a.plot(np.sum(np.abs(tmp), axis = (0, 2)))
    a.set_yscale('log')
    a = fig.add_subplot(325)
    a.plot(np.sum(np.abs(tmp), axis = (0, 1)))
    a.set_yscale('log')
    a = fig.add_subplot(322)
    tmp = np.fromfile('test_cvorticity_i{0:0>5x}'.format(stats.shape[0]-1),
                      dtype = np.complex64).reshape(opt.n, opt.n, opt.n/2+1, 3)
    a.plot(np.sum(np.abs(tmp), axis = (1, 2)))
    a.set_yscale('log')
    a = fig.add_subplot(324)
    a.plot(np.sum(np.abs(tmp), axis = (0, 2)))
    a.set_yscale('log')
    a = fig.add_subplot(326)
    a.plot(np.sum(np.abs(tmp), axis = (0, 1)))
    a.set_yscale('log')
    fig.savefig('cvort.pdf', format = 'pdf')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type = str)
    parser.add_argument('--run', dest = 'run', action = 'store_true')
    parser.add_argument('--ncpu', type = int, dest = 'ncpu', default = 2)
    parser.add_argument('--nsteps', type = int, dest = 'nsteps', default = 16)
    parser.add_argument('-n', type = int, dest = 'n', default = 32)
    opt = parser.parse_args()
    convergence_test(opt)

