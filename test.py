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

def basic_test(
        nsteps = 8):
    nsteps_str = '{0}'.format(nsteps)
    src_txt = """
            //@begincpp
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
            //@endcpp"""
    return src_txt

class stat_test(code):
    def __init__(self, name = 'stat_test'):
        super(stat_test, self).__init__()
        self.name = name
        self.parameters['niter_todo'] = 8
        self.parameters['dt'] = 0.01
        self.variables += ('double time;\n' +
                           'dobule stats[2];\n')
        self.variables += self.cdef_pars()
        self.definitions += self.cread_pars()
        self.definitions += """
                //@begincpp
                void do_stats()
                {
                    fs->compute_velocity(fs->cvorticity);
                    stats[0] = .5*fs->correl_vec(fs->cvelocity, fs->cvelocity);
                    stats[1] = .5*fs->correl_vec(fs->cvorticity, fs->cvorticity);
                    if (myrank == fs->cd->io_myrank)
                    {
                        fwrite((void*)&fs->iteration, sizeof(int), 1, stat_file);
                        fwrite((void*)&time, sizeof(double), 1, stat_file);
                        fwrite((void*)stats, sizeof(double), 2, stat_file);
                    }
                }
                //@endcpp"""
        self.stats_dtype = np.dtype([('iteration', np.int32),
                                     ('time', np.float64),
                                     ('energy', np.float64),
                                     ('enstrophy', np.float64)])
        pickle.dump(
                self.stats_dtype,
                open(self.name + '_dtype.pickle', 'w'))
        self.main = """
                //@begincpp
                fluid_solver<float> *fs;
                fs = new fluid_solver<float>(32, 32, 32);
                FILE *stat_file;
                if (myrank == fs->cd->io_myrank)
                    stat_file = fopen("stats.bin", "wb");

                fs->cd->read(
                        "Kdata0",
                        (void*)fs->cvorticity);
                fs->low_pass_Fourier(fs->cvorticity, 3, fs->kM);
                fs->force_divfree(fs->cvorticity);
                fs->symmetrize(fs->cvorticity, 3);
                time = 0.0;
                fs->iteration = iter0;
                do_stats();
                for (int t = 0; t < niter_todo; t++)
                {
                    fs->step(dt);
                    time += dt;
                    do_stats();
                }
                fclose(stat_file);

                delete fs;
                //@endcpp"""
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type = str)
    parser.add_argument('--ncpu', dest = 'ncpu', default = 2)
    parser.add_argument('--nsteps', dest = 'nsteps', default = 8)
    parser.add_argument('-n', dest = 'n', default = 32)
    opt = parser.parse_args()

    np.random.seed(7547)
    Kdata00 = generate_data_3D(opt.n, p = 1.5).astype(np.complex64)
    Kdata01 = generate_data_3D(opt.n, p = 1.5).astype(np.complex64)
    Kdata02 = generate_data_3D(opt.n, p = 1.5).astype(np.complex64)
    Kdata0 = np.zeros(
            Kdata00.shape + (3,),
            Kdata00.dtype)
    Kdata0[..., 0] = Kdata00
    Kdata0[..., 1] = Kdata01
    Kdata0[..., 2] = Kdata02
    Kdata0.tofile("Kdata0")
    c = code()
    c.main = locals()[opt.test_name]()
    c.write_src()
    c.write_par()
    c.run(ncpu = opt.ncpu)
    dtype = pickle.load(open('stats_dtype.pickle'))
    stats = np.fromfile('stats.bin', dtype = dtype)
    fig = plt.figure(figsize = (6,6))
    a = fig.add_subplot(111)
    a.plot(stats['time'], stats['energy'])
    fig.savefig('test.pdf', format = 'pdf')

