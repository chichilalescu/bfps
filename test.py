#! /usr/bin/env python2

from code import code
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import argparse
import pickle

def run_test(
        test_code = '\n;',
        test_name = 'base_test',
        ncpu = 4):
    c = code()
    # first, write file
    with open('src/' + test_name + '.cpp', 'w') as outfile:
        outfile.write(c.main_start)
        outfile.write(test_code)
        outfile.write(c.main_end)

    # now compile code and run
    if subprocess.call(['make', test_name + '.elf']) == 0:
        subprocess.call(['time',
                         'mpirun',
                         '-np',
                         '{0}'.format(ncpu),
                         './' + test_name + '.elf'])
    return None

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

def stat_test(
        nsteps = 8):
    nsteps_str = '{0}'.format(nsteps)
    stats_dtype = np.dtype([('iteration', np.int32),
                            ('time', np.float64),
                            ('correl', np.float64)])
    pickle.dump(
            stats_dtype,
            open('stats_dtype.pickle', 'w'))
    src_txt = """
            //@begincpp
            fluid_solver<float> *fs;
            fs = new fluid_solver<float>(32, 32, 32);
            FILE *stat_file;
            if (myrank == fs->cd->io_myrank)
                stat_file = fopen("stats.bin", "wb");
            double stats[3];
            double dt = 0.01;

            fs->cd->read(
                    "Kdata0",
                    (void*)fs->cvorticity);
            fs->low_pass_Fourier(fs->cvorticity, 3, fs->kM);
            fs->force_divfree(fs->cvorticity);
            fs->symmetrize(fs->cvorticity, 3);
            stats[0] = 0.0;
            stats[1] = fs->correl_vec(fs->cvorticity, fs->cvorticity);
            if (myrank == fs->cd->io_myrank)
            {
                fwrite((void*)&fs->iteration, sizeof(int), 1, stat_file);
                fwrite((void*)stats, sizeof(double), 2, stat_file);
            }
            for (int t = 0; t < """ + nsteps_str + """; t++)
            {
                fs->step(dt);
                stats[0] += dt;
                stats[1] = fs->correl_vec(fs->cvorticity, fs->cvorticity);
                if (myrank == fs->cd->io_myrank)
                {
                    fwrite((void*)&fs->iteration, sizeof(int), 1, stat_file);
                    fwrite((void*)stats, sizeof(double), 2, stat_file);
                }
            }
            fclose(stat_file);

            delete fs;
            //@endcpp"""
    return src_txt

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
    run_test(
            test_code = locals()[opt.test_name](
                    nsteps = opt.nsteps),
            test_name = opt.test_name,
            ncpu = opt.ncpu)
    dtype = pickle.load(open('stats_dtype.pickle'))
    stats = np.fromfile('stats.bin', dtype = dtype)
    print stats

