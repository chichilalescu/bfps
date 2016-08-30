import sys
import os
import numpy as np
from bfps import NavierStokes
import matplotlib.pyplot as plt

def main():
    c = NavierStokes()
    if not os.path.exists(c.get_data_file_name()):
        c.launch(
            ['-n', '288',
             '--ncpu', '4',
             '--src-wd', '/home/chichi/cscratch/Lundgren_forcing/database',
             '--src-simname', 'N0288_kMeta2',
             '--src-iteration', '4096',
             '--kMeta', '2',
             '--nparticles', '100000',
             '--sample_gradient',
             '--QR-stats',
             '--niter_out', '8',
             '--niter_todo', '8',
             '--niter_part', '4',
             '--niter_stat', '2',
             '--max_trS2_estimate', '1681'] +
            sys.argv[1:])
    c.read_parameters()
    c.compute_statistics()
    df = c.get_data_file()
    pf = c.get_particle_file()
    tindex = 2
    ii = 1
    jj = 2
    print(np.mean((pf['tracers0/velocity_gradient'][tindex, :, ii, jj])**2))
    print(np.mean(df['statistics/moments/velocity_gradient'][tindex, 2, ii, jj]))
    #print(np.where(np.abs(pf['tracers0/velocity_gradient'][0, :, 0, 0]) < 1e-4)[0].shape)
    f = plt.figure()
    a = f.add_subplot(111)
    a.hist(
            pf['tracers0/velocity_gradient'][tindex, :, ii, jj],
            histtype = 'step',
            normed = True,
            bins = 129)
    bins = np.linspace(-(3*c.parameters['max_trS2_estimate'])**.5,
                        (3*c.parameters['max_trS2_estimate'])**.5,
                        c.parameters['histogram_bins']+1)
    vals = .5*(bins[1:] + bins[:-1])
    a.plot(vals,
           df['statistics/histograms/velocity_gradient'][tindex, :, ii, jj] / (
               c.parameters['nx']**3 * (bins[1] - bins[0])))
    a.set_yscale('log')
    f.tight_layout()
    f.savefig('uxx_PDF.pdf')
    return None

if __name__ == '__main__':
    main()

