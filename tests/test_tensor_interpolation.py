import sys
import numpy as np
from bfps import NavierStokes
import matplotlib.pyplot as plt

def main():
    c = NavierStokes()
    #c.launch(
    #        ['-n', '288',
    #         '--ncpu', '4',
    #         '--src-wd', '/home/chichi/cscratch/Lundgren_forcing/',
    #         '--src-simname', 'N0288_kMeta2',
    #         '--src-iteration', '4096',
    #         '--kMeta', '2',
    #         '--nparticles', '100000',
    #         '--sample_gradient',
    #         '--QR-stats',
    #         '--niter_out', '16',
    #         '--niter_todo', '16',
    #         '--niter_part', '4',
    #         '--niter_stat', '2'] +
    #        sys.argv[1:])
    c.read_parameters()
    c.compute_statistics()
    df = c.get_data_file()
    pf = c.get_particle_file()
    print(5*np.mean((pf['tracers0/velocity_gradient'][:, :, 0, 1] / 288**3)**2))
    print(np.mean(df['statistics/moments/velocity_gradient'][:, 2, 0, 1]))
    f = plt.figure()
    a = f.add_subplot(111)
    a.hist(
            pf['tracers0/velocity_gradient'][:, :, 0, 1].ravel(),
            histtype = 'step',
            bins = 40)
    a.set_yscale('log')
    f.tight_layout()
    f.savefig('uxx_PDF.pdf')
    return None

if __name__ == '__main__':
    main()

