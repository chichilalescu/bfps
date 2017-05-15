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



from bfps.DNS import DNS
import numpy as np
import h5py
import matplotlib.pyplot as plt

def main_fluid(
        launch = True):
    niterations = 8
    divisions_to_make = 3
    if launch:
        c = [DNS() for i in range(divisions_to_make)]
        c[0].launch(
                ['NSVE',
                 '-n', '32',
                 '--simname', 'div0',
                 '--np', '2',
                 '--ntpp', '2',
                 '--niter_todo', '{0}'.format(niterations),
                 '--niter_out', '{0}'.format(niterations),
                 '--niter_stat', '1',
                 '--wd', './'])
        for div in range(1, divisions_to_make):
            c[div].launch(
                    ['NSVE',
                     '-n', '{0}'.format(32*2**div),
                     '--simname', 'div{0}'.format(div),
                     '--src-simname', 'div0',
                     '--np', '2',
                     '--ntpp', '2',
                     '--niter_todo', '{0}'.format(niterations * 2**div),
                     '--niter_out', '{0}'.format(niterations * 2**div),
                     '--niter_stat', '{0}'.format(2**div),
                     '--wd', './'])
    # dumbest test
    # just look at moments of a field, check that they converge
    field = 'velocity'
    flist = [h5py.File('div{0}.h5'.format(div), 'r')
             for div in range(divisions_to_make)]
    err = [(np.abs(flist[div]['/statistics/moments/vorticity'][:, :, 3] -
                   flist[div-1]['/statistics/moments/vorticity'][:, :, 3]) /
            np.abs(flist[div]['/statistics/moments/vorticity'][:, :, 3]))
           for div in range(1, divisions_to_make)]
    f = plt.figure()
    a = f.add_subplot(111)
    a.plot(err[0][:, 9])
    a.plot(err[1][:, 9])
    a.set_yscale('log')
    f.tight_layout()
    f.savefig('moments.pdf')
    plt.close(f)
    # look at common Fourier amplitudes
    flist = [h5py.File('div{0}_checkpoint_0.h5'.format(div), 'r')
             for div in range(divisions_to_make)]
    err = []
    dt = []
    for div in range(1, divisions_to_make):
        n = int(32*2**(div-1))
        f0 = flist[div]['/vorticity/complex/{0}'.format(niterations*2**div)][:n//4,:n//4, :n//4]
        f1 = flist[div-1]['/vorticity/complex/{0}'.format(niterations*2**(div-1))][:n//4,:n//4, :n//4]
        good_indices = np.where(np.abs(f0) > 0)
        err.append(np.mean(np.abs((f0 - f1)[good_indices]) / np.abs(f0[good_indices])))
        dt.append(h5py.File('div{0}.h5'.format(div-1), 'r')['parameters/dt'].value)
    err = np.array(err)
    dt = np.array(dt)
    f = plt.figure()
    a = f.add_subplot(111)
    a.plot(dt, err)
    a.plot(dt, dt)
    a.set_yscale('log')
    a.set_xscale('log')
    f.tight_layout()
    f.savefig('wavenumber_evdt.pdf')
    plt.close(f)
    return None

def main_particles(
        launch = False):
    niterations = 8
    divisions_to_make = 3
    nparticles = int(1e5)
    if launch:
        c = [DNS() for i in range(divisions_to_make)]
        c[0].launch(
                ['NSVEparticles',
                 '-n', '32',
                 '--simname', 'div0',
                 '--np', '2',
                 '--ntpp', '2',
                 '--niter_todo', '{0}'.format(niterations),
                 '--niter_out', '{0}'.format(niterations),
                 '--niter_stat', '1',
                 '--nparticles', '{0}'.format(nparticles),
                 '--particle-rand-seed', '13',
                 '--wd', './'])
        for div in range(1, divisions_to_make):
            c[div].launch(
                    ['NSVEparticles',
                     '-n', '{0}'.format(32*2**div),
                     '--simname', 'div{0}'.format(div),
                     '--src-simname', 'div0',
                     '--np', '2',
                     '--ntpp', '2',
                     '--niter_todo', '{0}'.format(niterations * 2**div),
                     '--niter_out', '{0}'.format(niterations * 2**div),
                     '--niter_stat', '{0}'.format(2**div),
                     '--nparticles', '{0}'.format(nparticles),
                     '--particle-rand-seed', '13',
                     '--wd', './'])

    # check distance between particles
    flist = [h5py.File('div{0}_checkpoint_0.h5'.format(div), 'r')
             for div in range(divisions_to_make)]
    err = []
    dt = []
    for div in range(1, divisions_to_make):
        n = int(32*2**(div-1))
        p0 = flist[div]['/tracers0/state/{0}'.format(niterations*2**div)][:]
        p1 = flist[div-1]['/tracers0/state/{0}'.format(niterations*2**(div-1))][:]
        err.append(np.mean(
            np.sum((p0 - p1)**2, axis = 1)**.5 / np.sum((p0)**2, axis = 1)**.5))
        dt.append(h5py.File('div{0}.h5'.format(div-1), 'r')['parameters/dt'].value)
    err = np.array(err)
    dt = np.array(dt)
    f = plt.figure()
    a = f.add_subplot(111)
    a.plot(dt, err, marker = '.')
    a.plot(dt, 2*1e4*dt**4, dashes = (1, 1), color = 'black')
    a.set_yscale('log')
    a.set_xscale('log')
    f.tight_layout()
    f.savefig('particle_position_evdt.pdf')
    plt.close(f)
    return None

if __name__ == '__main__':
    main_particles()

