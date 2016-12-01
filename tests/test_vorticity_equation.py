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
import bfps.tools

from bfps_addons import NSReader
import matplotlib.pyplot as plt

def main():
    c = bfps.NavierStokes()
    c.launch(
            ['-n', '24',
             '--simname', 'fluid_solver',
             '--ncpu', '2',
             '--niter_todo', '2',
             '--niter_out', '2',
             '--niter_stat', '1',
             '--wd', './'] +
            sys.argv[1:])
    data = c.read_cfield(iteration = 0)
    f = plt.figure(figsize = (6,6))
    a = f.add_axes([.0, .0, 1., 1.])
    a.set_axis_off()
    a.imshow(np.log(np.abs(data[:, :, 0, 0])),
             cmap = 'viridis',
             interpolation = 'none')
    f.savefig('figs/cut.pdf')
    f = h5py.File('vorticity_equation_cvorticity_i00000.h5', 'w')
    f['vorticity/complex/0'] = data
    f.close()
    c = bfps.NSVorticityEquation()
    c.launch(
            ['-n', '24',
             '--simname', 'vorticity_equation',
             '--ncpu', '2',
             '--niter_todo', '2',
             '--niter_out', '2',
             '--niter_stat', '1',
             '--wd', './'] +
            sys.argv[1:])
    c0 = NSReader(simname = 'fluid_solver')
    c1 = NSReader(simname = 'vorticity_equation')
    df0 = c0.get_data_file()
    df1 = c1.get_data_file()
    f = plt.figure(figsize=(6,10))
    a = f.add_subplot(211)
    a.plot(df0['statistics/moments/vorticity'][:, 2, 3],
           color = 'blue',
           marker = '.')
    a.plot(df1['statistics/moments/vorticity'][:, 2, 3],
           color = 'red',
           marker = '.')
    a = f.add_subplot(212)
    a.plot(df0['statistics/moments/velocity'][:, 2, 3],
           color = 'blue',
           marker = '.')
    a.plot(df1['statistics/moments/velocity'][:, 2, 3],
           color = 'red',
           marker = '.')
    f.tight_layout()
    f.savefig('figs/moments.pdf')
    f = plt.figure(figsize = (6, 10))
    a = f.add_subplot(111)
    a.plot(c0.statistics['enstrophy(t, k)'][0])
    a.plot(c1.statistics['enstrophy(t, k)'][0])
    a.set_yscale('log')
    f.tight_layout()
    f.savefig('figs/spectra.pdf')
    f = h5py.File('vorticity_equation_cvorticity_i00000.h5', 'r')
    data = c0.read_cfield()
    print(np.max(np.abs(f['vorticity/complex/0'].value - data)))
    #print(c0.statistics['enstrophy(t, k)'][0])
    #print(c1.statistics['enstrophy(t, k)'][0])
    #c0.do_plots()
    #c1.do_plots()
    return None

if __name__ == '__main__':
    main()

