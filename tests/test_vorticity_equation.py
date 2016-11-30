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
            ['-n', '72',
             '--simname', 'fluid_solver',
             '--ncpu', '4',
             '--niter_todo', '128',
             '--niter_out', '32',
             '--niter_stat', '1',
             '--wd', './'] +
            sys.argv[1:])
    data = c.read_cfield(iteration = 32)
    f = h5py.File('vorticity_equation_cvorticity_i00000.h5', 'w')
    f['vorticity/complex/0'] = data
    f.close()
    c = bfps.NSVorticityEquation()
    c.launch(
            ['-n', '72',
             '--simname', 'vorticity_equation',
             '--ncpu', '4',
             '--niter_todo', '128',
             '--niter_out', '32',
             '--niter_stat', '1',
             '--wd', './'] +
            sys.argv[1:])
    c0 = NSReader(simname = 'fluid_solver')
    c1 = NSReader(simname = 'vorticity_equation')
    print(c0.statistics['enstrophy(t)'])
    print(c1.statistics['enstrophy(t)'])
    c0.do_plots()
    c1.do_plots()
    return None

if __name__ == '__main__':
    main()

