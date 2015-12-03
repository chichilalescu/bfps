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



import os
import subprocess
import pickle

from pkg_resources import get_distribution, DistributionNotFound

try:
    _dist = get_distribution('bfps')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(os.path.realpath(_dist.location))
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'bfps')):
        # not installed, but there is another version that *is*
        header_dir = os.path.join(os.path.dirname(here), 'cpp')
        lib_dir = os.path.join(os.path.dirname(here), os.pardir)
        raise DistributionNotFound
    header_dir = os.path.join(os.path.join(dist_loc, 'bfps'), 'cpp')
    lib_dir = _dist.location
    __version__ = _dist.version
except DistributionNotFound:
    __version__ = ''

install_info = pickle.load(
        open(os.path.join(os.path.dirname(here),
                          'install_info.pickle'),
             'r'))

from .code import code
from .fluid_converter import fluid_converter
from .fluid_resize import fluid_resize
from .NavierStokes import NavierStokes

import argparse

def get_parser(base_class = NavierStokes,
               n = 32,
               ncpu = 2,
               precision = 'single',
               simname = 'test',
               work_dir = './',
               njobs = 1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', dest = 'run', action = 'store_true')
    parser.add_argument('-n',
            type = int, dest = 'n',
            default = n)
    parser.add_argument('--ncpu',
            type = int, dest = 'ncpu',
            default = ncpu)
    parser.add_argument('--precision',
            type = str, dest = 'precision',
            default = precision)
    parser.add_argument('--simname',
            type = str, dest = 'simname',
            default = simname)
    parser.add_argument('--wd',
            type = str, dest = 'work_dir',
            default = work_dir)
    parser.add_argument('--njobs',
            type = int, dest = 'njobs',
            default = njobs)
    c = base_class(simname = simname)
    for k in sorted(c.parameters.keys()):
        parser.add_argument(
                '--{0}'.format(k),
                type = type(c.parameters[k]),
                dest = k,
                default = c.parameters[k])
    return parser

