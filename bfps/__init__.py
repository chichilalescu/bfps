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
import sys
import pickle

import pkg_resources

__version__ = pkg_resources.require('bfps')[0].version

_dist = pkg_resources.get_distribution('bfps')
dist_loc = os.path.realpath(_dist.location)
here = os.path.normcase(__file__)
header_dir = os.path.join(os.path.join(dist_loc, 'bfps'), 'cpp')
lib_dir = os.path.join(dist_loc, 'bfps')

install_info = pickle.load(
        open(os.path.join(os.path.dirname(here),
                          'install_info.pickle'),
             'rb'))

homefolder = os.path.expanduser('~')
bfpsfolder = os.path.join(homefolder, '.config/', 'bfps')
sys.path.append(bfpsfolder)
from host_information import host_info

from .code import code
from .fluid_converter import fluid_converter
from .fluid_resize import fluid_resize
from .NavierStokes import NavierStokes

