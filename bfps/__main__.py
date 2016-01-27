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
import argparse

import bfps
from .NavierStokes import NavierStokes
from .FluidResize import FluidResize

def main():
    parser = argparse.ArgumentParser(prog = 'bfps')
    parser.add_argument(
            '-v', '--version',
            action = 'version',
            version = '%(prog)s ' + bfps.__version__)
    NSoptions = ['NavierStokes',
                 'NavierStokes-single',
                 'NavierStokes-double',
                 'NS',
                 'NS-single',
                 'NS-double']
    FRoptions = ['FluidResize',
                 'FluidResize-single',
                 'FluidResize-double',
                 'FR',
                 'FR-single',
                 'FR-double']
    parser.add_argument(
            'base_class',
            choices = NSoptions + FRoptions,
            type = str)
    # first option is the choice of base class or -h or -v
    # all other options are passed on to the base_class instance
    opt = parser.parse_args(sys.argv[1:2])
    # error is thrown if first option is not a base class, so launch
    # cannot be executed by mistake.
    if 'double' in opt.base_class:
        precision = 'double'
    else:
        precision = 'single'
    if opt.base_class in NSoptions:
        base_class = NavierStokes
    elif opt.base_class in FRoptions:
        base_class = FluidResize
    c = base_class(fluid_precision = precision)
    c.launch(args = sys.argv[2:])
    return None

if __name__ == '__main__':
    main()

