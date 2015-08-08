########################################################################
#
#  Copyright 2015 Max Planck Institute for Dynamics and SelfOrganization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: Cristian.Lalescu@ds.mpg.de
#
########################################################################

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
from .test_io import test_io
from .NavierStokes import NavierStokes

