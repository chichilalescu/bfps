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

import subprocess

from pkg_resources import get_distribution, DistributionNotFound

try:
    _dist = get_distribution('bfps')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'bfps')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    #__version__ = 'Please install this project with setup.py'
    import subprocess
    __version__ = 'git revision ' + subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
else:
    __version__ = _dist.version

from .code import code
