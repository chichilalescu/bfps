#! /usr/bin/env python2
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

from test_base import *

from test_frozen_field import FrozenFieldParticles
from test_convergence import convergence_test

# use ABC flow

def generate_ABC_flow(
        parameters = {'nx': 32,
                      'ny': 32,
                      'nz': 32},
        Fmode = 1,
        Famp = 1.0,
        dtype = np.complex64):
    Kdata = np.zeros((parameters['ny'],
                      parameters['nz'],
                      parameters['nx']//2+1,
                      3),
                     dtype = dtype)

    Kdata[                   Fmode, 0, 0, 0] =  Famp/2.
    Kdata[                   Fmode, 0, 0, 2] = -Famp/2.*1j
    Kdata[parameters['ny'] - Fmode, 0, 0, 0] =  Famp/2.
    Kdata[parameters['ny'] - Fmode, 0, 0, 2] =  Famp/2.*1j

    Kdata[0,                    Fmode, 0, 0] = -Famp/2.*1j
    Kdata[0,                    Fmode, 0, 1] =  Famp/2.
    Kdata[0, parameters['nz'] - Fmode, 0, 0] =  Famp/2.*1j
    Kdata[0, parameters['nz'] - Fmode, 0, 1] =  Famp/2.

    Kdata[0, 0,                    Fmode, 1] = -Famp/2.*1j
    Kdata[0, 0,                    Fmode, 2] =  Famp/2.
    return Kdata

if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.precision == 'single':
        dtype = np.complex64
    elif opt.precision == 'double':
        dtype = np.complex128
    Kdata = generate_ABC_flow(
            parameters = {'nx': opt.n,
                          'ny': opt.n,
                          'nz': opt.n},
            dtype = dtype)
    convergence_test(
            opt,
            launch,
            code_class = FrozenFieldParticles,
            init_vorticity = Kdata)

