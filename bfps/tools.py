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



import numpy as np

def generate_data_3D(
        n,
        dtype = np.complex128,
        p = 1.5):
    """
    generate something that has the proper shape
    """
    assert(n % 2 == 0)
    a = np.zeros((n, n, n/2+1), dtype = dtype)
    a[:] = np.random.randn(*a.shape) + 1j*np.random.randn(*a.shape)
    k, j, i = np.mgrid[-n/2+1:n/2+1, -n/2+1:n/2+1, 0:n/2+1]
    k = (k**2 + j**2 + i**2)**.5
    k = np.roll(k, n//2+1, axis = 0)
    k = np.roll(k, n//2+1, axis = 1)
    a /= k**p
    a[0, :, :] = 0
    a[:, 0, :] = 0
    a[:, :, 0] = 0
    ii = np.where(k == 0)
    a[ii] = 0
    ii = np.where(k > n/3)
    a[ii] = 0
    return a

def padd_with_zeros(
        a,
        n,
        odtype = None):
    if (type(odtype) == type(None)):
        odtype = a.dtype
    assert(a.shape[0] <= n)
    b = np.zeros((n, n, n/2 + 1) + a.shape[3:], dtype = odtype)
    m = a.shape[0]
    b[     :m/2,      :m/2, :m/2+1] = a[     :m/2,      :m/2, :m/2+1]
    b[     :m/2, n-m/2:   , :m/2+1] = a[     :m/2, m-m/2:   , :m/2+1]
    b[n-m/2:   ,      :m/2, :m/2+1] = a[m-m/2:   ,      :m/2, :m/2+1]
    b[n-m/2:   , n-m/2:   , :m/2+1] = a[m-m/2:   , m-m/2:   , :m/2+1]
    return b


