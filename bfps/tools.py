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



import numpy as np

def generate_data_3D(
        n0, n1, n2,
        dtype = np.complex128,
        p = 1.5):
    """
    generate something that has the proper shape
    """
    assert(n0 % 2 == 0 and n1 % 2 == 0 and n2 % 2 == 0)
    a = np.zeros((n0, n1, n2/2+1), dtype = dtype)
    a[:] = np.random.randn(*a.shape) + 1j*np.random.randn(*a.shape)
    k, j, i = np.mgrid[-n0/2+1:n0/2+1, -n1/2+1:n1/2+1, 0:n2/2+1]
    k = (k**2 + j**2 + i**2)**.5
    k = np.roll(k, n0//2+1, axis = 0)
    k = np.roll(k, n1//2+1, axis = 1)
    a /= k**p
    a[0, :, :] = 0
    a[:, 0, :] = 0
    a[:, :, 0] = 0
    ii = np.where(k == 0)
    a[ii] = 0
    ii = np.where(k > min(n0, n1, n2)/3)
    a[ii] = 0
    return a

def padd_with_zeros(
        a,
        n0, n1, n2,
        odtype = None):
    if (type(odtype) == type(None)):
        odtype = a.dtype
    assert(a.shape[0] <= n0 and
           a.shape[1] <= n1 and
           a.shape[2] <= n2)
    b = np.zeros((n0, n1, n2/2 + 1) + a.shape[3:], dtype = odtype)
    m0 = a.shape[0]
    m1 = a.shape[1]
    m2 = a.shape[2]
    b[       :m0/2,        :m1/2, :m2/2+1] = a[       :m0/2,        :m1/2, :m2/2+1]
    b[       :m0/2, n1-m1/2:    , :m2/2+1] = a[       :m0/2, m1-m1/2:    , :m2/2+1]
    b[n0-m0/2:    ,        :m1/2, :m2/2+1] = a[m0-m0/2:    ,        :m1/2, :m2/2+1]
    b[n0-m0/2:    , n1-m1/2:    , :m2/2+1] = a[m0-m0/2:    , m1-m1/2:    , :m2/2+1]
    return b


