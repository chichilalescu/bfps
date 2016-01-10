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



import math
import numpy as np

def generate_data_3D(
        n0, n1, n2,
        dtype = np.complex128,
        p = 1.5,
        amplitude = 0.5):
    """
    generate something that has the proper shape
    """
    assert(n0 % 2 == 0 and n1 % 2 == 0 and n2 % 2 == 0)
    a = np.zeros((n1, n0, n2/2+1), dtype = dtype)
    a[:] = amplitude*(np.random.randn(*a.shape) + 1j*np.random.randn(*a.shape))
    k, j, i = np.mgrid[-n1/2+1:n1/2+1, -n0/2+1:n0/2+1, 0:n2/2+1]
    k = (k**2 + j**2 + i**2)**.5
    k = np.roll(k, n1//2+1, axis = 0)
    k = np.roll(k, n0//2+1, axis = 1)
    k[0, 0, 0] = 1
    a /= k**p
    a[0, 0, 0] = 0
    for ky in range(1, n1//2):
        a[n1-ky, 0, 0] = np.conj(a[ky, 0, 0])
    for kz in range(1, n0//2):
        a[0, n0-kz, 0] = np.conj(a[0, kz, 0])
    for ky in range(1, n1//2):
        for kz in range(1, n0):
            a[n1-ky, n0-kz, 0] = np.conj(a[ky, kz, 0])
    ii = np.where(k > min(n0, n1, n2)/3.)
    a[ii] = 0
    return a

def randomize_phases(v):
    phi = np.random.random(v.shape[:3])*(2*np.pi)
    phi[0, 0, 0] = 0.0
    for ky in range(1, phi.shape[0]//2):
        phi[phi.shape[0] - ky, 0, 0] = - phi[ky, 0, 0]
    for kz in range(1, phi.shape[1]//2):
        phi[0, phi.shape[1] - kz, 0] = - phi[0, kz, 0]
    for ky in range(1, phi.shape[0]//2):
        for kz in range(1, phi.shape[1]):
            phi[phi.shape[0] - ky, phi.shape[1] - kz, 0] = - phi[ky, kz, 0]
    return v*(np.exp(1j*phi)[:, :, :, None]).astype(v.dtype)

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

def get_kindices(
        n = 64):
    nx = n
    nz = n
    kx = np.arange(0, nx//2+1, 1).astype(np.float)
    kvals = []
    radii = set([])
    index = []
    for iz in range(1, kx.shape[0]):
        for ix in range(1, kx.shape[0]):
            kval = (kx[iz]**2+kx[ix]**2)**.5
            tmp = math.modf(kval)
            if (tmp[0] == 0 and tmp[1] <= nx//2):
                kvals.append([kx[iz], kx[ix]])
                radii.add(math.floor(kval))
                index.append([ix, iz])

    kvals = np.array(kvals)
    index = np.array(index)
    new_kvals = []
    ordered_kvals = []
    radius_vals = []
    ii = []
    for r in radii:
        ncircle = np.count_nonzero((kvals[:, 0]**2 + kvals[:, 1]**2)**.5 == r)
        indices = np.where((kvals[:, 0]**2 + kvals[:, 1]**2)**.5 == r)[0]
        if ncircle > 2:
            ordered_kvals.append(kvals[indices])
            new_kvals += list(kvals[indices])
            radius_vals.append(r)
            ii += list(index[indices])
    ii = np.array(ii)
    good_indices = np.where(ii[:, 1] > 0)[0]
    i1 = (kx.shape[0] - 1)*2 - ii[good_indices, 1]
    i0 = ii[good_indices, 0]
    i1 = np.concatenate((ii[:, 1], i1)),
    i0 = np.concatenate((ii[:, 0], i0)),
    return np.vstack((i0, i1)).T.copy()

try:
    import sympy as sp
    rational0 = sp.Rational(0)
    rational1 = sp.Rational(1)
except ImportError:
    rational0 = 0.0
    rational1 = 1.0

def get_fornberg_coeffs(
        x = None,
        a = None):
    N = len(a) - 1
    d = []
    for m in range(N+1):
        d.append([])
        for n in range(N+1):
            d[m].append([])
            for j in range(N+1):
                d[m][n].append(rational0)
    d[0][0][0] = rational1
    c1 = rational1
    for n in range(1, N+1):
        c2 = rational1
        for j in range(n):
            c3 = a[n] - a[j]
            c2 = c2*c3
            for m in range(n+1):
                d[m][n][j] = ((a[n] - x)*d[m][n-1][j] - m*d[m-1][n-1][j]) / c3
        for m in range(n+1):
            d[m][n][n] = (c1 / c2)*(m*d[m-1][n-1][n-1] - (a[n-1] - x)*d[m][n-1][n-1])
        c1 = c2
    coeffs = []
    for m in range(len(d)):
        coeffs.append([])
        for j in range(len(d)):
            coeffs[-1].append(d[m][N][j])
    return np.array(coeffs).astype(np.float)

def particle_finite_diff_test(
        c,
        m = 3,
        species = 0,
        plot_on = False):
    d = c.get_data_file()
    group = d['particles/tracers{0}'.format(species)]
    acc_on = 'acceleration' in group.keys()
    pos = group['state'].value
    vel = group['velocity'].value
    if acc_on:
        acc = group['acceleration'].value
    n = m
    fc = get_fornberg_coeffs(0, range(-n, n+1))
    dt = d['parameters/dt'].value*d['parameters/niter_part'].value

    num_vel1 = sum(fc[1, n-i]*pos[1+n-i:pos.shape[0]-i-n-1] for i in range(-n, n+1)) / dt
    if acc_on:
        num_acc1 = sum(fc[1, n-i]*vel[1+n-i:vel.shape[0]-i-n-1] for i in range(-n, n+1)) / dt
        num_acc2 = sum(fc[2, n-i]*pos[1+n-i:pos.shape[0]-i-n-1] for i in range(-n, n+1)) / dt**2

    def SNR(a, b):
        return -10*np.log10(np.mean((a - b)**2, axis = (0, 2)) / np.mean(a**2, axis = (0, 2)))
    if acc_on:
        pid = np.argmin(SNR(num_acc1, acc[n+1:-n-1]))
    else:
        pid = np.argmin(SNR(num_vel1, vel[n+1:-n-1]))
    pars = d['parameters']
    to_print = (
            'steps={0}, interp={1}, neighbours={2}, '.format(
                pars['tracers{0}_integration_steps'.format(species)].value,
                pars[str(pars['tracers{0}_interpolator'.format(species)].value) + '_type'].value,
                pars[str(pars['tracers{0}_interpolator'.format(species)].value) + '_neighbours'].value))
    if 'spline' in pars['tracers{0}_interpolator'.format(species)].value:
        to_print += 'smoothness = {0}, '.format(pars[str(pars['tracers{0}_interpolator'.format(species)].value) + '_smoothness'].value)
    to_print += (
            'SNR d1p-vel={0:.3f}'.format(np.mean(SNR(num_vel1, vel[n+1:-n-1]))))
    if acc_on:
        to_print += (', d1v-acc={0:.3f}, d2p-acc={1:.3f}'.format(
                np.mean(SNR(num_acc1, acc[n+1:-n-1])),
                np.mean(SNR(num_acc2, acc[n+1:-n-1]))))
    print(to_print)
    if plot_on and acc_on:
        col = ['red', 'green', 'blue']
        fig = plt.figure()
        a = fig.add_subplot(111)
        for cc in range(3):
            a.plot(num_acc1[:, pid, cc], color = col[cc])
            a.plot(num_acc2[:, pid, cc], color = col[cc], dashes = (2, 2))
            a.plot(acc[m+1:, pid, cc], color = col[cc], dashes = (1, 1))

        for n in range(1, m):
            fc = get_fornberg_coeffs(0, range(-n, n+1))
            dt = d['parameters/dt'].value*d['parameters/niter_part'].value

            num_acc1 = sum(fc[1, n-i]*vel[n-i:vel.shape[0]-i-n] for i in range(-n, n+1)) / dt
            num_acc2 = sum(fc[2, n-i]*pos[n-i:pos.shape[0]-i-n] for i in range(-n, n+1)) / dt**2

            for cc in range(3):
                a.plot(num_acc1[m-n:, pid, cc], color = col[cc])
                a.plot(num_acc2[m-n:, pid, cc], color = col[cc], dashes = (2, 2))
        fig.tight_layout()
        fig.savefig('acc_test_{0}_{1}.pdf'.format(c.simname, species))
        plt.close(fig)
    return pid

