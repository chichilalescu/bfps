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



# relevant for results of "bfps TEST filter_test"

import h5py
import numpy as np
import matplotlib.pyplot as plt

def phi_b(
        x, ell):
    phi = (6. / (np.pi * ell**3)) * np.ones(x.shape, x.dtype)
    bindices = np.where(np.abs(x) > ell/2)
    phi[bindices] = 0
    return phi

def hat_phi_b(
        k, ell):
    arg = k * ell / 2
    phi = (3. / arg**3) * (np.sin(arg) - arg*np.cos(arg))
    return phi

def phi_s(
        x,
        ell,
        prefactor = 2*np.pi):
    kc = prefactor / ell
    arg = kc*x
    phi = (np.sin(arg) - arg*np.cos(arg)) / (2 * x**3 * np.pi**2)
    return phi

def hat_phi_s(
        k, ell, prefactor = 2*np.pi):
    kc = prefactor / ell
    bindices = np.where(np.abs(k) > kc)
    phi = np.ones(k.shape, k.dtype)
    phi[bindices] = 0
    return phi

def phi_g(
        x,
        ell,
        prefactor = 1):
    sigma = prefactor * ell
    phi = np.exp(- 0.5 * (x / sigma)**2) / (sigma**3 * (2*np.pi)**1.5)
    return phi

def hat_phi_g(
        k, ell,
        prefactor = 1):
    sigma = prefactor * ell
    return np.exp(-0.5 * (k * sigma)**2)


def filter_comparison(
        dd = None,
        base_name = 'filter_test_',
        dim = 0):
    b = dd.df['ball/real/{0}'.format(dim)].value
    g = dd.df['Gauss/real/{0}'.format(dim)].value
    s = dd.df['sharp_Fourier_sphere/real/{0}'.format(dim)].value
    d3V = dd.grid_spacing['x']*dd.grid_spacing['y']*dd.grid_spacing['z']
    print(np.sum(b)*d3V)
    print(np.sum(g)*d3V)
    print(np.sum(s)*d3V)
    levels = np.linspace(
            min(b.min(), g.min(), s.min()),
            max(b.max(), g.max(), s.max()),
            64)
    #levels = None
    f = plt.figure(figsize = (12, 6))
    a = f.add_subplot(131)
    v = np.roll(b[..., 0], b.shape[0]//2, axis = 0)
    v = np.roll(v, b.shape[0]//2, axis = 1)
    cc = a.contourf(v, levels = levels)
    f.colorbar(cc, ax = a, orientation = 'horizontal')
    a = f.add_subplot(132)
    v = np.roll(g[..., 0], g.shape[0]//2, axis = 0)
    v = np.roll(v, g.shape[0]//2, axis = 1)
    cc = a.contourf(v, levels = levels)
    f.colorbar(cc, ax = a, orientation = 'horizontal')
    a = f.add_subplot(133)
    v = np.roll(s[..., 0], s.shape[0]//2, axis = 0)
    v = np.roll(v, s.shape[0]//2, axis = 1)
    cc = a.contourf(v, levels = levels)
    f.colorbar(cc, ax = a, orientation = 'horizontal')
    f.tight_layout()
    f.savefig(base_name + '2D.pdf')
    f = plt.figure(figsize = (6, 5))
    a = f.add_subplot(111)
    zz = dd.get_coordinate('z')
    # ball filter
    a.plot(
            zz,
            b[:, 0, 0],
            label = '$\\phi^b$ numeric',
            color = 'red',
            dashes = (4, 4))
    a.plot(
            zz,
            phi_b(zz, dd.parameters['filter_length']),
            label = '$\\phi^b$ exact',
            color = 'red',
            dashes = (1, 1))
    a.plot(
            zz,
            g[:, 0, 0],
            label = '$\\phi^g$',
            color = 'magenta',
            dashes = (4, 4))
    a.plot(
            zz,
            phi_g(zz, dd.parameters['filter_length'], prefactor = 0.5),
            label = '$\\phi^g$ exact',
            color = 'magenta',
            dashes = (1, 1))
    a.plot(
            zz,
            s[:, 0, 0],
            label = '$\\phi^s$ numeric',
            color = 'blue',
            dashes = (4, 4))
    a.plot(
            zz,
            phi_s(zz, dd.parameters['filter_length'], prefactor = 2*np.pi),
            label = '$\\phi^s$ exact',
            color = 'blue',
            dashes = (1, 1))
    a.set_xlim(0, np.pi)
    a.legend(loc = 'best')
    f.tight_layout()
    f.savefig(base_name + '1D.pdf')
    return None

def resolution_comparison(
        dlist = None,
        base_name = 'normalization_test_',
        dim = 0,
        filter_type = 'Gauss'):
    f = plt.figure(figsize = (6, 5))
    a = f.add_subplot(111)
    for dd in dlist:
        s0 = dd.df[filter_type + '/real/{0}'.format(dim)].value
        a.plot(dd.get_coordinate('z'),
               s0[:, 0, 0],
               label = '{0}'.format(dd.simname))
    a.legend(loc = 'best')
    f.tight_layout()
    f.savefig(base_name + filter_type + '_1D.pdf')
    return None

class sim_data:
    def __init__(
            self,
            simname = 'bla'):
        self.simname = simname
        self.df = h5py.File(simname + '_fields.h5', 'r')
        pfile = h5py.File(simname + '.h5', 'r')
        self.parameters = {}
        for kk in pfile['parameters'].keys():
            self.parameters[kk] = pfile['parameters/' + kk].value
        self.grid_spacing = {}
        for kk in ['x', 'y', 'z']:
            self.grid_spacing[kk] = 2*np.pi / (self.parameters['dk' + kk] * self.parameters['n' + kk])
        return None
    def get_coordinate(
            self,
            c = 'x'):
        return np.linspace(
                0, 2*np.pi / self.parameters['dk' + c],
                self.parameters['n' + c],
                endpoint = False)

def main():
    #d32 = sim_data(simname = 'N32')
    #d48 = sim_data(simname = 'N48')
    #d64 = sim_data(simname = 'N64')
    #d128 = sim_data(simname = 'N128')
    #for ff in ['ball',
    #           'sharp_Fourier_sphere',
    #           'Gauss']:
    #    resolution_comparison(
    #            [d32, d48, d64, d128],
    #            dim = 2,
    #            filter_type = ff)
    dd = sim_data(simname = 'test')
    filter_comparison(
            dd,
            dim = 0)
    return None

if __name__ == '__main__':
    main()

