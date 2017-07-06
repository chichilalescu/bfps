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
    a.plot(dd.get_coordinate('z'), b[:, 0, 0], label = 'ball')
    a.plot(dd.get_coordinate('z'), g[:, 0, 0], label = 'Gauss', dashes = (3, 3))
    a.plot(dd.get_coordinate('z'), s[:, 0, 0], label = 'sinc', dashes = (1, 1))
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

