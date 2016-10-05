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
import h5py
import bfps

class NavierStokesDB(bfps.NavierStokes):
    """
        Example of how bfps is envisioned to be used.
        Standard NavierStokes class is inherited, and then new functionality
        added on top.
        In particular, this class will generate an HDF5 file containing a 5D
        array representing the time history of the velocity field.
        Snapshots are saved every "niter_stat" iterations.

        No effort was spent on optimizing the HDF5 file access, since the code
        was only used for a teeny DNS of 72^3 so far.
    """
    standard_names = ['NSDB',
                      'NSDB-single',
                      'NSDB-double']
    def __init__(
            self,
            name = 'NavierStokesDataBase-v' + bfps.__version__,
            **kwargs):
        bfps.NavierStokes.__init__(
                self,
                name = name,
                **kwargs)
        self.file_datasets_grow += """
                {
                    if (myrank == 0)
                    {
                        hid_t database_file;
                        char dbfname[256];
                        sprintf(dbfname, "%s_field_database.h5", simname);
                        database_file = H5Fopen(dbfname, H5F_ACC_RDWR, H5P_DEFAULT);
                        hsize_t dset = H5Dopen(database_file, "rvelocity", H5P_DEFAULT);
                        grow_single_dataset(dset, niter_todo/niter_stat);
                        H5Dclose(dset);
                        H5Fclose(database_file);
                    }
                }
                """
        self.stat_src += """
                {
                    fs->compute_velocity(fs->cvorticity);
                    *tmp_vec_field = fs->cvelocity;
                    tmp_vec_field->ift();
                    char dbfname[256];
                    sprintf(dbfname, "%s_field_database.h5", simname);
                    tmp_vec_field->io(dbfname, "rvelocity", fs->iteration / niter_stat, false);
                }
                """
        return None
    def get_database_file_name(self):
        return os.path.join(self.work_dir, self.simname + '_field_database.h5')
    def get_database_file(self):
        return h5py.File(self.get_postprocess_file_name(), 'r')
    def write_par(
            self,
            iter0 = 0,
            **kwargs):
        bfps.NavierStokes.write_par(
                self,
                iter0 = iter0,
                **kwargs)
        with h5py.File(self.get_database_file_name(), 'a') as ofile:
            ofile.create_dataset(
                    'rvelocity',
                    (1,
                     self.parameters['nz'],
                     self.parameters['ny'],
                     self.parameters['nx'],
                     3),
                    chunks = (1,
                              self.parameters['nz'],
                              self.parameters['ny'],
                              self.parameters['nx'],
                              3),
                    maxshape = (None,
                                self.parameters['nz'],
                                self.parameters['ny'],
                                self.parameters['nx'],
                                3),
                    dtype = self.rtype)
        return None

