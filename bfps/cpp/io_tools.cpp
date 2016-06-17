/**********************************************************************
*                                                                     *
*  Copyright 2015 Max Planck Institute                                *
*                 for Dynamics and Self-Organization                  *
*                                                                     *
*  This file is part of bfps.                                         *
*                                                                     *
*  bfps is free software: you can redistribute it and/or modify       *
*  it under the terms of the GNU General Public License as published  *
*  by the Free Software Foundation, either version 3 of the License,  *
*  or (at your option) any later version.                             *
*                                                                     *
*  bfps is distributed in the hope that it will be useful,            *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of     *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
*  GNU General Public License for more details.                       *
*                                                                     *
*  You should have received a copy of the GNU General Public License  *
*  along with bfps.  If not, see <http://www.gnu.org/licenses/>       *
*                                                                     *
* Contact: Cristian.Lalescu@ds.mpg.de                                 *
*                                                                     *
**********************************************************************/



#include <typeinfo>
#include <cassert>
#include "io_tools.hpp"


template <typename number>
std::vector<number> read_vector(
        hid_t group,
        std::string dset_name)
{
    std::vector<number> result;
    hsize_t vector_length;
    // first, read size of array
    hid_t dset, dspace;
    hid_t mem_dtype;
    if (typeid(number) == typeid(int))
        mem_dtype = H5Tcopy(H5T_NATIVE_INT);
    else if (typeid(number) == typeid(double))
        mem_dtype = H5Tcopy(H5T_NATIVE_DOUBLE);
    dset = H5Dopen(group, dset_name.c_str(), H5P_DEFAULT);
    dspace = H5Dget_space(dset);
    assert(H5Sget_simple_extent_ndims(dspace) == 1);
    H5Sget_simple_extent_dims(dspace, &vector_length, NULL);
    result.resize(vector_length);
    H5Dread(dset, mem_dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &result.front());
    H5Sclose(dspace);
    H5Dclose(dset);
    H5Tclose(mem_dtype);
    return result;
}

template std::vector<int> read_vector(
        hid_t, std::string);
template std::vector<double> read_vector(
        hid_t, std::string);

