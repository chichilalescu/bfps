/**********************************************************************
*                                                                     *
*  Copyright 2017 Max Planck Institute                                *
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



#ifndef HDF5_TOOLS_HPP
#define HDF5_TOOLS_HPP

#include <vector>
#include <hdf5.h>
#include "base.hpp"

namespace hdf5_tools
{
    int grow_single_dataset(
            hid_t dset,
            int tincrement);

    herr_t grow_dataset_visitor(
        hid_t o_id,
        const char *name,
        const H5O_info_t *info,
        void *op_data);

    int grow_file_datasets(
            const hid_t stat_file,
            const std::string group_name,
            int tincrement);

    int require_size_single_dataset(
            hid_t dset,
            int tincrement);

    herr_t require_size_dataset_visitor(
        hid_t o_id,
        const char *name,
        const H5O_info_t *info,
        void *op_data);

    int require_size_file_datasets(
            const hid_t stat_file,
            const std::string group_name,
            int tincrement);

    template <typename number>
    std::vector<number> read_vector(
            const hid_t group,
            const std::string dset_name);

    template <typename number>
    std::vector<number> read_vector_with_single_rank(
            const int myrank,
            const int rank_to_use,
            const MPI_Comm COMM,
            const hid_t group,
            const std::string dset_name);

    std::string read_string(
            const hid_t group,
            const std::string dset_name);

    template <typename number>
    number read_value(
            const hid_t group,
            const std::string dset_name);
}

#endif//HDF5_TOOLS_HPP

