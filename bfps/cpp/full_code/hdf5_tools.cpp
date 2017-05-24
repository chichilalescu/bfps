#include "hdf5_tools.hpp"

int hdf5_tools::grow_single_dataset(hid_t dset, int tincrement)
{
    int ndims;
    hsize_t space;
    space = H5Dget_space(dset);
    ndims = H5Sget_simple_extent_ndims(space);
    hsize_t *dims = new hsize_t[ndims];
    H5Sget_simple_extent_dims(space, dims, NULL);
    dims[0] += tincrement;
    H5Dset_extent(dset, dims);
    H5Sclose(space);
    delete[] dims;
    return EXIT_SUCCESS;
}

herr_t hdf5_tools::grow_dataset_visitor(
    hid_t o_id,
    const char *name,
    const H5O_info_t *info,
    void *op_data)
{
    if (info->type == H5O_TYPE_DATASET)
    {
        hsize_t dset = H5Dopen(o_id, name, H5P_DEFAULT);
        grow_single_dataset(dset, *((int*)(op_data)));
        H5Dclose(dset);
    }
    return EXIT_SUCCESS;
}


int hdf5_tools::grow_file_datasets(
        const hid_t stat_file,
        const std::string group_name,
        const int tincrement)
{
    int file_problems = 0;

    hid_t group;
    group = H5Gopen(stat_file, group_name.c_str(), H5P_DEFAULT);
    H5Ovisit(
            group,
            H5_INDEX_NAME,
            H5_ITER_NATIVE,
            grow_dataset_visitor,
            &tincrement);
    H5Gclose(group);
    return file_problems;
}

