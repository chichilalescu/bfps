#include "hdf5_tools.hpp"
#include <cfloat>
#include <climits>

int hdf5_tools::require_size_single_dataset(hid_t dset, int tsize)
{
    int ndims;
    hsize_t space;
    space = H5Dget_space(dset);
    ndims = H5Sget_simple_extent_ndims(space);
    hsize_t *dims = new hsize_t[ndims];
    hsize_t *maxdims = new hsize_t[ndims];
    H5Sget_simple_extent_dims(space, dims, maxdims);
    if (dims[0] < hsize_t(tsize) && maxdims[0] == H5S_UNLIMITED)
    {
        dims[0] = tsize;
        H5Dset_extent(dset, dims);
    }
    H5Sclose(space);
    delete[] maxdims;
    delete[] dims;
    return EXIT_SUCCESS;
}

int hdf5_tools::grow_single_dataset(hid_t dset, int tincrement)
{
    int ndims;
    hsize_t space;
    space = H5Dget_space(dset);
    ndims = H5Sget_simple_extent_ndims(space);
    hsize_t *dims = new hsize_t[ndims];
    hsize_t *maxdims = new hsize_t[ndims];
    H5Sget_simple_extent_dims(space, dims, maxdims);
    if (maxdims[0] == H5S_UNLIMITED)
    {
        dims[0] += tincrement;
        H5Dset_extent(dset, dims);
    }
    H5Sclose(space);
    delete[] maxdims;
    delete[] dims;
    return EXIT_SUCCESS;
}

herr_t hdf5_tools::require_size_dataset_visitor(
    hid_t o_id,
    const char *name,
    const H5O_info_t *info,
    void *op_data)
{
    if (info->type == H5O_TYPE_DATASET)
    {
        hsize_t dset = H5Dopen(o_id, name, H5P_DEFAULT);
        require_size_single_dataset(dset, *((int*)(op_data)));
        H5Dclose(dset);
    }
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
        int tincrement)
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


int hdf5_tools::require_size_file_datasets(
        const hid_t stat_file,
        const std::string group_name,
        int tsize)
{
    int file_problems = 0;

    hid_t group;
    group = H5Gopen(stat_file, group_name.c_str(), H5P_DEFAULT);
    H5Ovisit(
            group,
            H5_INDEX_NAME,
            H5_ITER_NATIVE,
            require_size_dataset_visitor,
            &tsize);
    H5Gclose(group);
    return file_problems;
}

template <typename number>
std::vector<number> hdf5_tools::read_vector(
        const hid_t group,
        const std::string dset_name)
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

template <typename number>
number hdf5_tools::read_value(
        const hid_t group,
        const std::string dset_name)
{
    number result;
    hid_t dset;
    hid_t mem_dtype;
    if (typeid(number) == typeid(int))
        mem_dtype = H5Tcopy(H5T_NATIVE_INT);
    else if (typeid(number) == typeid(double))
        mem_dtype = H5Tcopy(H5T_NATIVE_DOUBLE);
    if (H5Lexists(group, dset_name.c_str(), H5P_DEFAULT))
    {
        dset = H5Dopen(group, dset_name.c_str(), H5P_DEFAULT);
        H5Dread(dset, mem_dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &result);
        H5Dclose(dset);
    }
    else
    {
        DEBUG_MSG("attempted to read dataset %s which does not exist.\n",
                dset_name.c_str());
        if (typeid(number) == typeid(int))
            result = INT_MAX;
        else if (typeid(number) == typeid(double))
            result = number(DBL_MAX);
    }
    H5Tclose(mem_dtype);
    return result;
}

template <typename dtype>
std::vector<dtype> hdf5_tools::read_vector_with_single_rank(
        const int myrank,
        const int rank_to_use,
        const MPI_Comm COMM,
        const hid_t file_id,
        const std::string dset_name)
{
    std::vector<dtype> data;
    int vector_size;
    if (myrank == rank_to_use)
    {
        data = hdf5_tools::read_vector<dtype>(
                file_id,
                dset_name);
        vector_size = data.size();
    }
    MPI_Bcast(
            &vector_size,
            1,
            MPI_INT,
            rank_to_use,
            COMM);

    if (myrank != rank_to_use)
        data.resize(vector_size);
    MPI_Bcast(
            &data.front(),
            vector_size,
            (typeid(dtype) == typeid(int)) ? MPI_INT : MPI_DOUBLE,
            rank_to_use,
            COMM);
    return data;
}

std::string hdf5_tools::read_string(
        const hid_t group,
        const std::string dset_name)
{
    hid_t dset = H5Dopen(group, dset_name.c_str(), H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hid_t memtype = H5Dget_type(dset);
    char *string_data = (char*)malloc(256);
    H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &string_data);
    std::string std_string_data = std::string(string_data);
    free(string_data);
    H5Sclose(space);
    H5Tclose(memtype);
    H5Dclose(dset);
    return std_string_data;
}

template
std::vector<int> hdf5_tools::read_vector<int>(
        const hid_t,
        const std::string);

template
std::vector<double> hdf5_tools::read_vector<double>(
        const hid_t,
        const std::string);

template
std::vector<int> hdf5_tools::read_vector_with_single_rank<int>(
        const int myrank,
        const int rank_to_use,
        const MPI_Comm COMM,
        const hid_t file_id,
        const std::string dset_name);

template
std::vector<double> hdf5_tools::read_vector_with_single_rank<double>(
        const int myrank,
        const int rank_to_use,
        const MPI_Comm COMM,
        const hid_t file_id,
        const std::string dset_name);

template
int hdf5_tools::read_value<int>(
        const hid_t,
        const std::string);

template
double hdf5_tools::read_value<double>(
        const hid_t,
        const std::string);

