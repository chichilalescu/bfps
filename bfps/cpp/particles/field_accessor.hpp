#ifndef FIELD_ACCESSOR_HPP
#define FIELD_ACCESSOR_HPP

#include <algorithm>
#include <array>


class field_accessor {
    static const int nb_dim = 3;

    const double* field_date;
    std::array<size_t,3> local_field_dims;
    std::array<size_t,3> dim_offset;

public:
    field_accessor(const double* in_field_date, const std::array<size_t,3>& in_dims,
                   const std::array<size_t,3>& in_dim_offset)
            : field_date(in_field_date), local_field_dims(in_dims),
              dim_offset(in_dim_offset){
    }

    ~field_accessor(){}

    const double& getValue(const size_t in_index, const int in_dim) const {
        assert(in_index < local_field_dims[0]*local_field_dims[1]*local_field_dims[2]);
        return field_date[in_index*nb_dim + in_dim];
    }

    size_t getIndexFromGlobalPosition(const size_t in_global_x, const size_t in_global_y, const size_t in_global_z) const {
        return getIndexFromLocalPosition(in_global_x - dim_offset[0],
                                         in_global_y - dim_offset[1],
                                         in_global_z - dim_offset[2]);
    }

    size_t getIndexFromLocalPosition(const size_t in_local_x, const size_t in_local_y, const size_t in_local_z) const {
        assert(in_local_x < local_field_dims[0]);
        assert(in_local_y < local_field_dims[1]);
        assert(in_local_z < local_field_dims[2]);
        return (((in_local_z)*local_field_dims[1] +
                in_local_y)*(local_field_dims[2]) + // TODO there was a +2 on local_field_dims[2]
                in_local_x);
    }
};


#endif
