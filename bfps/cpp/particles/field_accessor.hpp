#ifndef FIELD_ACCESSOR_HPP
#define FIELD_ACCESSOR_HPP

#include <algorithm>
#include <array>

#include "particles_utils.hpp"

template <class real_number>
class field_accessor {
    static const int nb_dim = 3;

    const real_number* field_date;
    std::array<size_t,3> local_field_dims;
    std::array<size_t,3> local_field_offset;
    std::array<size_t,3> field_memory_dims;

public:
    field_accessor(const real_number* in_field_date, const std::array<size_t,3>& in_dims,
                   const std::array<size_t,3>& in_local_field_offset,
                   const std::array<size_t,3>& in_field_memory_dims)
            : field_date(in_field_date), local_field_dims(in_dims),
              local_field_offset(in_local_field_offset),
              field_memory_dims(in_field_memory_dims){
    }

    ~field_accessor(){}

    const real_number& getValue(const size_t in_index, const int in_dim) const {
        assert(in_index < field_memory_dims[IDX_X]*field_memory_dims[IDX_Y]*field_memory_dims[IDX_Z]);
        return field_date[in_index*nb_dim + in_dim];
    }

    size_t getIndexFromGlobalPosition(const size_t in_global_x, const size_t in_global_y, const size_t in_global_z) const {
        return getIndexFromLocalPosition(in_global_x - local_field_offset[IDX_X],
                                         in_global_y - local_field_offset[IDX_Y],
                                         in_global_z - local_field_offset[IDX_Z]);
    }

    size_t getIndexFromLocalPosition(const size_t in_local_x, const size_t in_local_y, const size_t in_local_z) const {
        assert(0 <= in_local_x && in_local_x < local_field_dims[IDX_X]);
        assert(0 <= in_local_y && in_local_y < local_field_dims[IDX_Y]);
        assert(0 <= in_local_z && in_local_z < local_field_dims[IDX_Z]);
        return (((in_local_z)*field_memory_dims[IDX_Y] +
                in_local_y)*(field_memory_dims[IDX_X]) +
                in_local_x);
    }
};


#endif
