#ifndef PARTICLES_UTILS_HPP
#define PARTICLES_UTILS_HPP

#include <cassert>
#include <stack>


namespace particles_utils {

template <int nb_values, class Predicate>
inline int partition(double* array, const int size, Predicate pdc)
{
    if(size == 0) return 0;
    if(size == 1) return (pdc(&array[0])?1:0);

    int idxInsert = 0;

    for(int idx = 0 ; idx < size && pdc(&array[idx*nb_values]); ++idx){
        idxInsert += 1;
    }

    for(int idx = idxInsert ; idx < size ; ++idx){
        if(pdc(&array[idx*nb_values])){
            for(int idxVal = 0 ; idxVal < nb_values ; ++idxVal){
                std::iter_swap(array[idx*nb_values + idxVal], array[idxInsert*nb_values + idxVal]);
            }
            idxInsert += 1;
        }
    }

    return idxInsert;
}


template <int nb_values, class Predicate1, class Predicate2>
inline int partition_extra(double* array, const int size, Predicate1 pdc, Predicate2 pdcswap, const int offset_idx_swap = 0)
{
    if(size == 0) return 0;
    if(size == 1) return (pdc(&array[0])?1:0);

    int idxInsert = 0;

    for(int idx = 0 ; idx < size && pdc(&array[idx*nb_values]); ++idx){
        idxInsert += 1;
    }

    for(int idx = idxInsert ; idx < size ; ++idx){
        if(pdc(&array[idx*nb_values])){
            for(int idxVal = 0 ; idxVal < nb_values ; ++idxVal){
                std::swap(array[idx*nb_values + idxVal], array[idxInsert*nb_values + idxVal]);
            }
            pdcswap(idx+offset_idx_swap, idxInsert+offset_idx_swap);
            idxInsert += 1;
        }
    }

    return idxInsert;
}

template <int nb_values, class Predicate1, class Predicate2>
inline void partition_extra_z(double* array, const int size, const int nb_partitions,
                              int partitions_size[], int partitions_offset[],
                              Predicate1 partitions_limits, Predicate2 pdcswap)
{
    if(nb_partitions == 0){
        return ;
    }

    partitions_offset[0] = 0;
    partitions_offset[nb_partitions] = size;

    if(nb_partitions == 1){
        partitions_size[0] = 0;
        return;
    }

    if(nb_partitions == 2){
        const double limit = partitions_limits(0);
        const int size_current = partition_extra<nb_values>(array, size,
                [&](const double inval[]){
            return inval[nb_values-1] < limit;
        }, pdcswap);
        partitions_size[0] = size_current;
        partitions_size[1] = size-size_current;
        partitions_offset[1] = size_current;
        return;
    }

    std::stack<std::pair<int,int>> toproceed;

    toproceed.push({0, nb_partitions});

    while(toproceed.size()){
        const std::pair<int,int> current_part = toproceed.top();
        toproceed.pop();

        assert(current_part.second-current_part.first >= 1);

        if(current_part.second-current_part.first == 1){
            partitions_size[current_part.first] = partitions_offset[current_part.first+1] - partitions_offset[current_part.first];
        }
        else{
            const int idx_middle = (current_part.second-current_part.first)/2 + current_part.first - 1;

            const int size_unpart = partitions_offset[current_part.second]- partitions_offset[current_part.first];

            const double limit = partitions_limits(idx_middle);
            const int size_current = partition_extra<nb_values>(&array[partitions_offset[current_part.first]*nb_values],
                                                     size_unpart,
                    [&](const double inval[]){
                return inval[nb_values-1] < limit;
            }, pdcswap, partitions_offset[current_part.first]);

            partitions_offset[idx_middle+1] = size_current + partitions_offset[current_part.first];

            toproceed.push({current_part.first, idx_middle+1});

            toproceed.push({idx_middle+1, current_part.second});
        }
    }
}

template <int nb_values, class Predicate1, class Predicate2>
inline std::pair<std::vector<int>,std::vector<int>> partition_extra_z(double* array, const int size,
                                                                      const int nb_partitions, Predicate1 partitions_limits,
                                                                        Predicate2 pdcswap){

    std::vector<int> partitions_size(nb_partitions);
    std::vector<int> partitions_offset(nb_partitions+1);
    partition_extra_z<nb_values, Predicate1, Predicate2>(array, size, nb_partitions,
                                                         partitions_size.data(), partitions_offset.data(),
                                                         partitions_limits, pdcswap);
    return {std::move(partitions_size), std::move(partitions_offset)};
}


template <class NumType = int>
class IntervalSplitter {
    const NumType nb_items;
    const NumType nb_intervals;
    const NumType my_idx;

    double step_split;
    NumType offset_mine;
    NumType size_mine;
public:
    IntervalSplitter(const NumType in_nb_items,
                     const NumType in_nb_intervals,
                     const NumType in_my_idx)
        : nb_items(in_nb_items), nb_intervals(in_nb_intervals), my_idx(in_my_idx),
          step_split(0), offset_mine(0), size_mine(0){
        if(nb_items <= nb_intervals){
            step_split = 1;
            if(my_idx < nb_intervals){
                offset_mine = my_idx;
                size_mine = 1;
            }
            else{
                offset_mine = nb_intervals;
                size_mine = 0;
            }
        }
        else{
            step_split = double(nb_items)/double(nb_intervals);
            offset_mine = NumType(step_split*double(my_idx));
            size_mine = NumType(step_split*double(my_idx+1)-step_split*double(my_idx));
            assert(my_idx != nb_intervals-1 || (offset_mine+size_mine) == nb_items);
        }
    }

    NumType getMySize() const {
        return size_mine;
    }

    NumType getMyOffset() const {
        return offset_mine;
    }

    NumType getSizeOther(const NumType in_idx_other) const {
        return IntervalSplitter<NumType>(nb_items, nb_intervals, in_idx_other).getMySize();
    }

    NumType getOffsetOther(const NumType in_idx_other) const {
        return IntervalSplitter<NumType>(nb_items, nb_intervals, in_idx_other).getMyOffset();
    }

    NumType getOwner(const NumType in_item_idx) const {
        return NumType(double(in_item_idx)/step_split);
    }
};

// http://en.cppreference.com/w/cpp/algorithm/transform
template<class InputIt, class OutputIt, class UnaryOperation>
OutputIt transform(InputIt first1, InputIt last1, OutputIt d_first,
                   UnaryOperation unary_op)
{
    while (first1 != last1) {
        *d_first++ = unary_op(*first1++);
    }
    return d_first;
}


template <class NumType>
void memzero(NumType* array, size_t size){
    memset(array, 0, size*sizeof(NumType));
}

template <class NumType>
void memzero(std::unique_ptr<NumType[]>& array, size_t size){
    memset(array.get(), 0, size*sizeof(NumType));
}


}

#endif
