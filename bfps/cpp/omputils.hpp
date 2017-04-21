#ifndef OMPUTILS_HPP
#define OMPUTILS_HPP

#include <omp.h>

namespace OmpUtils{

template <class IndexType>
inline IndexType ForIntervalStart(const IndexType size){
    const double chunk = double(size)/double(omp_get_num_threads());
    const IndexType start = IndexType(chunk*double(omp_get_thread_num()));
    return start;
}

template <class IndexType>
inline IndexType ForIntervalEnd(const IndexType size){
    const double chunk = double(size)/double(omp_get_num_threads());
    const IndexType end = (omp_get_thread_num() == omp_get_num_threads()-1) ?
                                size:
                                IndexType(chunk*double(omp_get_thread_num()+1));
    return end;
}

}


#endif
