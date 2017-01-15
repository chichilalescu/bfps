#ifndef SHAREDARRAY_HPP
#define SHAREDARRAY_HPP

#include <omp.h>
#include <functional>
#include <iostream>

// Cannot be used by different parallel section at the same time
template <class ValueType>
class shared_array{
    int currentNbThreads;
    ValueType** values;
    size_t dim;

    omp_lock_t locker;
    std::function<void(ValueType*)> initFunc;

    bool hasBeenMerged;

public:
    shared_array(const size_t inDim)
            : currentNbThreads(omp_get_max_threads()),
              values(nullptr), dim(inDim), hasBeenMerged(false){
        values = new ValueType*[currentNbThreads];
        values[0] = new ValueType[dim];
        for(int idxThread = 1 ; idxThread < currentNbThreads ; ++idxThread){
            values[idxThread] = nullptr;
        }
        omp_init_lock(&locker);
    }

    shared_array(const size_t inDim, std::function<void(ValueType*)> inInitFunc)
            : shared_array(inDim){
        setInitFunction(inInitFunc);
    }

    ~shared_array(){
        omp_destroy_lock(&locker);
        for(int idxThread = 0 ; idxThread < currentNbThreads ; ++idxThread){
            delete[] values[idxThread];
        }
        delete[] values;
        if(hasBeenMerged == false){
            // TODO remove when bug solved
            std::cerr << "A shared array has not been merged.... might be a bug" << std::endl;
        }
    }

    ValueType* getMasterData(){
        return values[0];
    }

    const ValueType* getMasterData() const{
        return values[0];
    }

    void merge(){
        ValueType* __restrict__ dest = values[0];
        for(int idxThread = 1 ; idxThread < currentNbThreads ; ++idxThread){
            if(values[idxThread]){
                const ValueType* __restrict__ src = values[idxThread];
                for( size_t idxVal = 0 ; idxVal < dim ; ++idxVal){
                    dest[idxVal] += src[idxVal];
                }
            }
        }
        hasBeenMerged = true;
    }

    void mergeParallel(){
        /*#pragma omp parallel
        {
            int mergeFlag = 1;
            while(mergeFlag < currentNbThreads) mergeFlag <<= 1;

            for( ; mergeFlag != 0 ; mergeFlag >>= 1){
                if(omp_get_thread_num() < mergeFlag){
                    const int otherThread = mergeFlag + omp_get_thread_num();
                    if(otherThread < currentNbThreads && values[otherThread]){
                        if(!values[omp_get_thread_num()]){
                            values[omp_get_thread_num()] = new ValueType[dim];
                            ValueType* __restrict__ dest = values[omp_get_thread_num()];
                            const ValueType* __restrict__ src = values[otherThread];
                            for( size_t idxVal = 0 ; idxVal < dim ; ++idxVal){
                                dest[idxVal] = src[idxVal];
                            }
                        }
                        else{
                            ValueType* __restrict__ dest = values[omp_get_thread_num()];
                            const ValueType* __restrict__ src = values[otherThread];
                            for( size_t idxVal = 0 ; idxVal < dim ; ++idxVal){
                                dest[idxVal] += src[idxVal];
                            }
                        }
                    }
                }
                #pragma omp barrier
            }
        }*/
                      for(int idxThread = 1 ; idxThread < currentNbThreads ; ++idxThread){
                         if(values[idxThread]){
                            ValueType* __restrict__ dest = values[0];
                            const ValueType* __restrict__ src = values[idxThread];
                            for( size_t idxVal = 0 ; idxVal < dim ; ++idxVal){
                                dest[idxVal] += src[idxVal];
                            }
                         }
                      }
        hasBeenMerged = true;
    }

    void setInitFunction(std::function<void(ValueType*)> inInitFunc){
        initFunc = inInitFunc;
        initFunc(values[0]);
    }

    ValueType* getMine(){
        if(omp_get_num_threads() > currentNbThreads){
            omp_set_lock(&locker);
            if(omp_get_num_threads() > currentNbThreads){
                ValueType** newValues = new ValueType*[omp_get_num_threads()];
                for(int idxThread = 0 ; idxThread < currentNbThreads ; ++idxThread){
                    newValues[idxThread] = values[idxThread];
                }
                for(int idxThread = currentNbThreads ; idxThread < omp_get_num_threads() ; ++idxThread){
                    newValues[idxThread] = nullptr;
                }
                values = newValues;
                currentNbThreads = omp_get_num_threads();
            }
            omp_unset_lock(&locker);
        }

        if(values[omp_get_thread_num()] == nullptr){
            ValueType* myValue = new ValueType[dim];
            if(initFunc){
                initFunc(myValue);
            }

            omp_set_lock(&locker);
            values[omp_get_thread_num()] = myValue;
            omp_unset_lock(&locker);
	    return myValue;
        }

        omp_set_lock(&locker);
        ValueType* myValue = values[omp_get_thread_num()];
        omp_unset_lock(&locker);
        return myValue;
    }
};

#endif
