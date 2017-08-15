#ifndef SHAREDARRAY_HPP
#define SHAREDARRAY_HPP

#include <omp.h>
#include <functional>
#include <iostream>

// Cannot be used by different parallel section at the same time
template <class ValueType>
class shared_array{
    int currentNbThreads;
    ValueType** __restrict__ values;
    size_t dim;

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
    }

    shared_array(const size_t inDim, std::function<void(ValueType*)> inInitFunc)
            : shared_array(inDim){
        setInitFunction(inInitFunc);
    }

    ~shared_array(){
        for(int idxThread = 0 ; idxThread < currentNbThreads ; ++idxThread){
            delete[] values[idxThread];
        }
        delete[] values;
        if(hasBeenMerged == false){
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
    
    template <class Func>
    void merge(Func func){
        ValueType* __restrict__ dest = values[0];
        for(int idxThread = 1 ; idxThread < currentNbThreads ; ++idxThread){
            if(values[idxThread]){
                const ValueType* __restrict__ src = values[idxThread];
                for( size_t idxVal = 0 ; idxVal < dim ; ++idxVal){
                    dest[idxVal] = func(idxVal, dest[idxVal], src[idxVal]);
                }
            }
        }
        hasBeenMerged = true;
    }

    void mergeParallel(){
        merge(); // not done yet
    }
    
    template <class Func>
    void mergeParallel(Func func){
        merge(func); // not done yet
    }

    void setInitFunction(std::function<void(ValueType*)> inInitFunc){
        initFunc = inInitFunc;
        initFunc(values[0]);
    }

    ValueType* getMine(){
        assert(omp_get_thread_num() < currentNbThreads);

        if(values[omp_get_thread_num()] == nullptr){
            ValueType* myValue = new ValueType[dim];
            if(initFunc){
                initFunc(myValue);
            }

            values[omp_get_thread_num()] = myValue;
	        return myValue;
        }

        return values[omp_get_thread_num()];
    }
};

#endif
