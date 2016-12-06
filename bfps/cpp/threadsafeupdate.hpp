#ifndef THREADSAFEUPDATE_HPP
#define THREADSAFEUPDATE_HPP

template <class NumType>
class ThreadSafeUpdateCore{
    NumType& num;
public:
    ThreadSafeUpdateCore(NumType& inNum) : num(inNum){}

    ThreadSafeUpdateCore& operator+=(const NumType& inNum){
        #pragma omp atomic update
        num += inNum;
        return *this;
    }

    ThreadSafeUpdateCore& operator-=(const NumType& inNum){
        #pragma omp atomic update
        num -= inNum;
        return *this;
    }

    ThreadSafeUpdateCore& operator*=(const NumType& inNum){
        #pragma omp atomic update
        num *= inNum;
        return *this;
    }

    ThreadSafeUpdateCore& operator/=(const NumType& inNum){
        #pragma omp atomic update
        num /= inNum;
        return *this;
    }
};

template <class NumType>
ThreadSafeUpdateCore<NumType> ThreadSafeUpdate(NumType& inNum){
    return ThreadSafeUpdateCore<NumType>(inNum);
}


#endif
