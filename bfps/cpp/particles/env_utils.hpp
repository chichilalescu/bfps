#ifndef ENV_UTILS_HPP
#define ENV_UTILS_HPP


#include <cstdlib>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstring>
#include <array>


class env_utils {
    template <class VariableType>
    static const VariableType StrToOther(const char* const str, const VariableType& defaultValue = VariableType()){
        std::istringstream iss(str,std::istringstream::in);
        VariableType value;
        iss >> value;
        if( /*iss.tellg()*/ iss.eof() ) return value;
        return defaultValue;
    }

public:
    static bool VariableIsDefine(const char inVarName[]){
        return getenv(inVarName) != 0;
    }

    template <class VariableType>
    static const VariableType GetValue(const char inVarName[], const VariableType defaultValue = VariableType()){
        const char*const value = getenv(inVarName);
        if(!value){
            return defaultValue;
        }
        return StrToOther(value,defaultValue);
    }

    static bool GetBool(const char inVarName[], const bool defaultValue = false){
        const char*const value = getenv(inVarName);
        if(!value){
            return defaultValue;
        }
        return (strcmp(value,"TRUE") == 0) || (strcmp(value,"true") == 0) || (strcmp(value,"1") == 0);
    }

    static const char* GetStr(const char inVarName[], const char* const defaultValue = 0){
        const char*const value = getenv(inVarName);
        if(!value){
            return defaultValue;
        }
        return value;
    }

    template <class VariableType, class ArrayType>
    static int GetValueInArray(const char inVarName[], const ArrayType& possibleValues, const int nbPossibleValues, const int defaultIndex = -1){
        const char*const value = getenv(inVarName);
        if(value){
            for(int idxPossible = 0 ; idxPossible < nbPossibleValues ; ++idxPossible){
                if( StrToOther(value,VariableType()) == possibleValues[idxPossible] ){
                    return idxPossible;
                }
            }
        }
        return defaultIndex;
    }


    template <class ArrayType>
    static int GetStrInArray(const char inVarName[], const ArrayType& possibleValues, const int nbPossibleValues, const int defaultIndex = -1){
        const char*const value = getenv(inVarName);
        if(value){
            for(int idxPossible = 0 ; idxPossible < nbPossibleValues ; ++idxPossible){
                if( strcmp(value,possibleValues[idxPossible]) == 0 ){
                    return idxPossible;
                }
            }
        }
        return defaultIndex;
    }
};

#endif

