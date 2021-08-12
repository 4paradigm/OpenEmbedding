#ifndef PARADIGM4_HYPEREMBEDDING_DATATYPE_H
#define PARADIGM4_HYPEREMBEDDING_DATATYPE_H

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <pico-core/Configure.h>
#include <pico-core/Archive.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

typedef float float32_t;
typedef double float64_t;

template<class T> struct TypeCase {};

class DataType {
public:
    enum DType {
        UNKNOWN = 0x0,
        INT8 = 0x1,
        INT16 = 0x2,
        INT32 = 0x4,
        INT64 = 0x8,

        FLOAT32 = 0x104,
        FLOAT64 = 0x108,
    };

    explicit DataType(int dtype = FLOAT32): dtype(dtype) {}

    DataType(const std::string& str) {
        if (str == "int8") {
            dtype = INT8;
        } else if (str == "int16") {
            dtype = INT16;
        } else if (str == "int32") {
            dtype = INT32;
        } else if (str == "int64") {
            dtype = INT64;
        } else if (str == "float32") {
            dtype = FLOAT32;
        } else if (str == "float64") {
            dtype = FLOAT64;
        } else {
            dtype = UNKNOWN;
        }
    }

    class ToString {
    public:
        void operator()(TypeCase<int8_t>, std::string& str) { str = "int8"; }
        void operator()(TypeCase<int16_t>, std::string& str) { str = "int16"; }
        void operator()(TypeCase<int32_t>, std::string& str) { str = "int32"; }
        void operator()(TypeCase<int64_t>, std::string& str) { str = "int64"; }
        void operator()(TypeCase<float32_t>, std::string& str) { str = "float32"; }
        void operator()(TypeCase<float64_t>, std::string& str) { str = "float64"; }
    };

    operator std::string()const {
        std::string str = "unknown";
        invoke(ToString(), str);
        return str;
    }

    std::string to_string()const {
        return *this;
    }

    template<class Function, typename... Params>
    void invoke(Function&& f, Params&&... params)const {
        switch (dtype) {
        case INT8:
            std::forward<Function>(f)(TypeCase<int8_t>(),
                  std::forward<Params>(params)...);
            break;
        case INT16:
            std::forward<Function>(f)(TypeCase<int16_t>(),
                  std::forward<Params>(params)...);
            break;
        case INT32:
            std::forward<Function>(f)(TypeCase<int32_t>(),
                  std::forward<Params>(params)...);
            break;
        case INT64:
            std::forward<Function>(f)(TypeCase<int64_t>(),
                  std::forward<Params>(params)...);
            break;
        case FLOAT32:
            std::forward<Function>(f)(TypeCase<float32_t>(),
                  std::forward<Params>(params)...);
            break;
        case FLOAT64:
            std::forward<Function>(f)(TypeCase<float64_t>(),
                  std::forward<Params>(params)...);
            break;
        case UNKNOWN:
            break;
        default:
            SLOG(FATAL) << "unexpected unknown datatype!";
        }
    }

    size_t size()const {
        return dtype & 0xFF;
    }

    template<class T>
    static DataType from() {
        return DataType(inner_from(TypeCase<T>()));
    }

    friend bool operator==(DataType a, DataType b) {
        return a.dtype == b.dtype;
    }

    friend bool operator!=(DataType a, DataType b) {
        return a.dtype != b.dtype;
    }

    static DType inner_from(TypeCase<int8_t>) { return INT8; }
    static DType inner_from(TypeCase<int16_t>) { return INT16; }
    static DType inner_from(TypeCase<int32_t>) { return INT32; }
    static DType inner_from(TypeCase<int64_t>) { return INT64; }
    static DType inner_from(TypeCase<float32_t>) { return FLOAT32; }
    static DType inner_from(TypeCase<float64_t>) { return FLOAT64; }

    int dtype = FLOAT32;

    PICO_SERIALIZATION(dtype);
};


}
}
}

#endif