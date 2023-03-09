//
// Created by adamb on 12.11.2021.
//

#ifndef NN_DIMENSIONSINCOMPATIBLEEXCEPTION_HPP
#define NN_DIMENSIONSINCOMPATIBLEEXCEPTION_HPP
#include "Matrix.h"
#include <string>


#include <exception>
using namespace std;
namespace AlgebraLibrary {
    template<typename NumericType>
    class DimensionsIncompatibleException : public std::exception {
    public:
        DimensionsIncompatibleException(const Matrix <NumericType>* A,
                                        const Matrix <NumericType>* B,
                                        string const& operation) :msg_(
                "Dimensions of matrix A(" + to_string(A->rows) + "x" + to_string(A->cols) +
                ") and B(" + to_string(B->rows) + "x" + to_string(B->cols) +
                ") is not compatible for " + operation + ".") {}
        [[nodiscard]] char const *what() const noexcept override { return msg_.c_str(); }

    private:
        string msg_;
    };

}
#endif //NN_DIMENSIONSINCOMPATIBLEEXCEPTION_HPP
