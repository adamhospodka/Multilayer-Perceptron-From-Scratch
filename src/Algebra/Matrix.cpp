#include "Matrix.h"

#include <iostream>
#include <iomanip>
#include "DimensionsIncompatibleException.hpp"
//#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <ostream>
#include <random>
//only works after C++20 standard
#include <bit>
#include <limits>
#include <cstdint>



using namespace std;

using namespace AlgebraLibrary;

template <typename NumericType>
Matrix<NumericType>::Matrix(int rows, int cols) : cols(cols), rows(rows), cokoliv(new NumericType[cols * rows]()) {}
// Option => Matrix<NumericType>::Matrix(int rows, int cols) : cols(cols), rows(rows), cokoliv(cols * rows, 0) {}

template <typename NumericType>
Matrix<NumericType>::Matrix(int rows,
/*     .-.            */ int cols,
/*    (o,o)           */ NumericType *cokoliv) : 
/*     (e)            */
/*   .-="=-.  \(_     */ rows(rows),
/*  //==I==\\,/       */ cols(cols),
/* ()  ="=  ()        */
/*    \`(0V0)         */ cokoliv(new NumericType[rows * cols])
/*     ||\\           */ {
/*     || \\          */ // fast memory copy, highly optimized, hope it works
/*     ()  ()         */ memcpy(this->cokoliv, cokoliv, sizeof(NumericType) * rows * cols);
/*    //  //          */}
/*   '/  '/           */
/*   "== "==          */

template <typename NumericType>
void Matrix<NumericType>::multiply(const Matrix<NumericType> *m, Matrix<NumericType> *out) {
    if (cols != m->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, m, "multiplication");
    } else if(m->cols != out->cols ) {
        throw DimensionsIncompatibleException<NumericType>(m, out, "multiplication target to output");
    } else if( rows != out->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, out, "multiplication source to output");
    }


    // int new_cols = m->cols;
    //int new_rows = this->rows;
    /*                  _________ cols = 2, rows = 4
     *                  |      0      |      1      |
     *                  |      2      |      3      |
     *                  |      4      |      5      |
    cols = 4, rows = 2  |      6      |      7      |
    _________________   ˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇ
    | 0 | 1 | 2 | 3 |     00+12+24+36 | 01+13+25+37
    | 4 | 5 | 6 | 7 |     40+52+64+76 | 41+53+65+77
    ˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇ
    */

    //cout << "Multiplication: ";
    for (int c = 0; c < out->cols; ++c) {
        for (int r = 0; r < out->rows; ++r) {
            //cout << r * out->cols + c << "=" << cols * r << "x" << c;
            out->cokoliv[r * out->cols + c] = cokoliv[cols * r] * m->cokoliv[c];
            for (int length = 1; length < cols; ++length) {
                //cout << "|" << r * out->cols + c << "=" << cols * r + length << "x" << m->cols * length + c;
                out->cokoliv[r * out->cols + c] += cokoliv[cols * r + length] * m->cokoliv[m->cols * length + c];
            }
            //cout << " ";
        }
    }
    //cout << endl;
}

template <typename NumericType>
void Matrix<NumericType>::multiply_swapped_second(const Matrix<NumericType> *m, Matrix<NumericType> *out) {
    if (cols != m->cols) {
        throw DimensionsIncompatibleException<NumericType>(this, m, "swapped second multiplication");
    } else if(m->rows != out->rows) {
        throw DimensionsIncompatibleException<NumericType>(m, out, "swapped second multiplication target to output");
    } else if(rows != out->cols) {
        throw DimensionsIncompatibleException<NumericType>(this, out, "swapped second multiplication source to output");
    }
    for (int c = 0; c < out->cols; ++c) {
        for (int r = 0; r < out->rows; ++r) {
            out->cokoliv[r * out->cols + c] = cokoliv[c * cols] * m->cokoliv[r * m->cols];
            for (int length = 1; length < cols; ++length) {
                out->cokoliv[r * out->cols + c] += cokoliv[c * cols + length] * m->cokoliv[r * m->cols + length];
            }
        }
    }
}

template <typename NumericType>
void Matrix<NumericType>::multiply_swapped_first(const Matrix<NumericType> *m, Matrix<NumericType> *out) {
    if (rows != m->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, m, "swapped first multiplication");
    } else if(m->cols != out->cols ) {
        throw DimensionsIncompatibleException<NumericType>(m, out, "swapped first multiplication target to output");
    } else if( cols != out->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, out, "swapped first multiplication source to output");
    }
    for (int c = 0; c < out->cols; ++c) {
        for (int r = 0; r < out->rows; ++r) {
            out->cokoliv[r * out->cols + c] = cokoliv[rows * r] * m->cokoliv[c];
            for (int length = 1; length < rows; ++length) {
                out->cokoliv[r * out->cols + c] += cokoliv[rows * r + length] * m->cokoliv[m->cols * length + c];
            }
        }
    }
}

template <typename NumericType>
void Matrix<NumericType>::multiply_swapped_first_add(const Matrix<NumericType> *m, Matrix<NumericType> *out) {
    if (rows != m->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, m, "swapped first multiplication");
    } else if(m->cols != out->cols ) {
        throw DimensionsIncompatibleException<NumericType>(m, out, "swapped first multiplication target to output");
    } else if( cols != out->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, out, "swapped first multiplication source to output");
    }
    for (int c = 0; c < out->cols; ++c) {
        for (int r = 0; r < out->rows; ++r) {
            for (int length = 0; length < rows; ++length) {
                out->cokoliv[r * out->cols + c] += cokoliv[rows * r + length] * m->cokoliv[m->cols * length + c];
            }
        }
    }
}

// TODO: UNUSED
template <typename NumericType>
void Matrix<NumericType>::multiply_add(const Matrix<NumericType> *m, Matrix<NumericType> *out) {
    if (cols != m->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, m, "added multiplication");
    } else if(m->cols != out->cols ) {
        throw DimensionsIncompatibleException<NumericType>(m, out, "added multiplication target to output ");
    } else if( rows != out->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, out, "added multiplication source output");
    }


    // multiply two matrices and add result to the third one
    for (int c = 0; c < out->cols; ++c) {
        for (int r = 0; r < out->rows; ++r) {
            for (int length = 0; length < cols; ++length) {
                out->cokoliv[r * out->cols + c] += cokoliv[cols * r + length] * m->cokoliv[m->cols * length + c];
            }
        }
    }
}



template <typename NumericType>
void Matrix<NumericType>::add_inplace(const Matrix<NumericType> *m)
{
    if (cols != m->cols || rows != m->rows)
    {
        throw DimensionsIncompatibleException<NumericType>(this, m, "adding in place");
    }
    for (int i = 0; i < cols * rows; ++i) {
        cokoliv[i] += m->cokoliv[i];
    }
}



template <typename NumericType>
void Matrix<NumericType>::print_matrix(int decimal_places, int decimal_total) {
    for (int r = 0; r < this->rows; ++r) {
        for (int c = 0; c < this->cols; ++c) {
            cout << FIXED_FLOAT(this->cokoliv[this->cols * r + c], decimal_places, decimal_total) << ' ';
        }
        cout << endl;
    }
}

template <typename NumericType>
Matrix<NumericType>::~Matrix()
{
    delete[] cokoliv;
}

template <typename NumericType>
Matrix<NumericType>::Matrix(const Matrix<NumericType> &matrix) : rows(matrix.rows),
                                                    cols(matrix.cols),
                                                    cokoliv(new NumericType[matrix.rows * matrix.cols])
{
    memcpy(cokoliv, matrix.cokoliv, sizeof(NumericType) * rows * cols);
}

/* template<typename NumericType>
void Matrix<NumericType>::apply_function(NumericType (*f)(NumericType)) {
    for (int i = 0; i < rows * cols; ++i) {
        cokoliv[i] = (*f)(cokoliv[i]);
    }
} */



/*
 * Derivace Sigmoidy na x S'(x) je rovna S(x)*(1-S(x))
 *
 */
/*
template <typename NumericType>
void Matrix<NumericType>::reshape(int rows_, int cols_)
{
    if (rows_ * cols_ != this->rows * this->cols)
    {
        throw DimensionsIncompatibleException<NumericType>(this, this, "Dimensions not applicable! [When reshaping]");
    }
    this->rows = rows_;
    this->cols = cols_;
}

template <typename NumericType>
void Matrix<NumericType>::sub_add(const Matrix *m, Matrix<NumericType> *out)
{
    // Substituts m2 from m1 and adds the result to out
    if (cols != m->cols || rows != m->rows ) {
        throw DimensionsIncompatibleException<NumericType>(this, m, "sub_adding");
    } else if(cols != out->cols || rows != out->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, out, "storing results from sub_adding");
    }
    for (int i = 0; i < cols * rows; ++i)
    {
        out->cokoliv[i] = out->cokoliv[i] + cokoliv[i] - m->cokoliv[i];
    }
}*/

template <typename NumericType>
void Matrix<NumericType>::sub(const Matrix *m, Matrix<NumericType> *out)
{
    // Substituts m2 from m1 and writes the result to out
    if (cols != m->cols || rows != m->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, m, "subtracting");
    } else if (cols != out->cols || rows != out->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, out, "storing results from subtracting");
    }
    for (int i = 0; i < cols * rows; ++i) {
        out->cokoliv[i] = cokoliv[i] - m->cokoliv[i];
    }
}

// TODO: Remove this. Leave just the ew_multiply with correct calls
template <typename NumericType>
void Matrix<NumericType>::ew_multiply_inplace(const Matrix *m) {
    ew_multiply(m, this);
}

template <typename NumericType>
void Matrix<NumericType>::ew_multiply(const Matrix *m, Matrix<NumericType>* out) {
    if (cols != m->cols || rows != m->rows) {
        throw DimensionsIncompatibleException(this, m, "elementwise multiplying");
    } else if (out->cols != m->cols || out->rows != m->rows) {
        throw DimensionsIncompatibleException(m, out, "elementwise multiplying output");
    }
    for (int i = 0; i < cols * rows; ++i) {
        out->cokoliv[i] = cokoliv[i] * m->cokoliv[i];
    }
}

template <typename NumericType>
void Matrix<NumericType>::ew_multiply_add(const Matrix *m, Matrix<NumericType>* out) {
    if (cols != m->cols || rows != m->rows) {
        throw DimensionsIncompatibleException(this, m, "additive elementwise multiplying");
    } else if (out->cols != m->cols || out->rows != m->rows) {
        throw DimensionsIncompatibleException(m, out, "additive elementwise multiplying output");
    }
    for (int i = 0; i < cols * rows; ++i) {
        out->cokoliv[i] += cokoliv[i] * m->cokoliv[i];
    }
}

template <typename NumericType>
void Matrix<NumericType>::sub_inplace(const Matrix *m)
{
    if (cols != m->cols || rows != m->rows) {
        throw DimensionsIncompatibleException(this, m,"subtracting inplace");
    }
    for (int i = 0; i < cols * rows; ++i) {
        cokoliv[i] = cokoliv[i] - m->cokoliv[i];
    }
}

template<typename NumericType>
void Matrix<NumericType>::set_zero() {
    for (int i = 0; i < cols * rows; ++i)
    {
        cokoliv[i] = 0;
    }
}

template <typename NumericType>
void Matrix<NumericType>::ReLU_deriv() {
    for (int i = 0; i < rows * cols; ++i) {
        if (cokoliv[i] < 0) {
            cokoliv[i] = 0.00001;
        } else {
            cokoliv[i] = 1;
        }
    }
}

template<typename NumericType>
void Matrix<NumericType>::ReLU(Matrix<NumericType> *out) {
    for (int i = 0; i < rows * cols; ++i) {
        if(cokoliv[i] < 0) {
            out->cokoliv[i] = 0.00001*cokoliv[i];
        } else {
            out->cokoliv[i] = cokoliv[i];
        }
    }
}

template<typename NumericType>
void Matrix<NumericType>::TanH(Matrix<NumericType>* out) {
    for (int i = 0; i < rows*cols; ++i) {
        out->cokoliv[i] = tanh(cokoliv[i]);
    }
}


template<typename NumericType>
void Matrix<NumericType>::scale(NumericType scalar) {
    for (int i = 0; i < rows * cols; ++i) {
        cokoliv[i] *= scalar;
    }
}



template<typename NumericType>
void Matrix<NumericType>::SoftMax(Matrix<NumericType>* out) {
    NumericType m, sum, constant;
    m = cokoliv[0];
    int i;
    for (i = 1; i < rows*cols; ++i) {
        if (m < cokoliv[i]) {
            m = cokoliv[i];
        }
    }
    sum = 0.0;
    for (i = 0; i < rows*cols; ++i) {
        sum += exp(cokoliv[i] - m);
    }
    constant = m + log(sum);
    for (i = 0; i < rows*cols; ++i) {
        out->cokoliv[i] = exp(cokoliv[i] - constant);
    }
}



template<typename NumericType>
void Matrix<NumericType>::set_values(Matrix<NumericType>* source) {
    if (cols*rows != source->cols * source->rows) {
        throw DimensionsIncompatibleException<NumericType>(this, source, "copying values");
    }
    memcpy(cokoliv, source->cokoliv, sizeof(NumericType) * rows * cols);
}

template<typename NumericType>
void Matrix<NumericType>::set_values_unsafe(NumericType* source) {
    memcpy(this->cokoliv, source, sizeof(NumericType) * rows * cols);
}

template<typename NumericType>
void Matrix<NumericType>::set_values(NumericType value) {
    for (int i = 0; i < rows * cols; ++i) {
        cokoliv[i]=value;
    }
}

template<typename NumericType>
void Matrix<NumericType>::SigmF(Matrix<NumericType>* out) {
    for (int i = 0; i < rows * cols; ++i) {
        out->cokoliv[i] = 0.5+cokoliv[i]/((2.+abs(cokoliv[i]))*2.);
    }
}

template<typename NumericType>
void Matrix<NumericType>::square() {
    for (int i = 0; i < rows * cols; ++i) {
        cokoliv[i] *= cokoliv[i];
    }
}

/* constexpr float Q_rsqrt(float number) noexcept {
    static_assert(std::numeric_limits<float>::is_iec559); // (enable only on IEEE 754)

    auto const y = std::bit_cast<float>(
            0x5f3759df - (std::bit_cast<std::uint32_t>(number) >> 1));
    return y * (1.5f - (number * 0.5f * y * y));
} */

template<typename NumericType>
void Matrix<NumericType>::inverse_square_root() {
    for (int i = 0; i < rows * cols; ++i) {
        cokoliv[i] = 1.f/(sqrt(cokoliv[i])+0.00000001f);
    }
}

template<typename NumericType>
void Matrix<NumericType>::SigmF_deriv() {
    for (int i = 0; i < rows * cols; ++i) {
        cokoliv[i] = 1.f/(4.0*abs(cokoliv[i])+cokoliv[i]*cokoliv[i]+4.0);
    }
}

template<typename NumericType>
void Matrix<NumericType>::Sigm(Matrix<NumericType> *out) {
    for (int i = 0; i < rows * cols; ++i) {
        out->cokoliv[i] = 1.0/(1.0+exp(-cokoliv[i]));
    }
}

mt19937* getGen() {
    static std::random_device randDev;
    static std::mt19937 twister(randDev());
    return &twister;
}

template<typename NumericType>
NumericType getUniform(const NumericType& A, const NumericType& B) {
    static std::uniform_real_distribution<NumericType> dist;
    dist.param(std::uniform_real_distribution<>::param_type(A, B));
    return dist(*getGen());
}

template<typename NumericType>
NumericType getNormal(const NumericType& Mean, const NumericType& Variance) {
    static std::normal_distribution<NumericType> dist;
    dist.param(std::normal_distribution<>::param_type(Mean, Variance));
    return dist(*getGen());
}


template<typename NumericType>
void Matrix<NumericType>::init_xavier() {
    for (int i = 0; i < rows * cols; ++i) {
        // TODO vectorize: Inverse square root
        cokoliv[i] = (NumericType)(getUniform(-1. / sqrt(rows), 1. / sqrt(rows)));
    }
}

template<typename NumericType>
void Matrix<NumericType>::init_he() {
    for (int i = 0; i < rows * cols; ++i) {
        // TODO vectorize: Inverse square root
        cokoliv[i] = (NumericType)(getNormal(0., 1.41421356237 * 1./sqrt(rows)));
    }
}

template<typename NumericType>
void Matrix<NumericType>::ln() {
    for (int i = 0; i < rows * cols; ++i) {
        // TODO vectorize?
        cokoliv[i] = log(cokoliv[i]+0.0000000000000001);
    }
}

template<typename NumericType>
NumericType Matrix<NumericType>::sum() {
    NumericType sum = 0;
    for (int i = 0; i < rows * cols; ++i) {
        sum += cokoliv[i];
    }
    return sum;
}


template <typename NumericType>
void Matrix<NumericType>::draw_image() {
    int position_in_row = 0;
    char pixel = '?';

    for (int i = 0; i < 784; ++i) {

            if(this->cokoliv[i] < 0.01f){
                pixel = '`';
            } 
            else if (this->cokoliv[i] > 0.01f && this->cokoliv[i] < 0.2f){
                pixel = '~';
            }
            else if (this->cokoliv[i] > 0.2f && this->cokoliv[i] < 0.5f){
                pixel = '*';
            }
            else if (this->cokoliv[i] > 0.5f && this->cokoliv[i] < 1.01f){
                pixel = '#';
            }
        
            cout << FIXED_FLOAT(pixel, 0, 3);

        if (position_in_row == 27){
            cout << endl;
            position_in_row = 0;
        }
        else {
            position_in_row++;
        }


    }
    cout << "\n\n";

}


template class AlgebraLibrary::Matrix<float>;
