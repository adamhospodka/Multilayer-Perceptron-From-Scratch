#ifndef NNLIB_ALGEBRA_H
#define NNLIB_ALGEBRA_H

#include <ostream>
#include <iomanip>
#define FIXED_FLOAT(x, d, w) std::fixed << std::setw( w ) << std::setprecision( d ) << (x)


namespace AlgebraLibrary {

    template<typename NumericType> class Matrix;
    template<typename NumericType> std::ostream& operator<< (std::ostream& o, const Matrix<NumericType>& m);


    template<typename NumericType>
    class Matrix {



    public:
        friend std::ostream &operator<< <>(std::ostream & os, const Matrix<NumericType> & m);
 
        NumericType *cokoliv;  // the "const" is for the pointer, not for the values behind it.
        // Option => std::vector<NumericType> cokoliv;
        int rows;
        int cols;

        Matrix(const Matrix<NumericType> &);

        Matrix(int rows, int cols, NumericType *cokoliv);

        Matrix<NumericType>(int rows, int cols);

        void set_values(Matrix<NumericType>* source);

        /**
         * Basic matrix multiplication. Multiplies this matrix by another matrix from the right.
         * @param m matrix to multiply by
         * @return
         */
        void multiply(const Matrix<NumericType> *m, Matrix<NumericType> *out);

        void multiply_add(const Matrix<NumericType> *m, Matrix<NumericType> *out);

        void multiply_swapped_second(const Matrix<NumericType> *m, Matrix<NumericType> *out);
        void multiply_swapped_first(const Matrix<NumericType> *m, Matrix<NumericType> *out);
        void multiply_swapped_first_add(const Matrix<NumericType> *m, Matrix<NumericType> *out);

        //void multiply_add_swapped(const Matrix<NumericType> *m, Matrix<NumericType> *out);

        void add_inplace(const Matrix<NumericType> *m);

        void ew_multiply_inplace(const Matrix<NumericType> *m);

        void ew_multiply(const Matrix *m, Matrix<NumericType> *out);
        void ew_multiply_add(const Matrix *m, Matrix<NumericType> *out);

        void sub_inplace(const Matrix *m);

        //void sub_add(const Matrix *m, Matrix<NumericType> *out);

        void sub(const Matrix *m, Matrix<NumericType> *out);



        void init_xavier();

        void set_zero();

        void ReLU(Matrix<NumericType> *out);

        void ReLU_deriv();

        void TanH(Matrix<NumericType> *out);

        void TanH_deriv();

        void SigmF(Matrix<NumericType> *out);
        void SigmF_deriv();

        void Sigm(Matrix<NumericType> *out);
        void SoftMax(Matrix<NumericType> *out);

        void ln();

        NumericType sum();


        void print_matrix(int decimal_places = 2, int decimal_total = 2);


        virtual ~Matrix();

        void set_values_unsafe(NumericType *source);

        void scale(NumericType scalar);
        //void increase(NumericType scalar);
        //void ew_divide_add(Matrix<NumericType> divisor, Matrix<NumericType> out);
        void set_values(NumericType value);

        void square();
        //void constexpr square_root() noexcept;
        void inverse_square_root();

        void init_he();
        
        void draw_image();
    };



    template<typename U>
    std::ostream &operator<<(std::ostream &os, const Matrix<U> &m) {

        if(m.rows >1) {
            os << "Matrix(" << m.rows << "x" << m.cols << "): ";
            os << std::endl;
        }
        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                os << FIXED_FLOAT(m.cokoliv[m.cols * r + c], 4, 6) << ' ';
            }
            if(m.rows >1) {
                os << std::endl;
            }
        }
        return os;
    }

}
//#include "Matrix.cpp"
#endif //NNLIB_ALGEBRA_H
