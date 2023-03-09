//
// Created by adamb on 29.09.2021.
//

#include <boost/test/unit_test.hpp>
#include "../src/Algebra/Matrix.h"

using namespace AlgebraLibrary;

struct BTF_ComposedOperations {

    float arr1[12] = {0, 1, 2, 0, 3, 4, 0, 1, 2, 0, 3, 4};
    float res3x3[9] = {0, 5, 8, 0, 18, 26, 0, 17, 26};
    float res4x4[16] = {7, 4, 6, 9, 17, 12, 12, 19, 7, 4, 6, 9, 17, 12, 12, 19};
    float arrU3[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float arrU4[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    Matrix<float> m3x4;
    Matrix<float> m4x3;
    Matrix<float> out3x3;
    Matrix<float> out4x4;
    Matrix<float> u3x3;
    Matrix<float> u4x4;
    Matrix<float> out2x2;

    BTF_ComposedOperations() :
            m3x4(Matrix(3, 4, arr1)),
            m4x3(Matrix(4, 3, arr1)),
            out3x3(Matrix<float>(3, 3)),
            out4x4(Matrix<float>(4, 4)),
            u3x3(Matrix(3, 3, arrU3)),
            u4x4(Matrix(4, 4, arrU4)),
            out2x2(Matrix<float>(2, 2)) {
        //BOOST_TEST_MESSAGE("Setting up test matrices");
    }

    ~BTF_ComposedOperations() {
        //BOOST_TEST_MESSAGE("Tearing down test matrices");
    }
};


BOOST_FIXTURE_TEST_SUITE(MMBasicNumericCorrectness, BTF_ComposedOperations)

    BOOST_AUTO_TEST_CASE(MultiplicationAdditionBasic1) {
        m3x4.multiply(&m4x3, &out3x3);
        m3x4.multiply_add(&m4x3, &out3x3);
        Matrix<float> out3x3_2(3, 3, res3x3);
        out3x3_2.add_inplace(&out3x3_2);
        BOOST_CHECK_EQUAL_COLLECTIONS(out3x3_2.cokoliv, out3x3_2.cokoliv + 9, out3x3.cokoliv, out3x3.cokoliv + 9);

    }

BOOST_AUTO_TEST_SUITE_END()



