//
// Created by adamb on 01.11.2021.
//


#include <boost/test/unit_test.hpp>
#include "../src/Algebra/Matrix.h"
#include "../src/Algebra/DimensionsIncompatibleException.hpp"

using namespace AlgebraLibrary;

struct BTF_MultiplicationAddition {

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

    BTF_MultiplicationAddition() :
            m3x4(Matrix(3, 4, arr1)),
            m4x3(Matrix(4, 3, arr1)),
            out3x3(Matrix<float>(3, 3)),
            out4x4(Matrix<float>(4, 4)),
            out2x2(Matrix<float>(2, 2)),
            u3x3(Matrix(3, 3, arrU3)),
            u4x4(Matrix(4, 4, arrU4)) {
        //BOOST_TEST_MESSAGE("Setting up test matrices");
    }

    ~BTF_MultiplicationAddition() {
        //BOOST_TEST_MESSAGE("Tearing down test matrices");
    }
};


BOOST_FIXTURE_TEST_SUITE(MMA_NumericalCorrectness, BTF_MultiplicationAddition)

    BOOST_AUTO_TEST_CASE(MulAddSameAsMulAndAdd1) {

            m3x4.multiply(&m4x3, &out3x3);
            out3x3.add_inplace(&u3x3);
            m3x4.multiply_add(&m4x3, &u3x3);
            BOOST_CHECK_EQUAL_COLLECTIONS(u3x3.cokoliv, u3x3.cokoliv + 9, out3x3.cokoliv, out3x3.cokoliv + 9);

    }

    BOOST_AUTO_TEST_CASE(MulAddSameAsMulAndAdd2) {
        m4x3.multiply(&m3x4, &out4x4);
        out4x4.add_inplace(&u4x4);

        m4x3.multiply_add(&m3x4, &u4x4);
        BOOST_CHECK_EQUAL_COLLECTIONS(u4x4.cokoliv, u4x4.cokoliv + 9, out4x4.cokoliv, out4x4.cokoliv + 9);

    }

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(MMA_DimensionsMatching, BTF_MultiplicationAddition)

    BOOST_AUTO_TEST_CASE(IncorrectDimensions1) {

        BOOST_REQUIRE_THROW(m3x4.multiply_add(&out2x2, &out3x3), DimensionsIncompatibleException<float>);
    }

    BOOST_AUTO_TEST_CASE(IncorrectDimensions2) {
        BOOST_REQUIRE_THROW(m3x4.multiply_add(&m4x3, &out2x2), DimensionsIncompatibleException<float>);
    }

    BOOST_AUTO_TEST_CASE(CorrectDimensions1) {
        BOOST_REQUIRE_NO_THROW(m3x4.multiply_add(&m4x3, &out3x3));
    }

    BOOST_AUTO_TEST_CASE(CorrectDimensions2) {
        BOOST_REQUIRE_NO_THROW(m4x3.multiply_add(&m3x4, &out4x4));
    }

    BOOST_AUTO_TEST_CASE(IncorrectDimensions3) {

        BOOST_REQUIRE_THROW(m4x3.multiply_add(&m3x4, &out3x3), DimensionsIncompatibleException<float>);
    }

BOOST_AUTO_TEST_SUITE_END()


