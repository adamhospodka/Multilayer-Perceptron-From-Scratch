//
// Created by adamb on 29.09.2021.
//

#define BOOST_TEST_MODULE Matrix Operations

#include <boost/test/included/unit_test.hpp>
#include "../src/Algebra/Matrix.h"
#include "../src/Algebra/DimensionsIncompatibleException.hpp"

using namespace AlgebraLibrary;
typedef Matrix<float> M;

struct BTF_Multiplication {

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

    BTF_Multiplication() :
            m3x4(Matrix(3, 4, arr1)),
            m4x3(Matrix(4, 3, arr1)),
            out3x3(Matrix<float>(3, 3)),
            out4x4(Matrix<float>(4, 4)),
            out2x2(Matrix<float>(2, 2)),
            u3x3(Matrix(3, 3, arrU3)),
            u4x4(Matrix(4, 4, arrU4)) {
        //BOOST_TEST_MESSAGE("Setting up test matrices");
    }

    ~BTF_Multiplication() {
        //BOOST_TEST_MESSAGE("Tearing down test matrices");
    }
};


BOOST_FIXTURE_TEST_SUITE(MMBasicNumericCorrectness, BTF_Multiplication)

    BOOST_AUTO_TEST_CASE(MultiplicationBasic1) {
        m3x4.multiply(&m4x3, &out3x3);
        BOOST_CHECK_EQUAL_COLLECTIONS(res3x3, res3x3 + 9, out3x3.cokoliv, out3x3.cokoliv + 9);
    }

    BOOST_AUTO_TEST_CASE(MultiplicationBasic2) {
        m4x3.multiply(&m3x4, &out4x4);
        BOOST_CHECK_EQUAL_COLLECTIONS(res4x4, res4x4 + 16, out4x4.cokoliv, out4x4.cokoliv + 16);
    }

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(MM_DimensionsMatching, BTF_Multiplication)

    BOOST_AUTO_TEST_CASE(IncorrectDimensions1) {

        BOOST_REQUIRE_THROW(m3x4.multiply(&out2x2, &out3x3), DimensionsIncompatibleException<float>);
    }

    BOOST_AUTO_TEST_CASE(IncorrectDimensions2) {
        BOOST_REQUIRE_THROW(m3x4.multiply(&m4x3, &out2x2), DimensionsIncompatibleException<float>);
    }

    BOOST_AUTO_TEST_CASE(CorrectDimensions1) {
        BOOST_REQUIRE_NO_THROW(m3x4.multiply(&m4x3, &out3x3));
    }

    BOOST_AUTO_TEST_CASE(CorrectDimensions2) {
        BOOST_REQUIRE_NO_THROW(m4x3.multiply(&m3x4, &out4x4));
    }

    BOOST_AUTO_TEST_CASE(IncorrectDimensions3) {

        BOOST_REQUIRE_THROW(m4x3.multiply(&m3x4, &out3x3), DimensionsIncompatibleException<float>);
    }

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(UnitaryDoesNotChangeValues, BTF_Multiplication)

    BOOST_AUTO_TEST_CASE(MulByUnitary1) {
        Matrix<float> out4x3 = Matrix<float>(4, 3);
        BOOST_REQUIRE_NO_THROW(m4x3.multiply(&u3x3, &out4x3));
        BOOST_CHECK_EQUAL_COLLECTIONS(m4x3.cokoliv, m4x3.cokoliv + 12, out4x3.cokoliv, out4x3.cokoliv + 12);
    }

    BOOST_AUTO_TEST_CASE(MulByUnitary2) {
        Matrix<float> out4x3 = Matrix<float>(3, 4);
        BOOST_REQUIRE_NO_THROW(m3x4.multiply(&u4x4, &out4x3));
        BOOST_CHECK_EQUAL_COLLECTIONS(m4x3.cokoliv, m4x3.cokoliv + 12, out4x3.cokoliv, out4x3.cokoliv + 12);
    }

    BOOST_AUTO_TEST_CASE(UnitarySquared1) {
        float arr1x1[1] = {1};
        Matrix<float> u1x1 = Matrix<float>(1, 1, arr1x1);
        Matrix<float> out1x1 = Matrix<float>(1, 1);

        BOOST_REQUIRE_NO_THROW(u1x1.multiply(&u1x1, &out1x1));
        BOOST_CHECK_EQUAL_COLLECTIONS(u1x1.cokoliv, u1x1.cokoliv + 1, out1x1.cokoliv, out1x1.cokoliv + 1);
    }

    BOOST_AUTO_TEST_CASE(UnitarySquared2) {
        float arr2x2[4] = {1, 0, 0, 1};
        Matrix<float> u2x2 = Matrix<float>(2, 2, arr2x2);
        Matrix<float> out2x2 = Matrix<float>(2, 2);

        BOOST_REQUIRE_NO_THROW(u2x2.multiply(&u2x2, &out2x2));
        BOOST_CHECK_EQUAL_COLLECTIONS(u2x2.cokoliv, u2x2.cokoliv + 4, out2x2.cokoliv, out2x2.cokoliv + 4);
    }

    BOOST_AUTO_TEST_CASE(UnitarySquared3) {
        Matrix<float> out3x3 = Matrix<float>(3, 3);

        BOOST_REQUIRE_NO_THROW(u3x3.multiply(&u3x3, &out3x3));
        BOOST_CHECK_EQUAL_COLLECTIONS(u3x3.cokoliv, u3x3.cokoliv + 9, out3x3.cokoliv, out3x3.cokoliv + 9);
    }

    BOOST_AUTO_TEST_CASE(UnitarySquared4) {
        Matrix<float> out4x4 = Matrix<float>(4, 4);

        BOOST_REQUIRE_NO_THROW(u4x4.multiply(&u4x4, &out4x4));
        BOOST_CHECK_EQUAL_COLLECTIONS(u4x4.cokoliv, u4x4.cokoliv + 16, out4x4.cokoliv, out4x4.cokoliv + 16);

    }
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(SwapedMultiplication, BTF_Multiplication)
    BOOST_AUTO_TEST_CASE(SwapVectorDimensionsCheckUnitary) {
        Matrix<float> vec4 = Matrix<float>(1, 4, arr1);
        Matrix<float> vec4_2 = Matrix<float>(4, 1, arr1);

        BOOST_CHECK_EQUAL_COLLECTIONS(vec4.cokoliv, vec4.cokoliv + 4, vec4_2.cokoliv, vec4_2.cokoliv + 4);

        Matrix<float> out1x4 = Matrix<float>(1, 4);
        Matrix<float> out4x1 = Matrix<float>(4, 1);
        Matrix<float> out4x4 = Matrix<float>(4,4);
        Matrix<float> out4x4_2 = Matrix<float>(4,4);


        BOOST_REQUIRE_NO_THROW(u4x4.multiply(&vec4_2, &out4x1));
        BOOST_REQUIRE_NO_THROW(u4x4.multiply_swapped_second(&vec4, &out1x4));

        BOOST_REQUIRE_NO_THROW(vec4_2.multiply(&vec4, &out4x4));
        BOOST_REQUIRE_NO_THROW(vec4.multiply_swapped_first(&vec4, &out4x4_2));

        BOOST_CHECK_EQUAL_COLLECTIONS(out4x1.cokoliv, out4x1.cokoliv + 4, vec4_2.cokoliv, vec4_2.cokoliv + 4);
        BOOST_CHECK_EQUAL_COLLECTIONS(vec4.cokoliv, vec4.cokoliv + 4, out1x4.cokoliv, out1x4.cokoliv + 4);
        BOOST_CHECK_EQUAL_COLLECTIONS(out1x4.cokoliv, out1x4.cokoliv + 4, out4x1.cokoliv, out4x1.cokoliv + 4);
        BOOST_CHECK_EQUAL_COLLECTIONS(out4x4.cokoliv, out4x4.cokoliv + 16, out4x4_2.cokoliv, out4x4_2.cokoliv + 16);

    }

    BOOST_AUTO_TEST_CASE(SwapVectorGeneralCorrectness) {
        Matrix<float> vec4 = Matrix<float>(1, 4, arr1);
        Matrix<float> vec4_2 = Matrix<float>(4, 1, arr1);
        float arr16[16] = {1, -1, -2, 0, 2, 3, 0, -3, 0, 1, 0, 2, -3, -1, 10, 1};
        Matrix<float> m4x4 = Matrix<float>(4, 4, arr16);
        Matrix<float> out1x4 = Matrix<float>(1, 4);
        Matrix<float> out4x1 = Matrix<float>(4, 1);
        Matrix<float> out1x4_2 = Matrix<float>(1, 4);

        BOOST_REQUIRE_NO_THROW(m4x4.multiply(&vec4_2, &out4x1));
        BOOST_REQUIRE_NO_THROW(m4x4.multiply_swapped_second(&vec4, &out1x4));
        BOOST_CHECK_EQUAL_COLLECTIONS(out1x4.cokoliv, out1x4.cokoliv + 4, out4x1.cokoliv, out4x1.cokoliv + 4);

        BOOST_REQUIRE_NO_THROW(vec4.multiply(&m4x4, &out1x4));
        BOOST_REQUIRE_NO_THROW(vec4_2.multiply_swapped_first(&m4x4, &out1x4_2));
        BOOST_CHECK_EQUAL_COLLECTIONS(out1x4.cokoliv, out1x4.cokoliv + 4, out1x4_2.cokoliv, out1x4_2.cokoliv + 4);

    }

    BOOST_AUTO_TEST_CASE(SwapVectorGeneralCorrectness2) {

        float arrW[6] = {1, 2, 3, 4, 3, 2};
        float arrY[4] = {1, 2, 1, 2};
        float res[3] = {5, 11, 7};
        M m1 = M(3, 2, arrW);
        M m2 = M(1, 2, arrY);
        M o = M(1,3);
        o.set_zero();
        m1.multiply_swapped_second(&m2, &o);
        BOOST_CHECK_EQUAL_COLLECTIONS(o.cokoliv, o.cokoliv + 3, res, res + 3);

    }

    BOOST_AUTO_TEST_CASE(SwapVectorGeneralCorrectness3) {

        float arrW[6] = {1, 2, 3, 4, 3, 2};
        float arrY[1] = {2};
        float res[3] = {2, 4, 6};
        M m1 = M(3, 1, arrW);
        M m2 = M(1, 1, arrY);
        M o = M(1,3);
        o.set_zero();
        m1.multiply_swapped_second(&m2, &o);
        BOOST_CHECK_EQUAL_COLLECTIONS(o.cokoliv, o.cokoliv + 3, res, res + 3);

    }
BOOST_AUTO_TEST_SUITE_END()

