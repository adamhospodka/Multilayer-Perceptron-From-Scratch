//
// Created by adamb on 29.09.2021.
//

#include <boost/test/unit_test.hpp>
#include "../src/Algebra/Matrix.h"
#include "../src/Algebra/DimensionsIncompatibleException.hpp"

using namespace AlgebraLibrary;

struct F2 {
    float arr1[9] = {1,1,1,1,1,1,1,1,1};
    float arr0[9] = {0,0,0,0,0,0,0,0,0};
    float arrA[9] = {1,2,3,1,2,3,1,2,3};
    float arrB[9] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    Matrix<float> m0;
    Matrix<float> m1;
    Matrix<float> mA;
    Matrix<float> mB;
    F2() :
    m1(Matrix<float>(3,3,arr1)),
    m0(Matrix<float>(3,3,arr0)),
    mA(Matrix<float>(3,3,arrA)),
    mB(Matrix<float>(3,3,arrB))
    {
        //BOOST_TEST_MESSAGE("Setting up test matrices");
    }
    ~F2() {
        //BOOST_TEST_MESSAGE("Tearing down test matrices");
    }
};

struct BTF_DimMismatch {
    float arr1[9] = {1,1,1,1,1,1,1,1,1};
    float arr0[9] = {0,0,0,0,0,0,0,0,0};
    float arrA[10] = {1,2,3,1,2,3,1,2,3,-5};
    Matrix<float> m0;
    Matrix<float> m1;
    Matrix<float> mA;
    BTF_DimMismatch() :
            m1(Matrix<float>(3,3,arr1)),
            m0(Matrix<float>(3,3,arr0)),
            mA(Matrix<float>(2,5,arrA))
    {
        //BOOST_TEST_MESSAGE("Setting up test matrices");
    }
    ~BTF_DimMismatch() {
        //BOOST_TEST_MESSAGE("Tearing down test matrices");
    }
};


BOOST_FIXTURE_TEST_SUITE(AdditionSubtractionBasicNumericCorrectness, F2)

    BOOST_AUTO_TEST_CASE(AddZerosToOnes) {
        BOOST_REQUIRE_NO_THROW(m0.add_inplace(&m1));
        BOOST_CHECK_EQUAL_COLLECTIONS(m0.cokoliv, m0.cokoliv + 9, m1.cokoliv, m1.cokoliv + 9);
    }

    BOOST_AUTO_TEST_CASE(AddZerosToZeros) {
        Matrix<float> m0_2 = m0;
        BOOST_REQUIRE_NO_THROW(m0_2.add_inplace(&m0));
        BOOST_CHECK_EQUAL_COLLECTIONS(m0_2.cokoliv, m0_2.cokoliv + 9, m0.cokoliv, m0.cokoliv + 9);
        BOOST_REQUIRE_NO_THROW(m0_2.add_inplace(&m0_2));
        BOOST_CHECK_EQUAL_COLLECTIONS(m0_2.cokoliv, m0_2.cokoliv + 9, m0.cokoliv, m0.cokoliv + 9);
    }
    BOOST_AUTO_TEST_CASE(SubZerosToZeros) {
        Matrix<float> m0_2 = m0;
        BOOST_REQUIRE_NO_THROW(m0_2.sub_inplace(&m0));
        BOOST_CHECK_EQUAL_COLLECTIONS(m0_2.cokoliv, m0_2.cokoliv + 9, m0.cokoliv, m0.cokoliv + 9);
        BOOST_REQUIRE_NO_THROW(m0_2.sub_inplace(&m0_2));
        BOOST_CHECK_EQUAL_COLLECTIONS(m0_2.cokoliv, m0_2.cokoliv + 9, m0.cokoliv, m0.cokoliv + 9);
    }

    BOOST_AUTO_TEST_CASE(AddSubCancelOut1) {
        Matrix<float> m1_2 = m1;
        BOOST_REQUIRE_NO_THROW(m1_2.sub_inplace(&m1));
        BOOST_REQUIRE_NO_THROW(m1_2.add_inplace(&m1));
        BOOST_CHECK_EQUAL_COLLECTIONS(m1_2.cokoliv, m1_2.cokoliv + 9, m1.cokoliv, m1.cokoliv + 9);

    }
    BOOST_AUTO_TEST_CASE(AddSubCancelOut2) {
        Matrix<float> m1_2 = m1;
        Matrix<float> out(3, 3);
        BOOST_REQUIRE_NO_THROW(m1_2.sub(&m1, &out));
        BOOST_REQUIRE_NO_THROW(out.add_inplace(&m1));
        BOOST_CHECK_EQUAL_COLLECTIONS(out.cokoliv, out.cokoliv + 9, m1.cokoliv, m1.cokoliv + 9);

    }

    BOOST_AUTO_TEST_CASE(AddArbitrary) {
        BOOST_REQUIRE_NO_THROW(mA.add_inplace(&mB));
        BOOST_REQUIRE_NO_THROW(m1.add_inplace(&m1));
        BOOST_CHECK_EQUAL_COLLECTIONS(mA.cokoliv, mA.cokoliv + 9, m1.cokoliv, m1.cokoliv + 9);
}
    BOOST_AUTO_TEST_CASE(SubArbitrary) {
        BOOST_REQUIRE_NO_THROW(m1.add_inplace(&m1));
        BOOST_REQUIRE_NO_THROW(m1.sub_inplace(&mA));
        BOOST_CHECK_EQUAL_COLLECTIONS(m1.cokoliv, m1.cokoliv + 9, mB.cokoliv, mB.cokoliv + 9);
    }
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(MADimensionChecks, BTF_DimMismatch)
    BOOST_AUTO_TEST_CASE(DimensionCheck1) {
        BOOST_REQUIRE_THROW(mA.add_inplace(&m1), DimensionsIncompatibleException<float>);
        BOOST_REQUIRE_NO_THROW(m1.add_inplace(&m1));
    }
    BOOST_AUTO_TEST_CASE(DimensionCheck2) {
        BOOST_REQUIRE_THROW(mA.add_inplace(&m0), DimensionsIncompatibleException<float>);
        BOOST_REQUIRE_NO_THROW(mA.add_inplace(&mA));
    }
    BOOST_AUTO_TEST_CASE(DimensionCheck3) {
        BOOST_REQUIRE_NO_THROW(m0.add_inplace(&m1));
    }
BOOST_AUTO_TEST_SUITE_END()