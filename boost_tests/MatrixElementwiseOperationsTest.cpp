//
// Created by adamb on 29.09.2021.
//

#include <boost/test/unit_test.hpp>
#include "../src/Algebra/Matrix.h"
#include "../src/Algebra/DimensionsIncompatibleException.hpp"

using namespace AlgebraLibrary;

struct BTF_EWOperations {

    float arr32_0[32] = {1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8};
    float arr32_1[32] = {7,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0};
    float arr32_0x1[32] = {7,12,15,16,15,12,7,0,7,12,15,16,15,12,7,0,7,12,15,16,15,12,7,0,7,12,15,16,15,12,7,0};
    Matrix<float> A, At, B, Bt, O, Ot;

    BTF_EWOperations():
        A(Matrix<float>(1, 32, arr32_0)),
        B(Matrix<float>(1, 32, arr32_1)),
        At(Matrix<float>( 32, 1, arr32_0)),
        Bt(Matrix<float>(32, 1, arr32_1)),
        O(Matrix<float>(1, 32)),
        Ot(Matrix<float>(32, 1))
        {
        //BOOST_TEST_MESSAGE("Setting up test matrices");
    }

    ~BTF_EWOperations() {
        //BOOST_TEST_MESSAGE("Tearing down test matrices");
    }
};


BOOST_FIXTURE_TEST_SUITE(EWMBasicNumericCorrectness, BTF_EWOperations)

    BOOST_AUTO_TEST_CASE(EWMultiply) {
        BOOST_REQUIRE_NO_THROW( A.ew_multiply(&B, &O));
        BOOST_REQUIRE_NO_THROW( At.ew_multiply(&Bt, &Ot));
        BOOST_CHECK_EQUAL_COLLECTIONS(Ot.cokoliv, Ot.cokoliv + 32, arr32_0x1, arr32_0x1 + 32);
        BOOST_CHECK_EQUAL_COLLECTIONS(O.cokoliv, O.cokoliv + 32, arr32_0x1, arr32_0x1 + 32);
    }

    BOOST_AUTO_TEST_CASE(EWDimensionChecks) {
        BOOST_REQUIRE_THROW( At.ew_multiply(&B, &O), DimensionsIncompatibleException<float>);
        BOOST_REQUIRE_THROW( A.ew_multiply(&Bt, &O), DimensionsIncompatibleException<float>);
        BOOST_REQUIRE_THROW( A.ew_multiply(&B, &Ot), DimensionsIncompatibleException<float>);
        BOOST_REQUIRE_THROW( At.ew_multiply(&B, &Ot), DimensionsIncompatibleException<float>);
        BOOST_REQUIRE_THROW( At.ew_multiply(&Bt, &O), DimensionsIncompatibleException<float>);

    }

BOOST_AUTO_TEST_SUITE_END()



