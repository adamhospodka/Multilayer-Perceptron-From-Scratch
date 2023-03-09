//
// Created by adamb on 07.11.2021.
//

#include <boost/test/unit_test.hpp>
#include "../src/Algebra/Matrix.h"

using namespace AlgebraLibrary;

struct BTF_DataAlignment {
    float arr1[26] = {1000, 0, 1, 2, 0, 3, 4, 0, 1, 2, 0, 3, 4, 1000, 0, 1, 2, 0, 3, 4, 0, 1, 2, 0, 3, 4};
    BTF_DataAlignment()
    {
        //BOOST_TEST_MESSAGE("Setting up test matrices");
    }
    ~BTF_DataAlignment() {
        //BOOST_TEST_MESSAGE("Tearing down test matrices");
    }
};

BOOST_FIXTURE_TEST_SUITE(DataAlignment, BTF_DataAlignment)
    BOOST_AUTO_TEST_CASE(AllocateMatricesFromShiftedFloatArray) {
        const int leng = 14;
        Matrix<float> *pls[leng];
        for (int i = 0; i < leng; ++i) {
            pls[i] = new Matrix(1, 12, arr1 + i);
            void* ptri = (void *) (pls[i]->cokoliv);
            BOOST_WARN((size_t) (ptri) % 16 == 0);
            if(i % (16/sizeof(float)) == 0) {
                BOOST_WARN((size_t) (arr1 + i) % 16 == 0);
            } else {
                BOOST_WARN((size_t) (arr1 + i) % 16 != 0);
            }
        }
        for (int i = 0; i < leng; ++i) {
            delete pls[i] ;
        }
    }
BOOST_AUTO_TEST_SUITE_END()