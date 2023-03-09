//
// Created by adamb on 07.10.2021.
//

#ifndef NN_CLASSIFIER_H
#define NN_CLASSIFIER_H

#define ERRORF_MSE 0
#define ERRORF_BCE 1

#include <vector>
#include <memory>
#include <random>
#include "../Algebra/Matrix.h"

using namespace AlgebraLibrary;
using namespace std;

template <typename NumericType>
class Classifier {
public:
    mt19937 gen;
    unsigned char error_function;
    vector<unsigned int> stochastic_mask;
    explicit Classifier(unsigned seed, unsigned char errf);

    /**
     * Implement the batch version, this one will just be a syntax shortcut using batch of size 1.
     * @tparam NumericType
     * @param input
     */
    virtual void train_one(vector<Matrix<NumericType>>& data_features, vector<Matrix<NumericType>>& data_targets) {
        train_batch(1, data_features, data_targets);
    }

    virtual void train_batch(int batch_size, vector<Matrix<NumericType>> &, vector<Matrix<NumericType>> &) = 0;

    virtual NumericType loss(Matrix<NumericType> &) = 0;

    virtual const Matrix<NumericType> * predict(Matrix<NumericType>* input) = 0;
};



#endif //NN_CLASSIFIER_H
