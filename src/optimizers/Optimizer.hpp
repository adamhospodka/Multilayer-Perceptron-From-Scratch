//
// Created by adamb on 24.11.2021.
//

#ifndef NN_OPTIMIZER_HPP
#define NN_OPTIMIZER_HPP
#include <vector>
#include "Matrix.h"

using namespace AlgebraLibrary;
using namespace std;


template<typename NumericType>
class Optimizer {
    typedef Matrix<NumericType> M;
public:
    const char* name;
    NumericType learning_rate;
    vector<M*> parameter_matrices_references;
    vector<M*> gradient_matrices_references;
    explicit Optimizer(const char* name="Patrick", NumericType learning_rate=0.03): name(name), learning_rate(learning_rate) {};

    virtual void register_model_parameters(vector<M> &parameters_matrices, vector<M> &parameters_gradients_matrices) {
        //unsigned int num_params = parameter_matrices.size();
        // increase capacity
        parameter_matrices_references.reserve(parameter_matrices_references.size() + parameters_matrices.size());
        gradient_matrices_references.reserve(parameter_matrices_references.size() + parameters_matrices.size());

        //auto iter = parameter_matrices.begin();
        for (M &mat_ref: parameters_matrices) {
            parameter_matrices_references.push_back(&mat_ref);
        }
        for (M &mat_ref: parameters_gradients_matrices) {
            gradient_matrices_references.push_back(&mat_ref);
        }
        cout << name << ": *registered " << parameters_matrices.size() <<  " parameters and "
        << parameters_gradients_matrices.size() << " gradient matrix references*" << endl;

    };
    virtual void prepare_memory() = 0;
    virtual void modify_weights() = 0;
};

template class Optimizer<float>;
#endif //NN_OPTIMIZER_HPP
