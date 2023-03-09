#include "Optimizer.hpp"
#include <iostream>

//
// Created by adamb on 25.11.2021.
//
using namespace std;
template <typename NT>
class SGD: public Optimizer<NT>{
public:
    //NT learning_rate;
    explicit SGD(NT learning_rate): Optimizer<NT>("Salvador Gonzales Delgado", learning_rate) {
        cout << this->name << ": *emerges from the shadow*" << endl;
    }

    void prepare_memory() override {
        cout << this->name << ": \"I don't need any memory, gringo. \"" << endl;

    }

    void modify_weights() override {
        for (int param_matrix_id = 0; param_matrix_id < this->parameter_matrices_references.size(); ++param_matrix_id) {
            this->gradient_matrices_references[param_matrix_id]->scale(this->learning_rate);
            //cout << *this->gradient_matrices_references[param_matrix_id] << endl;
            this->parameter_matrices_references[param_matrix_id]->sub_inplace(
                    this->gradient_matrices_references[param_matrix_id]
                    );
        }
    }


};
