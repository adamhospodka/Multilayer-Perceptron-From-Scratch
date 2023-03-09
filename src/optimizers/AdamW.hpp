//
// Created by adamb on 24.11.2021.
//

#ifndef NN_ADAMW_HPP
#define NN_ADAMW_HPP


#include "Optimizer.hpp"


template <typename NT>
class AdamW: public Optimizer<NT> {
    typedef Matrix<NT> Mx;
public:
    NT beta1;
    NT beta2;
    //NT epsilon;
    NT beta1_pow_t;
    NT beta2_pow_t;
    //unsigned int t;
    NT weight_decay;

    vector<Mx> Ms;
    vector<Mx> Vs;
    vector<Mx> Mcors;
    vector<Mx> Vcors;

    AdamW<NT>(NT learning_rate, NT beta1, NT beta2, NT weight_decay, const char* name="Adam Wong"): Optimizer<NT>(name, learning_rate),
    beta1(beta1), beta2(beta2), weight_decay(weight_decay),
    //epsilon(epsilon),
    //t(0),
    beta1_pow_t(beta1),
    beta2_pow_t(beta2) {
        cout << this->name << ": *draws his katana*" << endl;
    }

public:
    void prepare_memory() override {
        cout << this->name << ": *smiles*" << endl;
        cout << this->name << ": \"Hō... mukatte kuru no ka...\"" << endl;

        Ms.reserve(this->parameter_matrices_references.size());
        Vs.reserve(this->parameter_matrices_references.size());
        Mcors.reserve(this->parameter_matrices_references.size());
        Vcors.reserve(this->parameter_matrices_references.size());
        // for each Matrix registered as a parameter, prepare a matrix of support values
        for (const Mx* m_ref: this->parameter_matrices_references) {
            Ms.emplace_back(*m_ref);
            Vs.emplace_back(*m_ref);

            Mcors.emplace_back(*m_ref);
            Vcors.emplace_back(*m_ref);
        }
        cout << this->name << ": \"Roadroller-da!!!\"" << endl;

    }

    void modify_weights() override {
        for (int param_index_ = 0; param_index_ < this->gradient_matrices_references.size(); ++param_index_) {
            auto G = this->gradient_matrices_references[param_index_];
            Ms[param_index_].scale(beta1/(1.-beta1));
            Ms[param_index_].add_inplace(G);
            Ms[param_index_].scale(1.-beta1);

            Vs[param_index_].scale(beta2/(1.-beta2));
            G->square();
            Vs[param_index_].add_inplace(G);
            Vs[param_index_].scale(1.-beta2);

            Mcors[param_index_].set_values_unsafe(Ms[param_index_].cokoliv);
            Vcors[param_index_].set_values_unsafe(Vs[param_index_].cokoliv);

            Mcors[param_index_].scale(1./(1.-beta1_pow_t));
            Vcors[param_index_].scale(1./(1.-beta2_pow_t));

            Vcors[param_index_].inverse_square_root();

            Mcors[param_index_].scale(-this->learning_rate);
            Mcors[param_index_].ew_multiply_inplace(&Vcors[param_index_]);

            this->parameter_matrices_references[param_index_]->scale(1.-(weight_decay*this->learning_rate));
            this->parameter_matrices_references[param_index_]->add_inplace(&Mcors[param_index_]);
        }

        beta1_pow_t *= beta1;
        beta2_pow_t *= beta2;
    }

};

template class AdamW<float>;

#endif //NN_ADAMW_HPP
