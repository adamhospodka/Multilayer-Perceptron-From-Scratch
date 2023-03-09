//
// Created by adamb on 24.09.2021.
//

#include <iostream>
#include <random>
#include "FeedForwardNN.h"
#include "../optimizers/Adam.hpp"
#include "../optimizers/SGD.hpp"
#include <memory>
#include <cmath> // floor()
#define print(x){cout <<x<< endl;};
#define type(x){std::cout << typeid(x).name() << '\n';};


template<typename NumericType>
FeedForwardNN<NumericType>::FeedForwardNN(
        const vector<int> &_layers, const vector<int> &activations,
        Optimizer<NumericType> *optimizer_ref,
        unsigned char errf,
        unsigned seed
        ):
        Classifier<NumericType>(seed, errf), layers_count(_layers.size()),
        optimizer(optimizer_ref) {
    print("\nBonnie: *woke up*\n");
    
    this->init_required_matrices(_layers);
    this->activation_functions = activations;

    // Vector[0:layers], fill with matrises with dimensions given by network architecture
    int prev = 0; //number of neurons in layer
    for (int i:_layers) {
        neuron_values.emplace_back(1, i);
        if(prev != 0) {
            weights.emplace_back(prev, i);
            biases.emplace_back(1, i);
            neuron_potentials.emplace_back(1, i);

            weights_grads.emplace_back(prev, i);
            biases_grads.emplace_back(1, i);
            neuron_values_grads.emplace_back(1, i);
            helper_ew_values_grads_with_potentials.emplace_back(1, i);
        }
        prev = i;
    }
    /*if(optimizer_num == OPTIMIZER_Adam) {
        this->optimizer = new Adam<NumericType>(0.03f,0.9f,0.999f,0.00000001f);

    } else if(optimizer_num == OPTIMIZER_SGD) {
        this->optimizer = new SGD<NumericType>(0.03f);
    }*/
    optimizer->register_model_parameters(weights,weights_grads);
    optimizer->register_model_parameters(biases,biases_grads);
    optimizer->prepare_memory();

}


template<typename NumericType>
void FeedForwardNN<NumericType>::init_required_matrices(const vector<int>& _layers) {
    // Network attributes
    this->weights.reserve(this->layers_count-1); // W
    this->biases.reserve(this->layers_count-1); // B
    this->neuron_values.reserve(this->layers_count); // y
    this->neuron_potentials.reserve(this->layers_count-1); // ξ

    // Gradients
    this->weights_grads.reserve(this->layers_count-1); // ∇W
    this->biases_grads.reserve(this->layers_count-1); // ∇B
    this->neuron_values_grads.reserve(this->layers_count-1); // ∇y
    this->helper_ew_values_grads_with_potentials.reserve(this->layers_count-1);
}


// Method for forward feed in network
template<typename NumericType>
const Matrix<NumericType> * FeedForwardNN<NumericType>::predict(Matrix<NumericType>* input) {
    int lli = layers_count - 1;  // last layer index
    //Matrix<NumericType>* current = input;
    neuron_values[0].set_values(input);

    for (int i = 0; i < lli; ++i) {
        neuron_values[i].multiply(&weights[i], &neuron_potentials[i]);
        neuron_potentials[i].add_inplace(&biases[i]);
        switch (activation_functions[i]) {
            case ACTIVATION_FastSigmoid:
                neuron_potentials[i].SigmF(&neuron_values[i + 1]);
                break;

            case ACTIVATION_ReLU:
                neuron_potentials[i].ReLU(&neuron_values[i + 1]);
                break;
            case ACTIVATION_Sigmoid:
                neuron_potentials[i].Sigm(&neuron_values[i + 1]);
                break;
            case ACTIVATION_SoftMax:
                neuron_potentials[i].SoftMax(&neuron_values[i + 1]);
                break;
            default:
                throw invalid_argument("Untracked activation function.");
        }
    }

    //print("Prediction: ")
    //neuron_values[lli].print_matrix(2);
    return &neuron_values[lli];
}


template<typename NumericType>
void FeedForwardNN<NumericType>::backpropagate(Matrix<NumericType>* output) {
    // Calculate the ∇Y in the last layer
    switch(this->error_function) {
        case ERRORF_MSE:
            neuron_values[layers_count-1].sub(output,&neuron_values_grads[layers_count-2]);
            break;
        case ERRORF_BCE:
            if(this->activation_functions[layers_count-2] == ACTIVATION_SoftMax) {
                neuron_values[layers_count-1].sub(output,&helper_ew_values_grads_with_potentials[layers_count-2]);
                break;
            } else {
                throw runtime_error("BCE has t be used with SoftMax");
            }

    }

    // iterate through the hidden layers (skipping the first and the last)
    for (int i = layers_count - 2; i >= 1; --i) {
        // apply derivative of activation function to the potentials
        switch (activation_functions[i]) {
            case ACTIVATION_Sigmoid:
                neuron_potentials[i].set_values(1.f);
                neuron_potentials[i].sub_inplace(&neuron_values[i+1]);
                neuron_potentials[i].ew_multiply_inplace(&neuron_values[i+1]);
                break;

            case ACTIVATION_ReLU:
                neuron_potentials[i].ReLU_deriv();
                break;
            case ACTIVATION_FastSigmoid:
                neuron_potentials[i].SigmF_deriv();
                break;
            case ACTIVATION_SoftMax:
                if(i == layers_count-2) {
                    break;
                } else {
                    cout << "Y R U using SoftMax inside the network?";
                }
            default:
                throw invalid_argument("Untracked activation function.");
        }

        // Skip this step if SoftMax is Used
        if(activation_functions[i] != ACTIVATION_SoftMax) {
            // multiply elementwise the gradients in value with the activation function derivatives
            neuron_potentials[i].ew_multiply(
                    &neuron_values_grads[i],
                    &helper_ew_values_grads_with_potentials[i]
            );
        }
        // obtain the gradients in values for the previous layer by multiplying weights with values above
        weights[i].multiply_swapped_second(
                &helper_ew_values_grads_with_potentials[i],
                &neuron_values_grads[i - 1]);
    }
    // calculate values for the first layer separately, cuz we wont need gradients for the -1th layer
    switch (activation_functions[0]) {
        case ACTIVATION_Sigmoid:
            neuron_potentials[0].set_values(1.f);
            neuron_potentials[0].sub_inplace(&neuron_values[1]);
            neuron_potentials[0].ew_multiply_inplace(&neuron_values[1]);
            break;

        case ACTIVATION_FastSigmoid:
            neuron_potentials[0].SigmF_deriv();
            break;

        case ACTIVATION_ReLU:
            neuron_potentials[0].ReLU_deriv();
            break;
        case ACTIVATION_SoftMax:
            // skip haha? :D
            break;

        default:
            throw invalid_argument("Untracked activation function.");
    }
    // Skip this step if SoftMax is Used
    if(activation_functions[0] != ACTIVATION_SoftMax) {
        neuron_potentials[0].ew_multiply(
                &neuron_values_grads[0],
                &helper_ew_values_grads_with_potentials[0]
        );
    }
}


template<typename NumericType>
void FeedForwardNN<NumericType>::compute_gradients_first_example_of_batch() {
    for (int i = layers_count-2; i >= 0; --i) {
        neuron_values[i].multiply_swapped_first(
                &helper_ew_values_grads_with_potentials[i],
                &weights_grads[i]);
        // Here, the derivative of the activation function is already applied to the neuron potentials vector
        //neuron_values_grads[i].ew_multiply(
        //        &neuron_potentials[i],
        //        &biases_grads[i]);
        biases_grads[i].set_values(&helper_ew_values_grads_with_potentials[i]);
    }
}

template<typename NumericType>
void FeedForwardNN<NumericType>::compute_gradients_remaining_examples_of_batch()  {
    for (int i = layers_count-2; i >= 0; --i) {
        neuron_values[i].multiply_swapped_first_add(
                &helper_ew_values_grads_with_potentials[i],
                &weights_grads[i]);
        // Here, the derivative of the activation function is already applied to the neuron potentials vector
        //neuron_values_grads[i].ew_multiply_add(
        //        &neuron_potentials[i],
        //        &biases_grads[i]);
        biases_grads[i].add_inplace(&helper_ew_values_grads_with_potentials[i]);
    }
}


template<typename NumericType>
void FeedForwardNN<NumericType>::change_weights() {
    //cout << "Changing weights" <<endl;
    optimizer->modify_weights();

}

template<typename NumericType>
void FeedForwardNN<NumericType>::train_batch(int batch_size, vector<Matrix<NumericType>> &data_features,
                                             vector<Matrix<NumericType>> &data_targets) {


    uniform_int_distribution<unsigned int> dist(0, data_features.size());
    if(data_features.size() != data_targets.size()) {
        cout << "Bonnie: \"C'mon, fam... Your data and labels do not match in size!\"" << endl;
        cout << "Bonnie: \"*throws up*\"" << endl;
        throw runtime_error("Inputs are of different sizes. Bonnie just died, because of you!");
    }
    if(this->stochastic_mask.size() != data_features.size()) {
        this->stochastic_mask.resize(data_features.size());
        for (int i = 0; i < data_features.size(); ++i) {
            this->stochastic_mask[i] = i;
        }
    }
    std::shuffle(this->stochastic_mask.begin(), this->stochastic_mask.end(), this->gen);


    int remainder = data_features.size() % batch_size;
    int count = 0;
    float fourth_of_dataset = (data_features.size()/batch_size)/5;

    for (int batch_number = 0; batch_number < data_features.size() - remainder; batch_number += batch_size) {

        if (count == fourth_of_dataset){
            print("Bonnie: Learned " << 
                        FIXED_FLOAT( float(batch_number)/float(data_features.size()) * 100, 3, 1)  << " % of dataset ");
            count = 0;
        } else {
            count ++;
        }


        


        /*cout << "F" << batch_number * batch_size << ": " <<
        data_features[batch_number * batch_size] << " --> " <<
        data_targets[batch_number * batch_size] << endl;//*/
        this->predict(&data_features[this->stochastic_mask[batch_number]]);
        //this->print_network();
        this->backpropagate(&data_targets[this->stochastic_mask[batch_number]]);
        //this->print_network();
        this->compute_gradients_first_example_of_batch();
        //this->print_network();


        // REMAINING EXAMPLES OF BATCH
        for (int i = 1; i < batch_size; ++i) {
            //cout << this->stochastic_mask[batch_number + i] << " ";
            /*cout << "R" << batch_number * batch_size + i << ": " <<
            *data_features[batch_number * batch_size + i] << " --> " <<
            *data_targets[batch_number * batch_size + i] <<endl;//*/
            this->predict(&data_features[this->stochastic_mask[batch_number + i]]);
            //this->print_network();
            this->backpropagate(&data_targets[this->stochastic_mask[batch_number + i]]);
            this->compute_gradients_remaining_examples_of_batch();
        }
        normalize_gradients_by_batch_size(batch_size);

        this->change_weights();
    }
    //cout << endl;
}

template<typename NumericType>
void FeedForwardNN<NumericType>::normalize_gradients_by_batch_size(int batch_size) {
    //cout << "Normalidez layer by " << 1. / (float)batch_size << " ";
    for (int l = 0; l < layers_count - 1; ++l) {
        //cout << l ;
        weights_grads[l].scale(1. / (float)batch_size);
        biases_grads[l].scale(1. / (float)batch_size);
    }
}

template<typename NumericType>
NumericType FeedForwardNN<NumericType>::loss(Matrix<NumericType> &output) {
    Matrix<NumericType> error(this->neuron_values[layers_count-1]);
    switch (this->error_function) {
        case ERRORF_MSE:
            error.sub_inplace(&output);
            error.ew_multiply_inplace(&error);
            error.scale(0.5f);
            return error.sum();
        case ERRORF_BCE:
            if(this->activation_functions[layers_count-2] == ACTIVATION_SoftMax) {
                error.ln();
                error.ew_multiply_inplace(&output);
                return -error.sum();
            } else {
                throw runtime_error("For BCE you have to have SoftMax in the last layer");
            }

        default:
            throw runtime_error("Unrecognized error function.");
    }
}

template<typename NumericType>
int FeedForwardNN<NumericType>::getPrediction(){
    int prediction = maxValueIndex(&neuron_values[layers_count-1]);
    return prediction;
}

template<typename NumericType>
int FeedForwardNN<NumericType>::maxValueIndex(const Matrix<NumericType> *matrix){
    float maxValue = matrix->cokoliv[0];
    int indexMaxValue = 0;
    float currentValue;

    for (int i = 0; i < matrix->rows*matrix->cols; ++i){
        currentValue = matrix->cokoliv[i];
        if (currentValue > maxValue){
            maxValue = currentValue;
            indexMaxValue = i;
        }
        
    }
    return indexMaxValue;
}


template<typename NumericType>
int FeedForwardNN<NumericType>::test_prediction(Matrix<NumericType> &label) {

        int predicted_label = maxValueIndex(&neuron_values[layers_count-1]);
        int true_label = maxValueIndex(&label);
        
        if (predicted_label == true_label){
            return 1;
        } else {
            return 0;
        }


}










 
template<typename NumericType>
void FeedForwardNN<NumericType>::init_random() {
    for (int layer_ = 0; layer_ < this->layers_count - 1; ++layer_) {
        //cout << "Initializing layer" << layer_ << endl;
        switch (activation_functions[layer_]) {
            case ACTIVATION_FastSigmoid:
            case ACTIVATION_Sigmoid:
                this->weights[layer_].init_xavier();
                this->biases[layer_].init_xavier();
                break;
            case ACTIVATION_ReLU:
                this->weights[layer_].init_he();
                this->biases[layer_].init_he();
                break;
            case ACTIVATION_SoftMax:
                this->weights[layer_].init_xavier();
                break;
            default:
                throw runtime_error("Unrecognized activation function!");
        }


        //this->neuron_potentials[layer_].set_zero();
        //this->neuron_values_grads[layer_].set_zero();
        //this->helper_ew_values_grads_with_potentials[layer_].set_zero();
        //this->neuron_values[layer_].set_zero();
        //this->weights_grads[layer_].set_zero();
        //this->biases_grads[layer_].set_zero();
    }
    this->neuron_values[layers_count-1].set_zero();
}

template<typename NumericType>
void FeedForwardNN<NumericType>::print_layer_setup(int i, int digits, int decimals) {
    cout << "== weights and biases in layer " << i << " ==" << endl;
    this->weights[i-1].print_matrix(decimals, digits);
    print("---")
    this->biases[i-1].print_matrix(decimals, digits);
}

template<typename NumericType>
void FeedForwardNN<NumericType>::print_layer_values(int i, int digits, int decimals) {
    cout << "== calculated values, potentials, helpers in layer " << i << " ==" << endl;
    this->neuron_values[i].print_matrix(decimals, digits);
    print("---")
    this->neuron_potentials[i-1].print_matrix(decimals, digits);
    print("---")
    this->helper_ew_values_grads_with_potentials[i-1].print_matrix(decimals, digits);
}

template<typename NumericType>
void FeedForwardNN<NumericType>::print_layer_grads(int i, int digits, int decimals) {
    cout << "== values, weights and biases grads in layer " << i << " ==" << endl;
    this->neuron_values_grads[i-1].print_matrix(decimals, digits);
    print("---")
    this->weights_grads[i-1].print_matrix(decimals, digits);
    print("---")
    this->biases_grads[i-1].print_matrix(decimals, digits);
}

template<typename NumericType>
void FeedForwardNN<NumericType>::print_network(int digits, int decimals) {
    cout << "==== LAYER " << 0 << endl;
    cout << "== input values in layer " << 0 << " ==" << endl;
    this->neuron_values[0].print_matrix(decimals, digits);
    for (int layer_ = 1; layer_ < this->layers_count; ++layer_) {
        cout << "==== LAYER " << layer_ << endl;
        //print("-- setup")
        print_layer_setup(layer_, digits, decimals);
        //print("-- values")
        print_layer_values(layer_, digits, decimals);
        //print("-- grads")
        print_layer_grads(layer_, digits, decimals);
    }

}

template<class RandomIt, class URBG>
void shuffle(RandomIt first, RandomIt last, URBG&& g)
{
    typedef typename std::iterator_traits<RandomIt>::difference_type diff_t;
    typedef std::uniform_int_distribution<diff_t> distr_t;
    typedef typename distr_t::param_type param_t;

    distr_t D;
    diff_t n = last - first;
    for (diff_t i = n-1; i > 0; --i) {
        using std::swap;
        swap(first[i], first[D(g, param_t(0, i))]);
    }
}

template class FeedForwardNN<float>;



