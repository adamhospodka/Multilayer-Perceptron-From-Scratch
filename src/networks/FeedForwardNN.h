//
// Created by adamb on 24.09.2021.
//

#ifndef NNLIB_FEEDFORWARDNN_H
#define NNLIB_FEEDFORWARDNN_H

#define OPTIMIZER_SGD 0
#define OPTIMIZER_Adam 1

#define ACTIVATION_ReLU 0
#define ACTIVATION_FastSigmoid 1
#define ACTIVATION_TanH 2
#define ACTIVATION_Sigmoid 3
#define ACTIVATION_SoftMax 4






#include "../Algebra/Matrix.h"
#include "Classifier.h"
#include "../optimizers/Optimizer.hpp"

#include <vector>
#include <memory>
using namespace std;
using namespace AlgebraLibrary;
/**
 * Basic FeedForward neural network.
 * @tparam NumericType The numeric type used for calculations. Can be float or double.
 */
template <typename NumericType>  // myšlenka je taková, že to budeme moct spouštět s floaty nebo doubly podle potřeby
class FeedForwardNN:public Classifier<NumericType> {
    private:
    const unsigned int layers_count;
        vector<Matrix<NumericType>> biases;
        vector<Matrix<NumericType>> weights;
        vector<Matrix<NumericType>> neuron_values;
        vector<Matrix<NumericType>> neuron_potentials;

        vector<Matrix<NumericType>> biases_grads;
        vector<Matrix<NumericType>> weights_grads;
        vector<Matrix<NumericType>> neuron_values_grads;
        vector<Matrix<NumericType>> helper_ew_values_grads_with_potentials;

        void init_required_matrices(const vector<int>& _layers);

    public:
        //float learning_rate = 0.05f;
        Optimizer<NumericType>* optimizer;
        vector<int> activation_functions;
        explicit FeedForwardNN(const vector<int> &layers, const vector<int> &activations, Optimizer<NumericType> *optimizer_ref,
                               unsigned char errf= ERRORF_MSE, unsigned seed= 24);

        //FeedForwardNN(vector<Matrix<NumericType>> weights);

    
    //void train_batch(int batch_size, vector<unique_ptr<Matrix<NumericType>>> &data_features,
    //                 vector<unique_ptr<Matrix<NumericType>>> &data_targets) override;
    //void predict(Matrix<NumericType> *input) override;
    //void backpropagate_remaining_examples_of_batch(Matrix<NumericType> *output);
    void backpropagate(Matrix<NumericType> *output);
    void compute_gradients_first_example_of_batch();
    void compute_gradients_remaining_examples_of_batch();
    void change_weights();





    /**
     * The idea is that we get an array of biases, which tells us how many neurons we have to add_inplace to the next layer,
     * which biases to use and how many weights we need to initialize internally to connect the new layer.
     * @param biases array of biases of the neurons in the added layer
     */
//        void add_layer(Matrix<NumericType>* biases);

        /**
         * This method replaces a layer on a specified index with a specified matrix (1xn Matrix) of biases.
         * Just change the target of the unique_ptr, memory management is taken care of my unique_ptr.
         * We might rewrite it later for the means of performance, but it seems to be an overkill right now.
         * @param layer_index
         * @param biases
         */
  //      void modify_layer(int layer_index, Matrix<NumericType>* biases);


        void init_random();

        void print_layer_setup(int i, int digits=2, int decimals=6);

        void print_layer_values(int i, int digits=2, int decimals=6);

        void print_layer_grads(int i, int digits=2, int decimals=6);

        void print_network(int digits=2, int decimals=6);

    const Matrix<NumericType> * predict(Matrix<NumericType> *input) override;

    void train_batch(int batch_size, vector<Matrix<NumericType>> &vector1,
                     vector<Matrix<NumericType>> &vector2) override;

    NumericType loss(Matrix<NumericType> &output) override;

    int test_prediction(Matrix<NumericType> &label);

    int maxValueIndex(const Matrix<NumericType> *matrix);
    
    int getPrediction();


    void normalize_gradients_by_batch_size(int batch_size);
};


#endif //NNLIB_FEEDFORWARDNN_H
