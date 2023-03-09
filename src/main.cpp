//
// Created by adamb on 29.09.2021.
//

#include <string>
#include <iostream>
#include "Algebra/Matrix.h"
#include "Algebra/DimensionsIncompatibleException.hpp"
#include "networks/FeedForwardNN.h"
#include "Toolkit/fileHandler.h"
//#include "optimizers/Adam.hpp"
#include "optimizers/AdamW.hpp"
#include <ctime>

#define print(x){cout << x << endl;}

using namespace AlgebraLibrary;


float scheduled_learning_rate(unsigned int t) {
    const float LR = 0.01f;
    const float DF = 0.8;
    return LR * (pow(DF, (float)t) + ((sin((float)t) + 1.0f)/((float)t+1.0f)));
}



int main() {

    cout << endl
         << "// ============================= //" << endl
         << "      THE NEURAL NETWORK SHOW     "  << endl
         << "// ============================= //" << endl << endl
         << "           Characters:             " <<endl
         << "   -----------------------------   " << endl
         << " Bonnie      the Neural Networks   " << endl
         << " Dustin      the File Handler      " << endl
         << " Timekeeper  the Don't Burn AISA   " << endl
         << " Adam Wong   the Optimizer         " << endl << endl
         << "             The Scene:            " <<endl
         << "   -----------------------------   " << endl;






    // ======= TIMEKEEPER ======== //
    time_t start_time = time(nullptr);


    // ===== DUSTIN - CSV ====== //
    int data_dimensions = 784;
    int class_dimensions = 10;

    fileHandler<float> dustin = fileHandler<float>();

    // TRAIN DATA (ALL) -----------------------------------------

    vector<Matrix<float>> train_features_all = dustin.load_features(
        "../../data/fashion_mnist_train_vectors.csv", data_dimensions);

    vector<Matrix<float>> train_labels_all = dustin.load_labels(
        "../../data/fashion_mnist_train_labels.csv", class_dimensions);


    // TRAIN DATA (TRAIN) --------------------------------------

    auto const split = (unsigned long)((double)train_features_all.size() * 0.9);

    vector<Matrix<float>> train_features(
        train_features_all.begin(), train_features_all.begin() + split);

    vector<Matrix<float>> train_labels(
        train_labels_all.begin(), train_labels_all.begin() + split);


    // VALIDATION DATA (TRAIN) ---------------------------------

    vector<Matrix<float>> validation_features(
        train_features_all.begin() + split, train_features_all.end());

    vector<Matrix<float>> validation_labels(
        train_labels_all.begin() + split, train_labels_all.end());


    // TEST DATA -----------------------------------------------

    vector<Matrix<float>> test_features = dustin.load_features(
        "../../data/fashion_mnist_test_vectors.csv", data_dimensions);

    // --------------------------------------------------------




    // ===== BONNIE NN ====== //
    try {
        //Optimizer<float> *opt = new AdamW<float>(0.00048828125f, 0.95f, 0.999f, 0.087f);
        Optimizer<float> *opt = new AdamW<float>(0.004, 0.8f, 0.999f, 0.08f);
        vector<int> nn_schema = vector<int>({data_dimensions, 128, 10});
        vector<int> nn_activations = vector<int>({
            ACTIVATION_ReLU,
            ACTIVATION_SoftMax
        });
        FeedForwardNN<float> bonnie = FeedForwardNN<float>(nn_schema, nn_activations, opt, ERRORF_BCE);
        bonnie.init_random();
        //bonnie.optimizer->learning_rate = 0.003;


        bool continue_training = true;
        float epoch_accuracy;
        float max_accuracy = 0.0f;
        unsigned int epoch_count = 0;

        while (continue_training)
        {

            print("Bonnie: I'm learning!\n")
            print("Epoch: " << epoch_count << " LRate: " << FIXED_FLOAT(bonnie.optimizer->learning_rate,7,8))
            //bonnie.optimizer->learning_rate = scheduled_learning_rate(epoch_count);
            bonnie.train_batch(256, train_features, train_labels);

            print("Bonnie: Testing my accuracy!")
            int correctly_predicted = 0;
            float loss = 0;

            for (int i = 0; i < validation_features.size(); ++i) {
                bonnie.predict(&validation_features[i]);
                //int predicted_label = bonnie.getPrediction();
                correctly_predicted += bonnie.test_prediction(validation_labels[i]);
                loss += bonnie.loss(validation_labels[i]);
            }

            epoch_accuracy = (float)correctly_predicted / (float)validation_labels.size();
            print("Bonnie: My accuracy is " << FIXED_FLOAT(epoch_accuracy, 4, 6) << "/" << max_accuracy <<
            " (loss " << loss << ")")
            //print("Bonnie: The loss is " << loss)
            epoch_count += 1;

            if ( epoch_accuracy > max_accuracy ) {
                print("Dustin: We have new highest accuracy! ( " << epoch_accuracy << ")" )
                max_accuracy = epoch_accuracy;

                if (epoch_accuracy > 0.88f) {
                    int predicted_label;

                // Dump validation data
                    ofstream trainPredictions_dump("../../trainPredictions");
                    for (auto & train_feature : train_features_all) {
                        bonnie.predict(&train_feature);
                        predicted_label = bonnie.getPrediction();
                        trainPredictions_dump << predicted_label << endl;
                    }
                    trainPredictions_dump.close();
                    print("Dustin: I dumped train data predicitons (" << train_features_all.size() << ")" )

                    // Dump test data
                    ofstream actualTestPredictions_dump("../../actualTestPredictions");
                    for (auto & test_feature : test_features) {
                        bonnie.predict(&test_feature);
                        predicted_label = bonnie.getPrediction();
                        actualTestPredictions_dump << predicted_label << endl;
                    }
                    actualTestPredictions_dump.close();
                    print("Dustin: I dumped test data predicitons  (" << test_features.size() << ")")

                }
            }

            if (time(nullptr) - start_time > 1600) {   // 25 minutes - leaving some time overheads
                continue_training = false;
                print("Timekeeper: Time's up Mortal (" << time(nullptr) - start_time << "s)")
            }

        }
    }
    catch (DimensionsIncompatibleException<float> &dim_exc) {
        cout << "Exception " << dim_exc.what() << endl;
    }
    catch (runtime_error &exception) {
        cout << "Exception " << exception.what() << endl;
    }
    return 0;
}
