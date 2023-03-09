//
// Created by adamhospodka on 14.11.2021.
//

#define NNLIB_FEEDFORWARDNN_H

#include "../Algebra/Matrix.h"
#include "../networks/Classifier.h"

#include <vector>
#include <memory>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

// ------------------------------------------------------------

template <typename NumericType>
class fileHandler
{
public:
    fileHandler();
    vector<Matrix<NumericType>> load_features(const string& filename, int dimensions);
    vector<Matrix<NumericType>> load_labels(const string& filename, int dimensions);

    vector<Matrix<NumericType>> load_features_and_augment(const string& filename, int method, int dimensions);
    // Method 1: pixel shift, 2: noise
    
    vector<Matrix<NumericType>> load_labels_and_augment(const string& filename, int dimensions);
};
