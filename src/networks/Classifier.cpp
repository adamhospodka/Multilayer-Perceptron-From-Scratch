//
// Created by adamb on 07.10.2021.
//

#include "Classifier.h"

template<typename NumericType>
Classifier<NumericType>::Classifier(unsigned int seed, unsigned char errf): error_function(errf)  {
    this->gen = mt19937(seed);
}


template class Classifier<float>;