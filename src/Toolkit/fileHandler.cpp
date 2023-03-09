//
// Created by adamhospodka on 14.11.2021.
//


#include <iostream>
#include "fileHandler.h"
#include <cmath> // abs()

#define print(x){cout << x << endl;};
#define type(x){cout << typeid(x).name() << '\n';};

using namespace std;

template<typename NumericType>
fileHandler<NumericType>::fileHandler() {
    print("\nDustin: *woke up*\n");
}


template<typename NumericType>
vector<Matrix<NumericType>> fileHandler<NumericType>::load_features(const string& filename, int dimensions) {

    print("Dustin: Loading features!")

    vector<Matrix<NumericType>> feature_vectors_collection;
    Matrix feature_data_buffer = Matrix<float>(1, dimensions);
    feature_data_buffer.set_zero();


    ifstream soubor(filename, ifstream::in);

    // File offsets
    /*string content;
    soubor.seekg(0, ios::end);
    content.resize(soubor.tellg());
    soubor.seekg(0, ios::beg);
    soubor.read(&content[0], content.size());  // TODO: implementation defined behaviour - test on aisa
    soubor.close();

    // Initiate file stream
    stringstream file_stream(content);*/
    string row_as_string;
    string feature_value;

    // Stream contents of file by lines
    while (getline(soubor, row_as_string, '\n')) {
        stringstream row_stream(row_as_string);

        // Stream contents of row by characters
        int index = 0;
        while (getline(row_stream, feature_value, ',')){
            feature_data_buffer.cokoliv[index] = stof(feature_value);
            index++;
        }        
        feature_data_buffer.scale(1.f/255.f);
        feature_vectors_collection.push_back(feature_data_buffer);

        //feature_data_buffer.print_matrix(2);        
        feature_data_buffer.set_zero();

    }

    print("Dustin: Ready " << feature_vectors_collection.size() << " features");


    return feature_vectors_collection;
};


template<typename NumericType>
vector<Matrix<NumericType>> fileHandler<NumericType>::load_labels(const string& filename, int dimensions)
{

    print("Dustin: Loading labels!");

    vector<Matrix<NumericType>> feature_vectors_collection;
    Matrix feature_data_buffer = Matrix<float>(1, dimensions);
    feature_data_buffer.set_zero();


    ifstream soubor(filename, ifstream::in);

    /*// File offsets
    string content;
    soubor.seekg(0, ios::end);
    content.resize(soubor.tellg());
    soubor.seekg(0, ios::beg);
    soubor.read(&content[0], content.size());  // TODO: implementation defined behaviour - test on aisa
    soubor.close();

    // Initiate file stream
    stringstream file_stream(content);*/
    string row_as_string;
    string feature_value;



    // Stream contents of file by lines
    while (getline(soubor, row_as_string, '\n')) {
        stringstream row_stream(row_as_string);

        // Stream contents of row by characters
        int index;
        while (getline(row_stream, feature_value, ',')){
            // One-hot encode
            index = abs(stoi(feature_value));
            feature_data_buffer.cokoliv[index] = 1.00f;

        }
        feature_vectors_collection.push_back(feature_data_buffer);
        feature_data_buffer.cokoliv[index] = 0.f;
        //print("LABELS: ")
        //feature_data_buffer.print_matrix(2);        
        

    }

    print("Dustin: Loaded " << feature_vectors_collection.size() << " labels")


    return feature_vectors_collection;
}


// TODO
template<typename NumericType>
vector<Matrix<NumericType>> fileHandler<NumericType>::load_features_and_augment(const string& filename, int dimensions,  int method) {

    print("Dustin: Loading features and augmenting them!")

    vector<Matrix<NumericType>> feature_vectors_collection;
    Matrix feature_data_buffer = Matrix<float>(1, dimensions);
    Matrix feature_augmented_data_buffer = Matrix<float>(1, dimensions);


    ifstream soubor(filename, ifstream::in);

    /*// File offsets
    string content;
    soubor.seekg(0, ios::end);
    content.resize(soubor.tellg());
    soubor.seekg(0, ios::beg);
    soubor.read(&content[0], content.size());  // TODO: implementation defined behaviour - test on aisa
    soubor.close();

    // Initiate file stream
    stringstream file_stream(content);*/
    string row_as_string;
    string feature_value;

    // Stream contents of file by lines
    while (getline(soubor, row_as_string, '\n')) {
        stringstream row_stream(row_as_string);

        // Stream contents of row by characters //
        // Original features //
        int index = 0;
        float noise_threshold = 0.0f;
        while (getline(row_stream, feature_value, ',')){

            // Normal data //
            feature_data_buffer.cokoliv[index] = stof(feature_value);

            if (method == 1){
                // Augmented data - shift by one position to right //
                if (index >= 782) {
                    // dont jump out of the array
                    feature_augmented_data_buffer.cokoliv[index] = stof(feature_value);
                }
                else {
                    // random noise with integer modulo
                    feature_augmented_data_buffer.cokoliv[index + 2] = stof(feature_value);
                }
            }
            else if (method == 2){
                // Augmented data - add random noise turn //
                noise_threshold = (float)(rand() % 100) / 300.f;

                if (stof(feature_value) > noise_threshold){
                    feature_augmented_data_buffer.cokoliv[index] = 0.f;
                }
            }
            else if (method == 3){
                // Augmented data - add random noise //
                noise_threshold = (float)(rand() % 100) / 100.f;
                //print(noise_threshold);

                    feature_augmented_data_buffer.cokoliv[index] = stof(feature_value);
                    if (noise_threshold < 0.1){
                        feature_augmented_data_buffer.cokoliv[index] = 255.f;
                    }
                    else
                    {
                        feature_augmented_data_buffer.cokoliv[index] = stof(feature_value);
                    }

            index++;
          }
        }        
        feature_data_buffer.scale(1.f/255.f);
        feature_augmented_data_buffer.scale(1.f/255.f);

        //feature_augmented_data_buffer.draw_image();

        feature_vectors_collection.push_back(feature_data_buffer);
        feature_vectors_collection.push_back(feature_augmented_data_buffer);
     
        feature_data_buffer.set_zero();
        feature_augmented_data_buffer.set_zero();

    }

    print("Dustin: Loaded " << feature_vectors_collection.size() << " features");


    return feature_vectors_collection;
};



template<typename NumericType>
vector<Matrix<NumericType>> fileHandler<NumericType>::load_labels_and_augment(const string& filename, int dimensions)
{

    print("Dustin: Loading labels and extending for augmented data!");

    vector<Matrix<NumericType>> feature_vectors_collection;
    Matrix feature_data_buffer = Matrix<float>(1, dimensions);
    feature_data_buffer.set_zero();


    ifstream soubor(filename, ifstream::in);

    /*// File offsets
    string content;
    soubor.seekg(0, ios::end);
    content.resize(soubor.tellg());
    soubor.seekg(0, ios::beg);
    soubor.read(&content[0], content.size());  
    soubor.close();

    // Initiate file stream
    stringstream file_stream(content);*/
    string row_as_string;
    string feature_value;



    // Stream contents of file by lines
    while (getline(soubor, row_as_string, '\n')) {
        stringstream row_stream(row_as_string);

        // Stream contents of row by characters

        // Load actual data
        int index;
        while (getline(row_stream, feature_value, ',')){
            // One-hot encode
            index = abs(stoi(feature_value));
            feature_data_buffer.cokoliv[index] = 1.00f;

        }
        // Original label
        feature_vectors_collection.push_back(feature_data_buffer);
        // Label for augmented data
        feature_vectors_collection.push_back(feature_data_buffer);

        feature_data_buffer.cokoliv[index] = 0.f;         
        

    }

    print("Dustin: Ready " << feature_vectors_collection.size() << " labels")


    return feature_vectors_collection;
}


template class fileHandler<float>;