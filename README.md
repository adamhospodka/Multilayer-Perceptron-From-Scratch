469113
506521
# PV021 project | Deep Learning from Scratch

![pipline](https://gitlab.fi.muni.cz/xbajger/diy-nn/badges/master/pipeline.svg)
![build](https://gitlab.fi.muni.cz/xbajger/diy-nn/badges/master/build.svg)

## Authors
- Adam Hospodka `506521`
    [506521@fi.muni.cz](mailto:506521@fi.muni.cz)
- Adam Bajger `469113`
    [469113@fi.muni.cz](mailto:469113@fi.muni.cz


We state that the privided solution is **our own implementation** of Feed Forward Neural Network in the C++ language.

## Requirements

### Execution platform
Our solution is runnable on curret AISA server.

### Solution 
We are using backpropagation algorithm to train our network.

### Libraries
We **don't use any** high-level third-party libraries (math or other). We use some helper functions like `abs()` or `floor()` from `<cmath>` for input data checking and control prints.

### Performance
Our setup is able to achieve accuracy of 88% when traning on *Fashion MNIST Dataset* under 30 minutes

### Data handling
We split the trainng data file into train and validation set (train for training, validation for validation duh). We **only use test set vectors to predict the test labels**. Network outputs two requred prediction files `trainPredictions` and `actualTestPredictions` (correctly formated).

## Other info
- We use the recommended folder structure
- We provide the `RUN` script
- We do not shuffle test data
- We use templates for functions, but mainly use `double` num type
- We spiced up the solution with *Adam Optimizer*, *Weight Decay* and *Data Augmentation* 

## How to build on AISA

```bash
module add cmake-3.18.3
#module add gdb-8.3
mkdir build && cd build
cmake -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc ..
make

```

