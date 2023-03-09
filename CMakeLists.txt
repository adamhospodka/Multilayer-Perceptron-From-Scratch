cmake_minimum_required(VERSION 3.18)
project(NN)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX /openmp /O2 /Ot /Qpar")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

#add_subdirectory(Algebra)

#add_executable(NN_run main.cpp
#        networks/Classifier.cpp networks/Classifier.h
#        networks/FeedForwardNN.h networks/FeedForwardNN.cpp
#        Toolkit/fileHandler.h Toolkit/fileHandler.cpp
#        optimizers/Optimizer.hpp optimizers/Adam.hpp optimizers/SGD.hpp)
#
#target_include_directories(NN_run PUBLIC Algebra)
#target_link_libraries(NN_run PUBLIC Algebra_lib)
add_subdirectory(src)

#add_subdirectory(boost_tests)

#add_executable(fiddling fiddling.cpp)
