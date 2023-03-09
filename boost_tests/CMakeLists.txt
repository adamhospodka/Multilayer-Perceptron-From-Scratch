#set (Boost_USE_STATIC_LIBS OFF)


# adambajger
# ----------------
set (BOOST_ROOT A:/Windows/Programy/boost_1_77_0)
#SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "A:/Windows/Programy/boost_1_77_0")
#SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "A:/Windows/Programy/boost_1_77_0/boost_static_version/lib")
# ----------------

# adamhospodka
# ----------------
SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "/Applications/boost_lib_cpp/boost_1_77_0")
# ----------------

find_package (Boost REQUIRED COMPONENTS unit_test_framework)
if (Boost_FOUND)
    include_directories (${Boost_INCLUDE_DIRS})

    add_executable(Boost_Tests_run MatrixMultiplicationTest.cpp MatrixAdditionSubtractionTest.cpp
            MatrixMultiplicationAdditionTest.cpp MatrixDataAlignment.cpp MatrixComposedOperationsTest.cpp
            MatrixElementwiseOperationsTest.cpp)


    target_link_libraries (Boost_Tests_run ${Boost_LIBRARIES})
    target_link_libraries (Boost_Tests_run Algebra_lib)
endif ()