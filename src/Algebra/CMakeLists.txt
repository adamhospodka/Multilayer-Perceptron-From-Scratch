project(Algebra)

set(HEADER_FILES
        Matrix.h
        )

set(SOURCE_FILES
        Matrix.cpp
        DimensionsIncompatibleException.hpp
        )

add_library(Algebra_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})

