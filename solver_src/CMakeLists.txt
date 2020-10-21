# cmake_minimum_required(VERSION <specify CMake version here>)
project(src)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(src
        cplex4-hb.h
        cplex4-hw.h
        cplex4-pz.h
        cplex4.h
        cplex5-pz.h
        cplex5.h
        datastructure.h
        io.h
        log_processor.cc
        middleman.cc
        util.h
        viterbi.h)
